from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import ast
import re

from orchestrator.llm_clients import OllamaClient
from orchestrator.runner.docker_runner import DockerSandbox
from orchestrator.storage.traces import TraceWriter

# === EXAMPLES (shown only in round 1) ===
CODING_EXAMPLE = """# === IMPLEMENTATION ===
class Calculator:
    def add(self, a: int, b: int) -> int:
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("Inputs must be integers")
        return a + b

# === TESTS ===
import pytest

def test_add_valid():
    calc = Calculator()
    assert calc.add(2, 3) == 5

def test_add_error():
    calc = Calculator()
    with pytest.raises(TypeError):
        calc.add("2", 3)
"""


def _call_worker(worker: Dict[str, Any], prompt: str, temperature: float = 0.2) -> str:
    provider = worker["provider"]
    model = worker["model"]
    if provider == "ollama":
        return OllamaClient().generate(model=model, prompt=prompt, temperature=temperature)
    raise ValueError(f"Unsupported provider: {provider}")


def strip_code_fences(s: str) -> str:
    """Remove markdown code fences. FIX: Corrected string handling and indexing."""
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
    if s.rstrip().endswith("```"):
        s = s.rsplit("```", 1)[0]
    return s.strip()


def _extract_test_section(code: str) -> tuple[str, str]:
    """Split using explicit markers for reliability. FIX: Corrected list indexing."""
    if "# === TESTS ===" in code:
        parts = code.split("# === TESTS ===", 1)
        impl = parts[0]
        impl = re.sub(r"^# === IMPLEMENTATION ===\s*", "", impl, flags=re.MULTILINE).strip()
        tests = parts[1].strip()
        return impl, tests

    match = re.search(r"^import pytest$", code, re.MULTILINE)
    if match:
        impl = code[:match.start()].strip()
        tests = code[match.start():].strip()
        return impl, tests

    return code.strip(), ""


def _clean_tests(tests: str) -> str:
    """Remove duplicate imports and entrypoints."""
    tests = re.sub(r"^import pytest\s*$", "", (tests or ""), flags=re.MULTILINE).strip()
    tests = re.sub(
        r"if __name__\s*==\s*['\"]__main__['\"]:\s*[\s\S]*$",
        "",
        tests,
    ).strip()
    return tests


def _validate_tests(tests: str) -> bool:
    """Ensure at least 3 test functions exist."""
    test_funcs = re.findall(r"^\s*def (test_\w+)\(", tests or "", re.MULTILINE)
    has_pytest = "pytest" in (tests or "")
    return len(test_funcs) >= 3 and has_pytest


def _write_full_code(task_dir: Path, implementation: str, tests: str) -> None:
    """Write self-contained main.py with pytest.ini for discovery."""
    task_dir.mkdir(parents=True, exist_ok=True)
    target = (task_dir / "main.py").resolve()
    if task_dir.resolve() not in target.parents:
        raise ValueError("Refusing to write outside task_dir")

    tests_clean = _clean_tests(tests)
    full_code = f"""# Auto-generated: Implementation + Tests
{implementation}

# === TESTS ===
import pytest

{tests_clean}

if __name__ == '__main__':
    pytest.main(['-v', '--tb=short', __file__])
"""
    target.write_text(full_code, encoding="utf-8")
    
    # Write pytest.ini so pytest discovers main.py
    pytest_ini = """[pytest]
python_files = main.py
python_functions = test_*
python_classes = Test*
"""
    (task_dir / "pytest.ini").write_text(pytest_ini, encoding="utf-8")


def _format_lock_reject(reason: str) -> Dict[str, Any]:
    return {"ok": False, "reason": reason}


def _format_lock_check_python(src: str) -> Dict[str, Any]:
    """Hard format lock: produced code must parse as Python."""
    try:
        ast.parse(src)
        return {"ok": True}
    except SyntaxError as e:
        return _format_lock_reject(f"SyntaxError: {e.msg} (line {e.lineno})")
    except Exception as e:
        return _format_lock_reject(f"ParseError: {type(e).__name__}: {e}")


def _has_markdown_artifacts(s: str) -> bool:
    """Hard format lock: forbid markdown fences."""
    return "```" in (s or "")


@dataclass
class FailureInfo:
    failure_class: str  # FORMAT | CONTRACT | RUNTIME_SAFETY | LOGIC
    summary: str


def _classify_failure(stdout: str, stderr: str) -> FailureInfo:
    out = (stdout or "") + "\n" + (stderr or "")
    out_lower = out.lower()

    # FORMAT / collection errors
    format_markers = [
        "syntaxerror",
        "indentationerror",
        "taberror",
        "importerror",
        "modulenotfounderror",
        "error collecting",
        "collected 0 items / 1 error",
        "usage: pytest",
    ]
    if any(m in out_lower for m in format_markers):
        return FailureInfo("FORMAT", "pytest collection/import/syntax failure")

    # CONTRACT
    contract_markers = [
        "typeerror",
        "got an unexpected keyword argument",
        "missing 1 required positional argument",
        "takes ",
        "but ",
        "was given",
        "valueerror",
        "assert isinstance",
    ]
    if "assertionerror" not in out_lower and any(m in out_lower for m in contract_markers):
        return FailureInfo("CONTRACT", "API/validation mismatch (non-assertion failure)")

    # RUNTIME SAFETY (crashes)
    runtime_markers = [
        "attributeerror",
        "keyerror",
        "indexerror",
        "zerodivisionerror",
        "nonetype",
        "unboundlocalerror",
        "overflowerror",
    ]
    if any(m in out_lower for m in runtime_markers):
        return FailureInfo("RUNTIME_SAFETY", "runtime exception/crash")

    # LOGIC (assertions)
    if "assertionerror" in out_lower or "assert " in out_lower:
        return FailureInfo("LOGIC", "assertion failed (wrong result / behavior)")

    # Default bucket
    return FailureInfo("LOGIC", "unclassified failure (default to logic)")


def _build_fullgen_prompt(
    user_prompt: str, last_err: str, include_example: bool
) -> str:
    example_section = f"\nEXAMPLE FORMAT:\n{CODING_EXAMPLE}\n" if include_example else ""
    return f"""Write COMPLETE Python with pytest tests.

FORMAT (strict):
# === IMPLEMENTATION ===
[your code]

# === TESTS ===
[test functions - names MUST start with test_]

RULES:
- Minimum 3 test functions covering public API
- Test edge cases + errors with pytest.raises()
- Use type hints and docstrings
- Do NOT output markdown. Do NOT output triple backticks.

{example_section}

TASK:
{user_prompt}

PREVIOUS FAILURE (FIX THIS):
{last_err[-2000:] if last_err else "None"}

OUTPUT: Executable Python only (no markdown, no commentary).
"""


def _build_implonly_prompt(
    user_prompt: str,
    last_err: str,
    frozen_tests: str,
    previous_impl: str,
    failure_class: str,
) -> str:
    # Heuristic: If it's a runtime crash (traceback), focus on that specific error.
    # If it's an assertion error, show the diff.
    
    focus_instruction = ""
    if failure_class == "RUNTIME_SAFETY":
        focus_instruction = "FOCUS: Fix the runtime crash/exception shown in the traceback. Add guards for edge cases (e.g., empty lists, None values)."
    elif failure_class == "CONTRACT":
        focus_instruction = "FOCUS: Fix the API contract violation. Ensure exceptions (TypeError, ValueError) are raised exactly as requested."
    elif failure_class == "LOGIC":
        focus_instruction = "FOCUS: Fix the logic error. Compare your output with the expected output in the failure message."
        
    return f"""You are repairing ONLY the implementation to satisfy the frozen pytest tests.

HARD RULES:
- Output ONLY Python code for the implementation (no tests).
- Do NOT output markdown. Do NOT output triple backticks.
- Do NOT include "# === TESTS ===".
- Keep public API stable unless the error clearly demands a signature fix.

TASK:
{user_prompt}

FAILURE CLASS:
{failure_class}

{focus_instruction}

PYTEST FAILURE (fix this):
{last_err[-2500:] if last_err else "None"}

FROZEN TESTS (do not modify; for reference only):
{frozen_tests[:2000]}

PREVIOUS IMPLEMENTATION (for reference):
{previous_impl[:3000]}

OUTPUT: Implementation-only Python code.
"""




def run_coding_workflow(task_id: str, user_prompt: str, decision: Dict[str, Any]) -> Dict[str, Any]:
    task_dir = Path("runs") / task_id
    sb = DockerSandbox(container_name="agent-sandbox")
    tracer = TraceWriter(task_id, "coding", user_prompt, decision)

    workers = decision.get("workers", []) or []
    solver = next((w for w in workers if w.get("role") == "solver"), None)
    verifier = next((w for w in workers if w.get("role") in ("verifier", "critic")), None)

    if not solver:
        raise ValueError("coding workflow requires a solver worker")

    controls = decision.get("controls", {}) or {}
    max_rounds = int(controls.get("max_rounds", 3))
    max_rounds = max(1, min(max_rounds, 8))

    last_err = ""
    final_stdout = ""
    passed = False
    final_exit_code = -1

    # Phase A artifact state (in-memory)
    frozen_tests: Optional[str] = None
    current_impl: str = ""
    need_full_regen: bool = True
    prev_failure_class: str = "FORMAT"
    full_regens_used = 0
    impl_repairs_used = 0

    for r in range(1, max_rounds + 1):
        tracer.log(
            "round_start",
            round=r,
            need_full_regen=need_full_regen,
            full_regens_used=full_regens_used,
            impl_repairs_used=impl_repairs_used,
        )

        # ====== Decide generation mode ======
        if need_full_regen or frozen_tests is None:
            mode = "full_regen"
            full_regens_used += 1

            solver_prompt = _build_fullgen_prompt(
                user_prompt=user_prompt,
                last_err=last_err,
                include_example=(r == 1),
            )
            raw = _call_worker(solver, solver_prompt, temperature=0.2 if r == 1 else 0.1)
            raw = strip_code_fences(raw)

            tracer.log("solver_full_code", round=r, model=solver["model"], chars=len(raw))

            # Format locks (hard)
            if _has_markdown_artifacts(raw):
                last_err = "FORMAT_LOCK: Detected markdown fence ```"
                tracer.log("format_lock_fail", round=r, reason=last_err)
                need_full_regen = True
                continue

            implementation, tests = _extract_test_section(raw)

            if not (implementation or "").strip():
                last_err = "FORMAT_LOCK: Empty implementation generated."
                tracer.log("format_lock_fail", round=r, reason=last_err)
                need_full_regen = True
                continue

            if not _validate_tests(tests or ""):
                test_count = len(re.findall(r"^\s*def (test_\w+)\(", tests or "", re.MULTILINE))
                last_err = f"FORMAT_LOCK: Insufficient tests (found {test_count}; need >=3 and pytest)."
                tracer.log("test_validation_fail", round=r, reason=last_err, test_count=test_count)
                need_full_regen = True
                continue

            # Hard parse lock for full file we will execute
            check_full = _format_lock_check_python(
                f"{implementation}\n\n# === TESTS ===\n\nimport pytest\n\n{_clean_tests(tests)}\n"
            )
            if not check_full.get("ok"):
                last_err = f"FORMAT_LOCK: {check_full.get('reason')}"
                tracer.log("format_lock_fail", round=r, reason=last_err)
                need_full_regen = True
                continue

            # Freeze tests and move forward
            frozen_tests = tests
            current_impl = implementation.strip()
            need_full_regen = False

        else:
            mode = "impl_only"
            impl_repairs_used += 1

            solver_prompt = _build_implonly_prompt(
                user_prompt=user_prompt,
                last_err=last_err,
                frozen_tests=frozen_tests,
                previous_impl=current_impl,
                failure_class=prev_failure_class,
            )
            raw_impl = _call_worker(solver, solver_prompt, temperature=0.1)
            raw_impl = strip_code_fences(raw_impl)

            tracer.log("solver_impl_only", round=r, model=solver["model"], chars=len(raw_impl))

            # Format locks (hard)
            if _has_markdown_artifacts(raw_impl):
                last_err = "FORMAT_LOCK: Detected markdown fence ``` in implementation-only output."
                tracer.log("format_lock_fail", round=r, reason=last_err)
                need_full_regen = True
                continue

            if "# === TESTS ===" in raw_impl:
                last_err = "FORMAT_LOCK: Implementation-only output contained '# === TESTS ==='."
                tracer.log("format_lock_fail", round=r, reason=last_err)
                need_full_regen = True
                continue

            # Some models might include the IMPLEMENTATION marker; strip it.
            raw_impl = re.sub(r"^# === IMPLEMENTATION ===\s*", "", raw_impl, flags=re.MULTILINE).strip()

            if not raw_impl.strip():
                last_err = "FORMAT_LOCK: Empty implementation-only output."
                tracer.log("format_lock_fail", round=r, reason=last_err)
                need_full_regen = True
                continue

            # Parse lock for implementation
            check_impl = _format_lock_check_python(raw_impl)
            if not check_impl.get("ok"):
                last_err = f"FORMAT_LOCK: {check_impl.get('reason')}"
                tracer.log("format_lock_fail", round=r, reason=last_err)
                need_full_regen = True
                continue

            current_impl = raw_impl.strip()

        tracer.log("generation_mode", round=r, mode=mode)

        # Optional verifier review
        if verifier:
            preview = current_impl
            if frozen_tests:
                preview = f"{current_impl}\n\n# === TESTS (frozen preview) ===\n{frozen_tests}"
            review = _call_worker(verifier, f"Review for bugs:\n{preview[:2000]}", temperature=0.1)
            tracer.log("verifier_review", round=r, model=verifier["model"], review=review[:800])

        # ====== Write + run pytest ======
        _write_full_code(task_dir, current_impl, frozen_tests or "")

        sb.reset_task_dir()
        sb.copy_task_in(task_dir)

        if not (task_dir / "main.py").exists():
            last_err = "ERROR: main.py was not written successfully"
            tracer.log("container_verify", round=r, error=last_err)
            need_full_regen = True
            continue

        res = sb.pytest()

        stdout_head = res.stdout[:2000] if len(res.stdout or "") > 2000 else (res.stdout or "")
        stdout_tail = (res.stdout or "")[-2000:] if len(res.stdout or "") > 2000 else ""
        stderr_tail = (res.stderr or "")[-2000:]

        tracer.log(
            "pytest",
            round=r,
            exit_code=res.exit_code,
            stdout_head=stdout_head,
            stdout_tail=stdout_tail,
            stderr=stderr_tail,
        )

        final_stdout = res.stdout or ""
        final_exit_code = res.exit_code
        last_err = f"{(res.stdout or '')[:1500]}\n...\n{(res.stderr or '')[-1500:]}"

        if res.exit_code == 0:
            passed = True
            tracer.log("pytest_pass", round=r, summary=final_stdout[-300:])
            break

        # ====== Failure classification + policy ======
        finfo = _classify_failure(res.stdout or "", res.stderr or "")
        prev_failure_class = finfo.failure_class

        tracer.log(
            "failure_classification",
            round=r,
            failure_class=finfo.failure_class,
            summary=finfo.summary,
        )

        # Patch vs regenerate decision:
        # - FORMAT => full regen next round
        # - Others => impl-only repair (tests frozen)
        if finfo.failure_class == "FORMAT":
            need_full_regen = True
        else:
            need_full_regen = False

        # Escalation safety: if too many impl-only attempts, try full regen
        if impl_repairs_used >= 3 and not passed:
            tracer.log("escalate_full_regen", round=r, reason="impl_repairs_used>=3")
            need_full_regen = True

    tracer.set_result(success=passed, final_answer=final_stdout[-800:] if passed else last_err[:800])
    trace_path = tracer.flush()

    return {
        "task_id": task_id,
        "passed": passed,
        "stdout": final_stdout,
        "pytest_exit_code": final_exit_code,  # FIX: Return real exit code
        "trace": str(trace_path),
    }