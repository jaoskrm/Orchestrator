from pathlib import Path
from typing import Any, Dict
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
    """Remove markdown code fences. FIX: properly handle string split."""
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.rstrip().endswith("```"):
            s = s.rsplit("```", 1)[0]
    return s.strip()


def _extract_test_section(code: str) -> tuple[str, str]:
    """Split using explicit markers for reliability. FIX: prioritize markers over regex."""
    # Primary: explicit markers
    if "# === TESTS ===" in code:
        parts = code.split("# === TESTS ===", 1)
        # FIX: parts is a list, need to index it
        impl = parts[0]
        # FIX: Strip marker more defensively with regex
        impl = re.sub(r'^# === IMPLEMENTATION ===\s*', '', impl, flags=re.MULTILINE).strip()
        tests = parts[1].strip()
        return impl, tests
    
    # Fallback: detect pytest import (but avoid docstrings)
    match = re.search(r'^import pytest$', code, re.MULTILINE)
    if match:
        impl = code[:match.start()].strip()
        tests = code[match.start():].strip()
        return impl, tests
    
    # No tests found
    return code.strip(), ""


def _clean_tests(tests: str) -> str:
    """Remove duplicate imports and entrypoints. FIX: avoid double pytest.main()."""
    # Remove standalone import pytest (we add it ourselves)
    tests = re.sub(r'^import pytest\s*$', '', tests, flags=re.MULTILINE).strip()
    
    # Remove any existing if __name__ blocks (we add our own)
    # FIX: Simpler regex - nuke everything after if __name__
    tests = re.sub(
        r"if __name__\s*==\s*['\"]__main__['\"]:\s*[\s\S]*$",
        "",
        tests,
    ).strip()
    
    return tests


def _validate_tests(tests: str) -> bool:
    """Ensure at least 3 test functions exist. FIX: structural validation.
    
    Note: This can be gamed with empty test_ functions that just pass.
    For stronger guarantees, use AST inspection in the future.
    """
    test_funcs = re.findall(r'^\s*def (test_\w+)\(', tests, re.MULTILINE)
    has_pytest = "pytest" in tests
    return len(test_funcs) >= 3 and has_pytest


def _write_full_code(task_dir: Path, implementation: str, tests: str) -> None:
    """Write self-contained main.py. FIX: clean tests before writing."""
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


def run_coding_workflow(task_id: str, user_prompt: str, decision: Dict[str, Any]) -> Dict[str, Any]:
    task_dir = Path("runs") / task_id
    sb = DockerSandbox(container_name="agent-sandbox")
    tracer = TraceWriter(task_id, "coding", user_prompt, decision)

    workers = decision.get("workers", [])
    solver = next((w for w in workers if w.get("role") == "solver"), None)
    verifier = next((w for w in workers if w.get("role") in ("verifier", "critic")), None)
    if not solver:
        raise ValueError("coding workflow requires a solver worker")

    max_rounds = int(decision.get("controls", {}).get("max_rounds", 3))

    last_err = ""
    final_stdout = ""
    passed = False

    for r in range(1, max_rounds + 1):
        tracer.log("round_start", round=r)

        # FIX: examples only in round 1, concise prompt after
        if r == 1:
            example_section = f"\nEXAMPLE FORMAT:\n{CODING_EXAMPLE}\n"
        else:
            example_section = ""

        solver_prompt = f"""Write COMPLETE Python with pytest tests.

FORMAT (strict):
# === IMPLEMENTATION ===
[your code]

# === TESTS ===
[test functions - names MUST start with test_]

RULES:
- Minimum 3 test functions covering public API
- Test edge cases + errors with pytest.raises()
- Use type hints and docstrings

{example_section}

TASK: {user_prompt}

PREVIOUS FAILURE (FIX THIS):
{last_err[-1500:] if last_err else 'None'}

OUTPUT: Executable Python only. No markdown."""

        full_code = _call_worker(solver, solver_prompt, temperature=0.1 if r > 1 else 0.2)
        full_code = strip_code_fences(full_code)
        
        tracer.log("solver_full_code", round=r, model=solver["model"], chars=len(full_code))
        
        implementation, tests = _extract_test_section(full_code)
        
        # FIX: Fail fast on empty implementation
        if not implementation.strip():
            tracer.log("implementation_validation_fail", reason="Empty implementation generated")
            last_err = "ERROR: Empty implementation generated"
            continue
        
        # FIX: validate structure before writing
        test_count = len(re.findall(r'^\s*def (test_\w+)\(', tests, re.MULTILINE))
        
        # FIX: Log test count for observability
        tracer.log("test_validation", test_count=test_count, has_pytest="pytest" in tests)
        
        if not _validate_tests(tests):
            tracer.log("test_validation_fail", reason="<3 test functions found or missing pytest import")
            last_err = "ERROR: Generated code has insufficient tests (need >=3 test_ functions with pytest)"
            continue
        
        _write_full_code(task_dir, implementation, tests)

        # Verifier review (optional)
        if verifier:
            verify_prompt = f"Review for bugs:\n{full_code[:2000]}"
            review = _call_worker(verifier, verify_prompt, temperature=0.1)
            tracer.log("verifier_review", round=r, model=verifier["model"], review=review[:800])

        # Docker sandbox
        sb.reset_task_dir()
        sb.copy_task_in(task_dir)
        
        # FIX: verify file exists locally before running pytest
        # Note: This checks host FS, not container FS. If Docker copy issues arise, 
        # this is the first place to investigate.
        if not (task_dir / "main.py").exists():
            tracer.log("container_verify", error="main.py not found in task_dir")
            last_err = "ERROR: main.py was not written successfully"
            continue
        
        # FIX: Explicitly tell pytest to run main.py (pytest doesn't auto-discover it)
        res = sb.pytest("main.py")
        
        # FIX: capture both head and tail of output
        stdout_head = res.stdout[:2000] if len(res.stdout) > 2000 else res.stdout
        stdout_tail = res.stdout[-2000:] if len(res.stdout) > 2000 else ""
        
        tracer.log("pytest", round=r, exit_code=res.exit_code, 
                  stdout_head=stdout_head, stdout_tail=stdout_tail,
                  stderr=res.stderr[-2000:])

        final_stdout = res.stdout
        last_err = f"{res.stdout[:1500]}\n...\n{res.stderr[-1500:]}"

        if res.exit_code == 0:
            passed = True
            tracer.log("pytest_pass", summary=final_stdout[-300:])
            break
        else:
            tracer.log("pytest_fail", error_head=last_err[:500])

    tracer.set_result(success=passed, final_answer=final_stdout[-800:] if passed else last_err[:800])
    trace_path = tracer.flush()

    return {
        "task_id": task_id,
        "passed": passed,
        "stdout": final_stdout,
        "pytest_exit_code": res.exit_code if 'res' in locals() else -1,
        "trace": str(trace_path),
    }