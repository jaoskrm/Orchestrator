from pathlib import Path
from typing import Any, Dict

from orchestrator.llm_clients import OllamaClient
from orchestrator.runner.docker_runner import DockerSandbox
from orchestrator.storage.traces import TraceWriter

def _call_worker(worker: Dict[str, Any], prompt: str, temperature: float = 0.2) -> str:
    provider = worker["provider"]
    model = worker["model"]
    if provider == "ollama":
        return OllamaClient().generate(model=model, prompt=prompt, temperature=temperature)
    raise ValueError(f"Unsupported provider: {provider}")

def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # drop first line like ``````python
        s = s.split("\n", 1)[1] if "\n" in s else ""
        # drop the last closing fence
        if s.rstrip().endswith("```"):
            s = s.rsplit("```", 1)[0]
    return s.strip()

def _write_main_py(task_dir: Path, code: str) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    target = (task_dir / "main.py").resolve()
    if task_dir.resolve() not in target.parents:
        raise ValueError("Refusing to write outside task_dir")
    target.write_text(strip_code_fences(code) + "\n", encoding="utf-8")



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

    # Ensure tests exist (v0 expects tests provided by you/dataset)
    tests_dir = task_dir / "tests"
    if not tests_dir.exists():
        raise FileNotFoundError(f"Missing tests/ in {tests_dir}. For v0, provide pytest tests.")

    last_err = ""
    final_stdout = ""
    passed = False

    for r in range(1, max_rounds + 1):
        tracer.log("round_start", round=r)

        solver_prompt = (
            "You are a coding agent.\n"
            "Write ONLY Python code for main.py.\n"
            "Do NOT modify tests.\n"
            "No markdown.\n\n"
            f"Task:\n{user_prompt}\n\n"
            f"Pytest failure:\n{last_err}\n"
            )

        code = _call_worker(solver, solver_prompt, temperature=0.2)
        tracer.log("solver_output", round=r, model=solver["model"], chars=len(code))
        _write_main_py(task_dir, code)

        # Optional verifier: ask for quick review, but still trust pytest
        if verifier:
            verify_prompt = (
                "You are a code reviewer. Identify likely bugs/edge cases in this code.\n\n"
                f"Task:\n{user_prompt}\n\n"
                f"Code:\n{code}\n"
            )
            review = _call_worker(verifier, verify_prompt, temperature=0.1)
            tracer.log("verifier_review", round=r, model=verifier["model"], review=review[:1500])

        # Run pytest in docker
        sb.reset_task_dir()
        sb.copy_task_in(task_dir)
        res = sb.pytest()
        tracer.log("pytest", round=r, exit_code=res.exit_code, stdout=res.stdout[-2000:], stderr=res.stderr[-2000:])

        final_stdout = res.stdout
        last_err = (res.stdout + "\n" + res.stderr)[-4000:]


        if res.exit_code == 0:
            passed = True
            break

    tracer.set_result(success=passed, final_answer="PASSED" if passed else "FAILED")
    trace_path = tracer.flush()

    return {
        "task_id": task_id,
        "passed": passed,
        "stdout": final_stdout,
        "trace": str(trace_path),
    }
