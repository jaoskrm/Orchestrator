from typing import Any, Dict
from orchestrator.llm_clients import OllamaClient
from orchestrator.storage.traces import TraceWriter

def _call(worker: Dict[str, Any], prompt: str, temperature: float = 0.2) -> str:
    if worker["provider"] == "ollama":
        return OllamaClient().generate(model=worker["model"], prompt=prompt, temperature=temperature)
    raise ValueError(f"Unsupported provider: {worker['provider']}")

def run_science_workflow(task_id: str, user_prompt: str, decision: Dict[str, Any]) -> Dict[str, Any]:
    tracer = TraceWriter(task_id, "science", user_prompt, decision)

    workers = decision.get("workers", [])
    solver = next((w for w in workers if w.get("role") == "solver"), None)
    verifier = next((w for w in workers if w.get("role") in ("verifier", "critic")), None)
    if not solver:
        raise ValueError("science workflow requires a solver worker")

    solve_prompt = (
        "You are a science problem solver (math/physics/chem).\n"
        "Format:\nGiven:\nUnknown:\nSteps:\nUnits check:\nFinal answer:\n\n"
        f"Problem:\n{user_prompt}\n"
    )
    draft = _call(solver, solve_prompt, temperature=0.2)
    tracer.log("solver_draft", model=solver["model"], text=draft[:2000])

    final = draft
    if verifier:
        verify_prompt = (
            "You are a verifier. Check algebra, units, assumptions, and final numeric result.\n"
            "If wrong, give corrected solution.\n\n"
            f"Problem:\n{user_prompt}\n\n"
            f"Draft solution:\n{draft}\n"
        )
        checked = _call(verifier, verify_prompt, temperature=0.1)
        tracer.log("verifier", model=verifier["model"], text=checked[:2000])
        # v0: if verifier provides a correction, use it as final
        final = checked

    tracer.set_result(success=True, final_answer=final)
    trace_path = tracer.flush()
    return {"task_id": task_id, "answer": final, "trace": str(trace_path)}
