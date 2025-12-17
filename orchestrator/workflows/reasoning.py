from typing import Any, Dict
from orchestrator.llm_clients import OllamaClient
from orchestrator.storage.traces import TraceWriter

def _call(worker: Dict[str, Any], prompt: str, temperature: float = 0.2) -> str:
    if worker["provider"] == "ollama":
        return OllamaClient().generate(model=worker["model"], prompt=prompt, temperature=temperature)
    raise ValueError(f"Unsupported provider: {worker['provider']}")

def run_reasoning_workflow(task_id: str, user_prompt: str, decision: Dict[str, Any]) -> Dict[str, Any]:
    tracer = TraceWriter(task_id, "reasoning", user_prompt, decision)

    workers = decision.get("workers", [])
    solver = next((w for w in workers if w.get("role") == "solver"), None)
    critic = next((w for w in workers if w.get("role") in ("critic", "verifier")), None)
    if not solver:
        raise ValueError("reasoning workflow requires a solver worker")

    use_debate = bool(decision.get("controls", {}).get("use_debate", True))

    solve_prompt = (
        "Solve the problem clearly and correctly.\n"
        "If information is missing, state assumptions.\n\n"
        f"Problem:\n{user_prompt}\n"
    )
    draft = _call(solver, solve_prompt, temperature=0.2)
    tracer.log("solver_draft", model=solver["model"], text=draft[:2000])

    final = draft
    if use_debate and critic:
        critic_prompt = (
            "You are a strict critic. Find mistakes, missing cases, or weak reasoning.\n\n"
            f"Problem:\n{user_prompt}\n\n"
            f"Proposed answer:\n{draft}\n"
        )
        critique = _call(critic, critic_prompt, temperature=0.1)
        tracer.log("critic", model=critic["model"], text=critique[:2000])

        revise_prompt = (
            "Revise the answer using the critique. Output the final improved answer.\n\n"
            f"Problem:\n{user_prompt}\n\n"
            f"Critique:\n{critique}\n\n"
            f"Original answer:\n{draft}\n"
        )
        final = _call(solver, revise_prompt, temperature=0.2)
        tracer.log("solver_final", model=solver["model"], text=final[:2000])

    tracer.set_result(success=True, final_answer=final)
    trace_path = tracer.flush()

    return {"task_id": task_id, "answer": final, "trace": str(trace_path)}
