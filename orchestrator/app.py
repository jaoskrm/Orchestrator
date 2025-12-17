import json
from pathlib import Path

from orchestrator.judge.judge import judge_trace
from orchestrator.workflows.coding import run_coding_workflow
from orchestrator.workflows.rag_qa import run_rag_qa_workflow
from orchestrator.workflows.reasoning import run_reasoning_workflow
from orchestrator.workflows.science import run_science_workflow


def _load_trace(trace_path: str) -> dict:
    return json.loads(Path(trace_path).read_text(encoding="utf-8"))


def main(task_id: str):
    prompt = Path("runs") / task_id / "prompt.txt"
    user_prompt = prompt.read_text(encoding="utf-8")

    # decision = route_task(user_prompt)   # assuming you already do this
    decision = route_task(user_prompt)

    wf = decision["workflow"]
    if wf == "coding":
        result = run_coding_workflow(task_id, user_prompt, decision)
    elif wf == "rag_qa":
        result = run_rag_qa_workflow(task_id, user_prompt, decision)
    elif wf == "reasoning":
        result = run_reasoning_workflow(task_id, user_prompt, decision)
    elif wf == "science":
        result = run_science_workflow(task_id, user_prompt, decision)
    else:
        raise ValueError(f"Unknown workflow: {wf}")

    # judge
    trace = _load_trace(result["trace"])
    verdict = judge_trace(trace)

    # unify output
    return {
        "task_id": task_id,
        "workflow": wf,
        "passed": verdict["passed"],
        "score": verdict["score"],
        "reason": verdict["reason"],
        "trace": result["trace"],
        "result": result,
    }

from orchestrator.workflows.router import route_task  # add this near the top with imports

if __name__ == "__main__":
    # Default task id; optionally override with: python -m orchestrator.app task_002
    import sys
    task_id = sys.argv[1] if len(sys.argv) > 1 else "task_002"
    print(main(task_id))

