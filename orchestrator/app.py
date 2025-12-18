# orchestrator/app.py
import json
from pathlib import Path
from typing import Optional

from orchestrator.judge.judge import judge_trace
from orchestrator.workflows.coding import run_coding_workflow
from orchestrator.workflows.rag_qa import run_rag_qa_workflow
from orchestrator.workflows.reasoning import run_reasoning_workflow
from orchestrator.workflows.science import run_science_workflow
from orchestrator.workflows.router import route_task


def _load_trace(trace_path: str) -> dict:
    return json.loads(Path(trace_path).read_text(encoding="utf-8"))


def main(task_id: str, workflow_override: Optional[str] = None):
    prompt = Path("runs") / task_id / "prompt.txt"
    user_prompt = prompt.read_text(encoding="utf-8")

    decision = route_task(user_prompt, workflow_override=workflow_override)
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

    trace = _load_trace(result["trace"])
    verdict = judge_trace(trace)

    return {
        "task_id": task_id,
        "workflow": wf,
        "passed": verdict["passed"],
        "score": verdict["score"],
        "reason": verdict["reason"],
        "trace": result["trace"],
        "result": result,
    }


if __name__ == "__main__":
    import sys
    task_id = sys.argv[1] if len(sys.argv) > 1 else "task_003"
    override = sys.argv[2] if len(sys.argv) > 2 else None  # e.g. "rag_qa"
    print(main(task_id, workflow_override=override))
