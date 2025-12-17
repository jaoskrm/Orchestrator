from typing import Any, Dict


def judge_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    workflow = trace.get("workflow")
    events = trace.get("events", [])

    if workflow == "coding":
        pytest_event = next((e for e in reversed(events) if e.get("kind") == "pytest"), None)
        exit_code = (pytest_event or {}).get("payload", {}).get("exit_code")
        passed = exit_code == 0
        return {"passed": passed, "score": 1.0 if passed else 0.0, "reason": f"pytest exit_code={exit_code}"}

    # Fallback: trust trace.success if present
    passed = bool(trace.get("success", False))
    return {"passed": passed, "score": 1.0 if passed else 0.0, "reason": "trace.success"}
