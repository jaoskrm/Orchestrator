import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

TRACE_DIR = Path("traces")
TRACE_DIR.mkdir(exist_ok=True)

def now_ms() -> int:
    return int(time.time() * 1000)

@dataclass
class TraceEvent:
    ts_ms: int
    kind: str
    payload: Dict[str, Any]

@dataclass
class TraceRecord:
    run_id: str
    task_id: str
    workflow: str
    prompt: str
    decision: Dict[str, Any]
    started_ms: int
    finished_ms: Optional[int]
    success: Optional[bool]
    final_answer: Optional[str]
    events: list

class TraceWriter:
    def __init__(self, task_id: str, workflow: str, prompt: str, decision: Dict[str, Any]):
        self.task_id = task_id
        self.workflow = workflow
        self.prompt = prompt
        self.decision = decision
        self.run_id = f"{task_id}-{now_ms()}"
        self.started_ms = now_ms()
        self.events: list[TraceEvent] = []
        self.final_answer: Optional[str] = None
        self.success: Optional[bool] = None

    def log(self, kind: str, **payload: Any) -> None:
        self.events.append(TraceEvent(ts_ms=now_ms(), kind=kind, payload=payload))

    def set_result(self, success: bool, final_answer: Optional[str] = None) -> None:
        self.success = success
        self.final_answer = final_answer

    def flush(self) -> Path:
        rec = TraceRecord(
            run_id=self.run_id,
            task_id=self.task_id,
            workflow=self.workflow,
            prompt=self.prompt,
            decision=self.decision,
            started_ms=self.started_ms,
            finished_ms=now_ms(),
            success=self.success,
            final_answer=self.final_answer,
            events=[asdict(e) for e in self.events],
        )
        out = TRACE_DIR / f"{self.run_id}.json"
        out.write_text(json.dumps(asdict(rec), ensure_ascii=False, indent=2), encoding="utf-8")
        return out
