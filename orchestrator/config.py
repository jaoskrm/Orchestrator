# orchestrator/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # pip install pyyaml
except Exception:  # keep project runnable even if pyyaml not installed
    yaml = None

ROUTER_MODEL = "llama3.2:latest"

DEFAULT_WORKERS: Dict[str, list[dict[str, Any]]] = {
    "coding": [
        {"provider": "ollama", "model": "deepseek-coder-v2:16b", "role": "solver"},  # ✅ NEW
        {"provider": "ollama", "model": "llama3:latest", "role": "verifier"},
    ],
    "reasoning": [
        {"provider": "ollama", "model": "qwen2.5:14b-instruct", "role": "solver"},
        {"provider": "ollama", "model": "llama3:latest", "role": "critic"},
    ],
    "science": [
        {"provider": "ollama", "model": "qwen2.5:14b-instruct", "role": "solver"},
        {"provider": "ollama", "model": "llama3:latest", "role": "verifier"},
    ],
    "rag_qa": [
        {"provider": "ollama", "model": "qwen2.5:7b", "role": "retriever"},
        {"provider": "ollama", "model": "qwen2.5:7b", "role": "solver"},
        {"provider": "ollama", "model": "llama3:latest", "role": "verifier"},
    ],
}

class Config:
    def __init__(self, path: Path | None = None):
        self.path = path or (Path(__file__).with_name("config.yaml"))
        self.data: Dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        self.data = {}
        if yaml is None:
            return
        if not self.path.exists():
            return
        try:
            self.data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except Exception:
            self.data = {}

    @property
    def router_model(self) -> str:
        return (
            (self.data.get("models", {}) or {}).get("router")
            or ROUTER_MODEL
        )

    def get_workers(self, workflow: str) -> list[dict[str, Any]]:
        wf = (self.data.get("workflows", {}) or {}).get(workflow, {}) or {}
        workers = wf.get("workers")
        if isinstance(workers, list) and workers:
            return workers
        return DEFAULT_WORKERS.get(workflow, [])

config = Config()
