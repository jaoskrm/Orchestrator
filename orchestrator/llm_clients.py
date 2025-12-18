# orchestrator/llm_clients.py

from __future__ import annotations

from typing import Any, Dict, Optional

import requests


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout_s: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "temperature": float(temperature),
            "stream": False,
        }

        # Ollama uses "options" for many decoding params
        options: Dict[str, Any] = {}
        if max_tokens is not None:
            options["num_predict"] = int(max_tokens)
        if seed is not None:
            options["seed"] = int(seed)
        if options:
            payload["options"] = options

        r = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        data = r.json()
        # Ollama returns {"response": "...", ...}
        return (data.get("response") or "").strip()
