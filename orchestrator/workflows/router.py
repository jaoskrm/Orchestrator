import json
from typing import Any, Dict

from orchestrator.llm_clients import OllamaClient
from orchestrator.config import ROUTER_MODEL, DEFAULT_WORKERS

ALLOWED_WORKFLOWS = {"coding", "reasoning", "science", "rag_qa"}
ALLOWED_ROLES = {"solver", "critic", "verifier", "retriever"}
ALLOWED_PROVIDERS = {"ollama", "openai", "anthropic", "gemini", "groq"}
ALLOWED_OLLAMA_MODELS = {"llama3.2:latest", "qwen2.5:7b", "llama3:latest"}

ROUTER_SYSTEM = """You are a routing controller for a multi-model agent system.

Return ONLY valid JSON (no markdown, no commentary).
Choose:
- workflow: one of ["coding","reasoning","science","rag_qa"]
- workers: list of workers with provider/model/role
- tools: python_sandbox (bool), rag (bool)
- controls: use_debate (bool), max_rounds (1-4)

Available local models (Ollama):
- llama3.2:latest (router, small)
- qwen2.5:7b (strong general/coding/science)
- llama3:latest (alternative general, good critic)

Guidelines:
- coding: include python_sandbox=true and add a verifier/critic worker.
- rag_qa: set tools.rag=true and include a retriever worker role.
- science: include a verifier for unit checks / algebra checks.
- reasoning: use_debate=true for hard multi-step logic.

Output JSON only.
"""

EXAMPLE_1 = {
    "workflow": "coding",
    "workers": [
        {"provider": "ollama", "model": "qwen2.5:7b", "role": "solver"},
        {"provider": "ollama", "model": "llama3:latest", "role": "verifier"},
    ],
    "tools": {"python_sandbox": True, "rag": False},
    "controls": {"use_debate": True, "max_rounds": 3},
}

EXAMPLE_2 = {
    "workflow": "rag_qa",
    "workers": [
        {"provider": "ollama", "model": "qwen2.5:7b", "role": "retriever"},
        {"provider": "ollama", "model": "qwen2.5:7b", "role": "solver"},
        {"provider": "ollama", "model": "llama3:latest", "role": "verifier"},
    ],
    "tools": {"python_sandbox": False, "rag": True},
    "controls": {"use_debate": False, "max_rounds": 2},
}

def _build_prompt(user_task: str) -> str:
    return (
        f"{ROUTER_SYSTEM}\n\n"
        f"Example A:\n{json.dumps(EXAMPLE_1)}\n\n"
        f"Example B:\n{json.dumps(EXAMPLE_2)}\n\n"
        f"Task:\n{user_task}\n\n"
        f"JSON:"
    )

def _extract_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Router did not return JSON. Raw:\n{raw[:500]}")
    return json.loads(raw[start:end + 1])

def route_task(user_task: str) -> Dict[str, Any]:
    client = OllamaClient()
    prompt = _build_prompt(user_task)
    raw = client.generate(model=ROUTER_MODEL, prompt=prompt, temperature=0.1)
    decision = _extract_json(raw)

    wf = decision.get("workflow")
    if wf not in ALLOWED_WORKFLOWS:
        raise ValueError(f"Invalid workflow: {wf}")

    # Fill defaults
    decision.setdefault("workers", DEFAULT_WORKERS.get(wf, []))
    decision.setdefault("tools", {"python_sandbox": False, "rag": False})
    decision.setdefault("controls", {"use_debate": False, "max_rounds": 1})

    # Validate workers
    clean_workers = []
    for w in decision["workers"]:
        if w.get("provider") not in ALLOWED_PROVIDERS:
            continue
        if w.get("role") not in ALLOWED_ROLES:
            continue
        if w["provider"] == "ollama" and w.get("model") not in ALLOWED_OLLAMA_MODELS:
            continue
        clean_workers.append(w)

    # Fallback if router gave unusable workers
    if not clean_workers:
        clean_workers = DEFAULT_WORKERS.get(wf, [])
    decision["workers"] = clean_workers

    # Normalize controls
    decision["controls"]["max_rounds"] = max(
        1, min(int(decision["controls"].get("max_rounds", 1)), 4)
    )

    return decision
