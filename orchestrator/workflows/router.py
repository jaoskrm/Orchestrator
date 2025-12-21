# orchestrator/workflows/router.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Set

from orchestrator.llm_clients import OllamaClient
from orchestrator.config import config

ALLOWED_WORKFLOWS = {"coding", "reasoning", "science", "rag_qa"}
ALLOWED_ROLES = {"solver", "critic", "verifier", "retriever"}
ALLOWED_PROVIDERS = {"ollama", "openai", "anthropic", "gemini", "groq"}


def _allowed_ollama_models() -> Set[str]:
    models: Set[str] = set()
    # From config
    for wf in ALLOWED_WORKFLOWS:
        for w in config.get_workers(wf):
            if isinstance(w, dict) and w.get("provider") == "ollama" and w.get("model"):
                models.add(str(w["model"]))
    
    # Optional: also fetch from Ollama API (slower but catches new models)
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.ok:
            for model in resp.json().get("models", []):
                models.add(model["name"])
    except Exception:
        pass  # Fallback to config-only if Ollama unreachable
    
    return models



ROUTER_SYSTEM = """You are a ROUTING CONTROLLER, NOT a code generator.

Your ONLY job: Classify task type and return JSON.

NEVER write code, explanations, or markdown. JSON ONLY.

Task types:
- "coding": Python programs, classes, scripts, tests
- "science": Physics/chem/math with numbers + units  
- "rag_qa": Questions needing document lookup
- "reasoning": Pure logic puzzles

Examples show exact JSON format. Match exactly.

Output JSON only - no other text.
"""


EXAMPLE_1 = {
    "workflow": "coding",
    "workers": [
        {"provider": "ollama", "model": "deepseek-coder-v2:16b", "role": "solver"}, 
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
        f"Example A:\n{json.dumps(EXAMPLE_1, ensure_ascii=False)}\n\n"
        f"Example B:\n{json.dumps(EXAMPLE_2, ensure_ascii=False)}\n\n"
        f"Task:\n{user_task}\n\n"
        f"JSON:"
    )


def _extract_first_json_object(raw: str) -> dict:
    s = (raw or "").strip()
    if not s:
        raise ValueError("Router returned empty output")

    # Fast path
    if s.startswith("{"):
        try:
            return json.loads(s)
        except Exception:
            pass

    in_str = False
    esc = False
    depth = 0
    start = -1

    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    candidate = s[start : i + 1]
                    try:
                        remaining = s[i+1:].strip()
                        if remaining and '{' in remaining[:50]:
                            import logging
                            logging.warning(f"Router returned multiple JSON objects; using first. Remaining: {remaining[:100]}")
                        return json.loads(candidate)
                    except Exception:
                        pass

    raise ValueError(f"Router did not return a valid JSON object. Raw: {s[:500]}")


def _deduplicate_workers(workers: list[dict]) -> list[dict]:
    """
    Remove duplicate workers by (provider, model, role) tuple.
    Keeps first occurrence to preserve router/config priority order.
    """
    seen: Set[tuple] = set()
    clean = []
    for w in workers:
        if not isinstance(w, dict):
            continue
        key = (w.get("provider"), w.get("model"), w.get("role"))
        if key not in seen:
            seen.add(key)
            clean.append(w)
    return clean


def _has_role(workers: list[dict], role: str) -> bool:
    """Check if workers list contains at least one worker with given role."""
    return any(w.get("role") == role for w in workers if isinstance(w, dict))


def _ensure_worker(workers: list[dict], worker_spec: dict) -> list[dict]:
    """
    Add worker_spec if its role is missing. Always deduplicate after.
    """
    if not _has_role(workers, worker_spec.get("role", "")):
        workers.append(worker_spec)
    return _deduplicate_workers(workers)


def route_task(user_task: str, workflow_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Route task to appropriate workflow. If workflow_override provided, skip LLM routing.
    """
    # Override path
    if workflow_override and workflow_override in ALLOWED_WORKFLOWS:
        wf = workflow_override
        decision: Dict[str, Any] = {
            "workflow": wf,
            "workers": config.get_workers(wf),
            "tools": {},
            "controls": {},
        }
        raw = ""  # No LLM call made
    else:
        client = OllamaClient()
        prompt = _build_prompt(user_task)
        raw = client.generate(model=config.router_model, prompt=prompt, temperature=0.1)
        decision = _extract_first_json_object(raw)

    wf = decision.get("workflow")
    if wf not in ALLOWED_WORKFLOWS:
        raise ValueError(f"Invalid workflow: {wf}")

    # Coerce types + defaults
    workers = decision.get("workers")
    if not isinstance(workers, list):
        workers = []
    tools = decision.get("tools")
    if not isinstance(tools, dict):
        tools = {}
    controls = decision.get("controls")
    if not isinstance(controls, dict):
        controls = {}

    # Fill defaults
    if not workers:
        workers = config.get_workers(wf)  # ✅ Fixed: removed second argument
    tools.setdefault("python_sandbox", False)
    tools.setdefault("rag", False)
    controls.setdefault("use_debate", True)
    controls.setdefault("max_rounds", 3)

    decision["workers"] = workers
    decision["tools"] = tools
    decision["controls"] = controls

    # Validate workers
    allowed_ollama = _allowed_ollama_models()  # ✅ Fixed: call function instead of non-existent variable
    clean_workers = []
    for w in decision["workers"]:
        if not isinstance(w, dict):
            continue
        if w.get("provider") not in ALLOWED_PROVIDERS:
            continue
        if w.get("role") not in ALLOWED_ROLES:
            continue
        if w.get("provider") == "ollama":
            model = w.get("model")
            if not model or str(model) not in allowed_ollama:
                continue
        clean_workers.append(w)

    if not clean_workers:
        clean_workers = config.get_workers(wf)  # ✅ Fixed: removed second argument

    decision["workers"] = clean_workers

    # Normalize controls
    try:
        mr = int(decision["controls"].get("max_rounds", 1))
    except Exception:
        mr = 1
    decision["controls"]["max_rounds"] = max(1, min(mr, 4))
    decision["controls"]["use_debate"] = bool(decision["controls"].get("use_debate", False))

    # Enforce workflow invariants (deduplicated)
    if wf == "coding":
        decision["tools"]["python_sandbox"] = True
        decision["tools"]["rag"] = False
        decision["controls"]["max_rounds"] = max(3, decision["controls"].get("max_rounds", 3))
        decision["controls"]["use_debate"] = True

        if not _has_role(decision["workers"], "solver"):
            decision["workers"] = config.get_workers("coding") + decision["workers"]

        decision["workers"] = _ensure_worker(
            decision["workers"],
            {"provider": "ollama", "model": "llama3:latest", "role": "verifier"}
        )

    elif wf == "rag_qa":
        decision["tools"]["rag"] = True
        decision["tools"]["python_sandbox"] = False

        if not _has_role(decision["workers"], "solver"):
            decision["workers"] = config.get_workers("rag_qa") + decision["workers"]

        decision["workers"] = _ensure_worker(
            decision["workers"],
            {"provider": "ollama", "model": "qwen2.5:7b", "role": "retriever"}
        )

    elif wf == "science":
        decision["tools"]["rag"] = False
        decision["tools"]["python_sandbox"] = False

        if decision["controls"]["max_rounds"] == 1:
            decision["controls"]["max_rounds"] = 2

        if not _has_role(decision["workers"], "solver"):
            decision["workers"] = config.get_workers("science") + decision["workers"]

        decision["workers"] = _ensure_worker(
            decision["workers"],
            {"provider": "ollama", "model": "llama3:latest", "role": "verifier"}
        )

    elif wf == "reasoning":
        decision["tools"]["rag"] = False
        decision["tools"]["python_sandbox"] = False
        
        # Only set default if LLM didn't include the field at all
        if "use_debate" not in controls:  # Check original controls dict
            decision["controls"]["use_debate"] = True
        else:
            # LLM explicitly set it, respect that choice (even if False)
            decision["controls"]["use_debate"] = bool(controls.get("use_debate"))

        if not _has_role(decision["workers"], "solver"):
            decision["workers"] = config.get_workers("reasoning") + decision["workers"]

        if decision["controls"]["use_debate"]:
            decision["workers"] = _ensure_worker(
                decision["workers"],
                {"provider": "ollama", "model": "llama3:latest", "role": "critic"}
            )

    # Final global dedupe
    decision["workers"] = _deduplicate_workers(decision["workers"])

    return decision
