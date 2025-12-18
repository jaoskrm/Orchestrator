import json
from typing import Any, Dict

from orchestrator.llm_clients import OllamaClient
from orchestrator.config import ROUTER_MODEL, DEFAULT_WORKERS

ALLOWED_WORKFLOWS = {"coding", "reasoning", "science", "rag_qa"}
ALLOWED_ROLES = {"solver", "critic", "verifier", "retriever"}
ALLOWED_PROVIDERS = {"ollama", "openai", "anthropic", "gemini", "groq"}
ALLOWED_OLLAMA_MODELS = {
    "llama3.2:latest",
    "qwen2.5:7b",
    "qwen2.5:14b-instruct",
    "llama3:latest",
}


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

def _extract_first_json_object(raw: str) -> dict:
    s = raw.strip()

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
        else:
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
                        candidate = s[start:i+1]
                        return json.loads(candidate)

    raise ValueError(f"Router did not return a valid JSON object. Raw: {raw[:500]}")


def route_task(user_task: str) -> Dict[str, Any]:
    client = OllamaClient()
    prompt = _build_prompt(user_task)
    raw = client.generate(model=ROUTER_MODEL, prompt=prompt, temperature=0.1)
    decision = _extract_first_json_object(raw)

    wf = decision.get("workflow")
    if wf not in ALLOWED_WORKFLOWS:
        raise ValueError(f"Invalid workflow: {wf}")

    # ---- Coerce types + defaults (LLM-safe) ----
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
        workers = DEFAULT_WORKERS.get(wf, [])
    tools.setdefault("python_sandbox", False)
    tools.setdefault("rag", False)
    controls.setdefault("use_debate", False)
    controls.setdefault("max_rounds", 1)

    decision["workers"] = workers
    decision["tools"] = tools
    decision["controls"] = controls

    # ---- Validate workers (keep only allowed) ----
    clean_workers = []
    for w in decision["workers"]:
        if not isinstance(w, dict):
            continue
        if w.get("provider") not in ALLOWED_PROVIDERS:
            continue
        if w.get("role") not in ALLOWED_ROLES:
            continue
        if w.get("provider") == "ollama" and w.get("model") not in ALLOWED_OLLAMA_MODELS:
            continue
        clean_workers.append(w)

    if not clean_workers:
        clean_workers = DEFAULT_WORKERS.get(wf, [])
    decision["workers"] = clean_workers

    # Ensure required roles exist (post-validation)
    if wf == "science" and not any(w.get("role") == "solver" for w in decision["workers"]):
        decision["workers"] = DEFAULT_WORKERS.get("science", []) + decision["workers"]

    if wf == "reasoning" and not any(w.get("role") == "solver" for w in decision["workers"]):
        decision["workers"] = DEFAULT_WORKERS.get("reasoning", []) + decision["workers"]

    if wf == "coding" and not any(w.get("role") == "solver" for w in decision["workers"]):
        decision["workers"] = DEFAULT_WORKERS.get("coding", []) + decision["workers"]

    if wf == "rag_qa" and not any(w.get("role") == "solver" for w in decision["workers"]):
        decision["workers"] = DEFAULT_WORKERS.get("rag_qa", []) + decision["workers"]

    # ---- Normalize controls ----
    try:
        mr = int(decision["controls"].get("max_rounds", 1))
    except Exception:
        mr = 1
    decision["controls"]["max_rounds"] = max(1, min(mr, 4))
    decision["controls"]["use_debate"] = bool(decision["controls"].get("use_debate", False))

    # ---- Enforce workflow invariants ----

    if wf == "coding":
        decision["tools"]["python_sandbox"] = True
        decision["tools"]["rag"] = False
        if not any(w.get("role") in ("verifier", "critic") for w in decision["workers"]):
            decision["workers"].append({"provider": "ollama", "model": "llama3:latest", "role": "verifier"})

    elif wf == "rag_qa":
        decision["tools"]["rag"] = True
        decision["tools"]["python_sandbox"] = False
        if not any(w.get("role") == "retriever" for w in decision["workers"]):
            decision["workers"] = [{"provider": "ollama", "model": "qwen2.5:7b", "role": "retriever"}] + decision["workers"]

    elif wf == "science":
        decision["tools"]["rag"] = False
        decision["tools"]["python_sandbox"] = False
        decision["controls"]["use_debate"] = bool(decision["controls"].get("use_debate", False))
        decision["controls"]["max_rounds"] = int(decision["controls"].get("max_rounds", 2))

        # science MUST have verifier/critic
        if not any(w.get("role") in ("verifier", "critic") for w in decision["workers"]):
            decision["workers"].append({"provider": "ollama", "model": "llama3:latest", "role": "verifier"})

    elif wf == "reasoning":
        decision["tools"]["rag"] = False
        decision["tools"]["python_sandbox"] = False
        # reasoning debate default ON unless explicitly false
        decision["controls"]["use_debate"] = bool(decision["controls"].get("use_debate", True))
        if decision["controls"]["use_debate"] and not any(w.get("role") == "critic" for w in decision["workers"]):
            decision["workers"].append({"provider": "ollama", "model": "llama3:latest", "role": "critic"})

    return decision


