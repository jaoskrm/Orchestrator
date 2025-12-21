# debug_router.py
from orchestrator.workflows.router import _allowed_ollama_models, route_task
from orchestrator.config import config

print("=== Config Workers ===")
print(config.get_workers("coding"))

print("\n=== Allowed Ollama Models ===")
allowed = _allowed_ollama_models()
print("deepseek-coder-v2:16b" in allowed, "← Should be True")
print(sorted([m for m in allowed if "deepseek" in m.lower()]))

print("\n=== Route Decision ===")
decision = route_task("write a fibonacci function", workflow_override="coding")
print(decision["workers"])
