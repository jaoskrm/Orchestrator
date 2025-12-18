# orchestrator/config.py

ROUTER_MODEL = "llama3.2:latest"

# Default worker pools per workflow.
# You can still override via router JSON, but this is the fallback.
DEFAULT_WORKERS = {
    "coding": [
        {"provider": "ollama", "model": "qwen2.5:7b", "role": "solver"},
        {"provider": "ollama", "model": "llama3:latest", "role": "verifier"},
    ],
    "reasoning": [
        # Use the stronger 14B for multi-step reasoning
        {"provider": "ollama", "model": "qwen2.5:14b-instruct", "role": "solver"},
        {"provider": "ollama", "model": "llama3:latest", "role": "critic"},
    ],
    "science": [
        # Use the stronger 14B for math/physics/chem style problems
        {"provider": "ollama", "model": "qwen2.5:14b-instruct", "role": "solver"},
        {"provider": "ollama", "model": "llama3:latest", "role": "verifier"},
    ],
    "rag_qa": [
        {"provider": "ollama", "model": "qwen2.5:7b", "role": "retriever"},
        {"provider": "ollama", "model": "qwen2.5:7b", "role": "solver"},
        {"provider": "ollama", "model": "llama3:latest", "role": "verifier"},
    ],
}
