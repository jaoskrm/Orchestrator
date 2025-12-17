ROUTER_MODEL = "llama3.2:latest"

DEFAULT_WORKERS = {
    "coding": [
        {"provider":"ollama","model":"qwen2.5:7b","role":"solver"},
        {"provider":"ollama","model":"llama3:latest","role":"verifier"},
    ],
    "reasoning": [
        {"provider":"ollama","model":"qwen2.5:7b","role":"solver"},
        {"provider":"ollama","model":"llama3:latest","role":"critic"},
    ],
    "science": [
        {"provider":"ollama","model":"qwen2.5:7b","role":"solver"},
        {"provider":"ollama","model":"llama3:latest","role":"verifier"},
    ],
    "rag_qa": [
        {"provider":"ollama","model":"qwen2.5:7b","role":"retriever"},
        {"provider":"ollama","model":"qwen2.5:7b","role":"solver"},
        {"provider":"ollama","model":"llama3:latest","role":"verifier"},
    ],
}
