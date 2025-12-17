import requests

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def generate(self, model: str, prompt: str, temperature: float = 0.2) -> str:
        r = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": model, "prompt": prompt, "temperature": temperature, "stream": False},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["response"]
