import json

import requests


class LLMRouter:
    """Minimal local-only router for Ollama-backed text generation."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self._base_url = base_url.rstrip("/")

    def call_local_chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0,
    ) -> str:
        prompt = (
            f"{system_prompt.strip()}\n\n"
            "Return only the final answer.\n\n"
            f"{user_prompt.strip()}"
        )
        try:
            response = requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=120,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to reach Ollama at {self._base_url}: {exc}") from exc
        if response.status_code != 200:
            body = response.text.strip()
            raise RuntimeError(
                f"Ollama generate failed with status {response.status_code}: {body}"
            )
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Ollama returned non-JSON response: {response.text.strip()}"
            ) from exc
        output = payload.get("response")
        if not isinstance(output, str):
            raise RuntimeError(
                f"Ollama response missing 'response' field: {json.dumps(payload, ensure_ascii=False)}"
            )
        return output.strip()
