import json
from urllib.request import Request, urlopen
from urllib.error import URLError


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3.5:9b"


class LocalLLM:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        ollama_url: str = OLLAMA_URL,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:

        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }

        request = Request(
            self.ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request) as response:
                data = json.loads(
                    response.read().decode("utf-8")
                )

        except URLError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            ) from exc

        try:
            return data["message"]["content"]

        except KeyError as exc:
            raise RuntimeError(
                f"Unexpected Ollama response: {data}"
            ) from exc