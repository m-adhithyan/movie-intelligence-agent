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
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.max_retries = max_retries

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

        for attempt in range(1, self.max_retries + 1):
            request = Request(
                self.ollama_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            try:
                with urlopen(request, timeout=120) as response:
                    data = json.loads(
                        response.read().decode("utf-8")
                    )

            except URLError as exc:
                raise RuntimeError(
                    "Could not connect to Ollama. "
                    "Make sure Ollama is running."
                ) from exc

            try:
                content = data["message"]["content"]
            except KeyError as exc:
                raise RuntimeError(
                    f"Unexpected Ollama response: {data}"
                ) from exc

            # The thinking model sometimes returns empty content
            # on the first attempt. Retry if that happens.
            if content and content.strip():
                return content

            if attempt < self.max_retries:
                continue  # retry

        # All retries exhausted — return whatever we have (may be empty)
        return content