import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        backend: str = "ollama",
        model: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
    ) -> None:
        self.backend = backend
        self.model = model or "mixtral:8x7b"
        self.openai_model = openai_model

    def chat(self, messages: List[dict], temperature: float = 0.0) -> str:
        if self.backend == "ollama":
            return self._chat_ollama(messages, temperature=temperature)
        if self.backend == "openai":
            return self._chat_openai(messages, temperature=temperature)
        raise ValueError(f"Unsupported LLM backend: {self.backend}")

    def _chat_ollama(self, messages: List[dict], temperature: float) -> str:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError("ollama package not installed") from exc
        response = ollama.chat(model=self.model, messages=messages, options={"temperature": temperature})
        return response["message"]["content"]

    def _chat_openai(self, messages: List[dict], temperature: float) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package not installed") from exc
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            temperature=temperature,
        )
        return completion.choices[0].message.content


def safe_llm_call(fn_name: str, func, default: str = "") -> str:
    try:
        return func()
    except Exception as exc:
        logger.debug("LLM call %s failed: %s", fn_name, exc)
        return default
