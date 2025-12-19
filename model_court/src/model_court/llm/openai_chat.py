from __future__ import annotations

from dataclasses import dataclass

from model_court.llm.base import LLM


@dataclass(frozen=True)
class OpenAIChatLLM(LLM):
    """
    Optional provider (not enabled by default). Install `openai` and set `OPENAI_API_KEY`.
    This is intentionally minimal to keep the project runnable without network keys.
    """

    def complete(self, *, prompt: str, model_id: str, temperature: float, seed: int | None) -> str:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("OpenAI provider requires `pip install openai`.") from e

        client = OpenAI()
        # Note: OpenAI APIs may evolve; treat this as a skeleton.
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

