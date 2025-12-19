from __future__ import annotations

from abc import ABC, abstractmethod


class LLM(ABC):
    @abstractmethod
    def complete(
        self,
        *,
        prompt: str,
        model_id: str,
        temperature: float,
        seed: int | None,
    ) -> str:
        raise NotImplementedError

