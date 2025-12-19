from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class Question(BaseModel):
    question_id: str
    question: str
    answer_key: str | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class QuestionSource(ABC):
    @abstractmethod
    def sample(self, *, n: int, seed: int) -> list[Question]:
        raise NotImplementedError

