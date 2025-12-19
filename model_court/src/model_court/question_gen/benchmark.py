from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from model_court.question_gen.base import Question, QuestionSource


@dataclass(frozen=True)
class JSONLBenchmarkAdapter(QuestionSource):
    """
    Pluggable adapter for existing benchmarks (e.g., GSM8K, TruthfulQA) stored locally.

    Expected JSONL schema per line (minimal):
      {"id": "...", "question": "...", "answer": "...", "metadata": {...}}
    """

    path: Path
    source_name: str

    def sample(self, *, n: int, seed: int) -> list[Question]:
        # Deterministic shuffle via seed, no external downloads.
        lines = self.path.read_text(encoding="utf-8").splitlines()
        items = [json.loads(l) for l in lines if l.strip()]
        rng = __import__("random").Random(seed)
        rng.shuffle(items)
        out: list[Question] = []
        for obj in items[:n]:
            out.append(
                Question(
                    question_id=str(obj.get("id") or obj.get("question_id") or ""),
                    question=str(obj["question"]),
                    answer_key=(str(obj.get("answer")) if obj.get("answer") is not None else None),
                    source=self.source_name,
                    metadata=dict(obj.get("metadata") or {}),
                )
            )
        return out

