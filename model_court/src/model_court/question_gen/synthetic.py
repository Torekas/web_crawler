from __future__ import annotations

import random
from dataclasses import dataclass

from model_court.question_gen.base import Question, QuestionSource


@dataclass(frozen=True)
class SyntheticMathV1(QuestionSource):
    difficulty: str = "mixed"  # easy|medium|mixed

    def sample(self, *, n: int, seed: int) -> list[Question]:
        rng = random.Random(seed)
        out: list[Question] = []
        for i in range(n):
            a, b = _draw_operands(rng, self.difficulty)
            op = rng.choice(["+", "-", "*"])
            q, ans = _render(a, b, op, rng)
            out.append(
                Question(
                    question_id=f"synthetic_math_v1_{seed}_{i:05d}",
                    question=q,
                    answer_key=str(ans),
                    source="synthetic_math_v1",
                    metadata={"a": a, "b": b, "op": op, "difficulty": self.difficulty},
                )
            )
        return out


def _draw_operands(rng: random.Random, difficulty: str) -> tuple[int, int]:
    if difficulty == "easy":
        return rng.randint(1, 20), rng.randint(1, 20)
    if difficulty == "medium":
        return rng.randint(10, 200), rng.randint(10, 200)
    return rng.randint(1, 200), rng.randint(1, 200)


def _render(a: int, b: int, op: str, rng: random.Random) -> tuple[str, int]:
    if op == "+":
        return f"What is {a} + {b}?", a + b
    if op == "-":
        return f"What is {a} - {b}?", a - b
    # Multiply: occasionally render a short word problem.
    if rng.random() < 0.4:
        return (
            f"A box holds {a} items. You have {b} identical boxes. How many items total?",
            a * b,
        )
    return f"What is {a} * {b}?", a * b

