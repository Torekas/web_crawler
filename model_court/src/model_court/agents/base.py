from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from model_court.agents.contracts import AgentMeta, LLMCallRecord
from model_court.llm.base import LLM

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Agent(Generic[T]):
    role: str
    model_id: str
    llm: LLM
    output_model: type[T]
    prompt_path: Path
    temperature: float = 0.0

    def _render_prompt(self, *, input_json: dict[str, Any]) -> str:
        prompt = self.prompt_path.read_text(encoding="utf-8")
        return prompt.replace("{{INPUT_JSON}}", json.dumps(input_json, ensure_ascii=False, indent=2))

    def run(self, *, input_json: dict[str, Any], seed: int | None = None) -> tuple[T, LLMCallRecord]:
        prompt = self._render_prompt(input_json=input_json)
        text = self.llm.complete(
            prompt=prompt,
            model_id=self.model_id,
            temperature=self.temperature,
            seed=seed,
        )
        parsed = _best_effort_parse_json(text)
        obj = self.output_model.model_validate(parsed if parsed is not None else {"final_answer": text})
        record = LLMCallRecord(
            role=self.role,
            model_id=self.model_id,
            prompt=prompt,
            response_text=text,
            parsed_json=parsed,
        )
        return obj, record

    def meta(self, *, seed: int | None) -> AgentMeta:
        return AgentMeta(
            agent_role=self.role,
            model_id=self.model_id,
            temperature=self.temperature,
            seed=seed,
        )


def _best_effort_parse_json(text: str) -> dict[str, Any] | None:
    # Expect a single JSON object; tolerate surrounding text/code fences.
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json", "", 1).strip()
    first = s.find("{")
    last = s.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    candidate = s[first : last + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None

