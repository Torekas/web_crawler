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
        payload = parsed if parsed is not None else _fallback_payload(self.output_model, text)
        obj = self.output_model.model_validate(payload)
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


def _fallback_payload(output_model: type[BaseModel], text: str) -> dict[str, Any]:
    fields = output_model.model_fields
    if "answer" in fields:
        return {
            "answer": text,
            "reasoning": "Fallback: model output could not be parsed.",
            "applied_rules": [
                {"rule_id": "fallback_parse", "pass": False, "evidence": ["response_text"]}
            ],
            "policy_decisions": {},
            "risk_flags": ["unparsed_output"],
            "confidence_by_claim": {"fallback_claim": 0.0},
            "evidence_map": {"fallback_claim": ["response_text"]},
            "uncertainty": {"required": True, "reason": "unparseable output"},
        }
    if "verdict" in fields and "scores" in fields:
        return {
            "verdict": "reject",
            "confidence": 0.0,
            "scores": {
                "correctness": 0.0,
                "completeness": 0.0,
                "reasoning_quality": 0.0,
                "factuality": 0.0,
                "robustness": 0.0,
            },
            "justification": ["Fallback: model output could not be parsed."],
            "key_failures": ["unparseable_output"],
            "required_fixes": ["regenerate_response"],
            "consistency_checks": {},
            "tags": ["fallback"],
        }
    return {}
