from __future__ import annotations

import re
from typing import Iterable

from pydantic import BaseModel, Field

from model_court.agents.contracts import CandidateAnswer


class ContradictionResult(BaseModel):
    contradiction_flags: list[str] = Field(default_factory=list)
    conflicting_claims: list[str] = Field(default_factory=list)
    missing_required_fields: list[str] = Field(default_factory=list)


_CONTRADICTION_PAIRS = [
    ("yes", "no"),
    ("true", "false"),
    ("always", "never"),
    ("can", "cannot"),
    ("must", "must not"),
]


def detect_contradictions(
    candidate: CandidateAnswer,
    *,
    required_fields: Iterable[str] | None = None,
) -> ContradictionResult:
    text = f"{candidate.answer} {candidate.reasoning}".lower()
    contradiction_flags: list[str] = []
    conflicting_claims: list[str] = []

    for left, right in _CONTRADICTION_PAIRS:
        left_hit = re.search(rf"\\b{re.escape(left)}\\b", text) is not None
        right_hit = re.search(rf"\\b{re.escape(right)}\\b", text) is not None
        if left_hit and right_hit:
            contradiction_flags.append(f"{left}_vs_{right}")
            conflicting_claims.append(f"Contains both '{left}' and '{right}'.")

    missing_required_fields: list[str] = []
    check_fields = list(required_fields) if required_fields is not None else ["answer", "reasoning", "applied_rules"]
    for field_name in check_fields:
        if field_name == "answer" and not candidate.answer.strip():
            missing_required_fields.append(field_name)
        elif field_name == "reasoning" and not candidate.reasoning.strip():
            missing_required_fields.append(field_name)
        elif field_name == "applied_rules" and not candidate.applied_rules:
            missing_required_fields.append(field_name)
        elif field_name == "confidence_by_claim" and not candidate.confidence_by_claim:
            missing_required_fields.append(field_name)
        elif field_name == "evidence_map" and not candidate.evidence_map:
            missing_required_fields.append(field_name)

    return ContradictionResult(
        contradiction_flags=contradiction_flags,
        conflicting_claims=conflicting_claims,
        missing_required_fields=missing_required_fields,
    )
