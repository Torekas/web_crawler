from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentMeta(BaseModel):
    agent_role: str
    model_id: str
    temperature: float = 0.0
    seed: int | None = None


class AppliedRule(BaseModel):
    rule_id: str = Field(..., description="Registered rule identifier.")
    passed: bool = Field(..., alias="pass", description="Deterministic pass/fail result.")
    evidence: list[str] = Field(default_factory=list, description="Evidence pointers for the rule.")

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _validate_evidence(self) -> "AppliedRule":
        if not self.rule_id.strip():
            raise ValueError("applied_rules.rule_id must be non-empty")
        if not self.evidence:
            raise ValueError("applied_rules.evidence must be non-empty")
        if any(not ev.strip() for ev in self.evidence):
            raise ValueError("applied_rules.evidence entries must be non-empty")
        return self


class UncertaintyInfo(BaseModel):
    required: bool = Field(..., description="Whether uncertainty disclosure is required.")
    reason: str = Field(..., description="Why uncertainty is required.")

    @model_validator(mode="after")
    def _validate_reason(self) -> "UncertaintyInfo":
        if self.required and not self.reason.strip():
            raise ValueError("uncertainty.reason must be non-empty when required")
        return self


class CandidateAnswer(BaseModel):
    answer: str = Field(..., description="The user-facing final answer.")
    reasoning: str = Field(
        ...,
        description="Concise reasoning (no hidden chain-of-thought).",
    )
    applied_rules: list[AppliedRule] = Field(
        default_factory=list,
        description="Rule trace with evidence for each applied rule.",
    )
    policy_decisions: dict[str, Any] = Field(
        default_factory=dict,
        description="Deterministic policy decisions and outcomes.",
    )
    risk_flags: list[str] = Field(
        default_factory=list, description="Deterministic risk flags."
    )
    confidence_by_claim: dict[str, float] = Field(
        default_factory=dict, description="Claim_id -> confidence score (0..1)."
    )
    evidence_map: dict[str, list[str]] = Field(
        default_factory=dict, description="Claim_id -> evidence sources."
    )
    uncertainty: UncertaintyInfo | None = Field(
        default=None,
        description="Structured uncertainty disclosure (required/reason).",
    )

    @field_validator("answer", "reasoning")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer and reasoning must be non-empty")
        return value

    @field_validator("applied_rules")
    @classmethod
    def _applied_rules_present(cls, value: list[AppliedRule]) -> list[AppliedRule]:
        if not value:
            raise ValueError("applied_rules must be non-empty")
        return value

    @field_validator("confidence_by_claim")
    @classmethod
    def _confidence_range(cls, value: dict[str, float]) -> dict[str, float]:
        for claim_id, score in value.items():
            if score < 0.0 or score > 1.0:
                raise ValueError(f"confidence_by_claim[{claim_id!r}] out of range")
        return value

    @field_validator("evidence_map")
    @classmethod
    def _evidence_map_non_empty(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        for claim_id, sources in value.items():
            if not sources:
                raise ValueError(f"evidence_map[{claim_id!r}] must be non-empty")
            if any(not src.strip() for src in sources):
                raise ValueError(f"evidence_map[{claim_id!r}] contains empty sources")
        return value

    @model_validator(mode="after")
    def _evidence_map_covers_claims(self) -> "CandidateAnswer":
        if self.confidence_by_claim:
            missing = [cid for cid in self.confidence_by_claim if cid not in self.evidence_map]
            if missing:
                raise ValueError(f"evidence_map missing claims: {missing}")
        return self


class ProsecutorCritique(BaseModel):
    objections: list[str] = Field(
        default_factory=list,
        description="Aggressive objections; each should be specific and testable.",
    )
    counterexamples: list[str] = Field(default_factory=list)
    missing_assumptions: list[str] = Field(default_factory=list)
    factuality_flags: list[str] = Field(
        default_factory=list, description="Potential factual errors or unverifiable claims."
    )
    targeted_questions: list[str] = Field(
        default_factory=list, description="Questions that force clarification or proof."
    )


class DefenseRebuttal(BaseModel):
    rebuttals: list[str] = Field(default_factory=list)
    corrections: list[str] = Field(
        default_factory=list, description="Corrections to the candidate answer, if needed."
    )
    strengthened_answer: str | None = Field(
        default=None, description="Optional improved final answer after rebuttal."
    )
    resolved_objections: list[str] = Field(
        default_factory=list, description="Objection strings resolved by the defense."
    )
    unresolved_objections: list[str] = Field(
        default_factory=list, description="Objection strings the defense admits remain."
    )


class JudgeScores(BaseModel):
    correctness: float = Field(..., ge=0.0, le=1.0)
    completeness: float = Field(..., ge=0.0, le=1.0)
    reasoning_quality: float = Field(..., ge=0.0, le=1.0)
    factuality: float = Field(..., ge=0.0, le=1.0)
    robustness: float = Field(..., ge=0.0, le=1.0)

    def overall(self) -> float:
        return float(
            0.30 * self.correctness
            + 0.20 * self.completeness
            + 0.20 * self.reasoning_quality
            + 0.20 * self.factuality
            + 0.10 * self.robustness
        )


class JudgeDecision(BaseModel):
    verdict: Literal["accept", "revise", "reject"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: JudgeScores
    justification: list[str] = Field(
        default_factory=list, description="Concrete reasons referencing arguments."
    )
    key_failures: list[str] = Field(default_factory=list)
    required_fixes: list[str] = Field(
        default_factory=list, description="If verdict=revise, what must change."
    )
    consistency_checks: dict[str, bool] = Field(
        default_factory=dict,
        description="Named checks (e.g., 'no_unresolved_objections': true).",
    )
    tags: list[str] = Field(default_factory=list, description="Failure patterns / topics.")


class LLMCallRecord(BaseModel):
    role: str
    model_id: str
    prompt: str
    response_text: str
    parsed_json: dict[str, Any] | None = None
