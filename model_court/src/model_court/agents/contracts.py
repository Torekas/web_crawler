from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentMeta(BaseModel):
    agent_role: str
    model_id: str
    temperature: float = 0.0
    seed: int | None = None


class CandidateAnswer(BaseModel):
    final_answer: str = Field(..., description="The user-facing final answer.")
    reasoning_summary: list[str] = Field(
        default_factory=list,
        description="Concise bullet reasoning summary (not hidden chain-of-thought).",
    )
    assumptions: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list, description="Optional URLs/refs if used.")
    uncertainty: str | None = Field(
        default=None, description="What the model is uncertain about, if anything."
    )


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

