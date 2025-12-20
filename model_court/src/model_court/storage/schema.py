from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from model_court.agents.contracts import (
    CandidateAnswer,
    DefenseRebuttal,
    JudgeDecision,
    LLMCallRecord,
    ProsecutorCritique,
)


class RoundRecord(BaseModel):
    round_idx: int = Field(..., ge=0)
    candidate: CandidateAnswer
    prosecutor: ProsecutorCritique
    defense: DefenseRebuttal
    judge: JudgeDecision
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    compliance: dict[str, Any] = Field(
        default_factory=dict, description="Deterministic validation outputs per round."
    )


class CaseRecord(BaseModel):
    """
    Canonical JSONL schema for a single "case".

    Stored for every case:
    - question: `question_id`, `question`, optional `answer_key`, `source`, `metadata`
    - full court transcript per round: candidate/prosecutor/defense/judge + raw LLM prompts/responses
    - final outputs and scores: `final_answer`, `final_verdict`, `final_overall_score`
    - metadata: config snapshot, timestamps, model ids, seeds
    """

    schema_version: str = "case_v1"
    case_id: str
    experiment_id: str
    created_at: str
    seed: int

    question: dict[str, Any]
    rounds: list[RoundRecord]

    final_answer: str
    final_verdict: Literal["accept", "revise", "reject"]
    final_overall_score: float = Field(..., ge=0.0, le=1.0)

    metadata: dict[str, Any] = Field(default_factory=dict)
