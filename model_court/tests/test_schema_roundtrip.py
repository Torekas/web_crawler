from __future__ import annotations

from model_court.agents.contracts import CandidateAnswer, DefenseRebuttal, JudgeDecision, JudgeScores, ProsecutorCritique
from model_court.storage.schema import CaseRecord, RoundRecord


def test_case_record_roundtrip_json():
    cand = CandidateAnswer(final_answer="4", reasoning_summary=["2+2=4"], assumptions=[], citations=[], uncertainty=None)
    pros = ProsecutorCritique(objections=["show work"], counterexamples=[], missing_assumptions=[], factuality_flags=[], targeted_questions=[])
    defense = DefenseRebuttal(rebuttals=["basic arithmetic"], corrections=[], strengthened_answer=None, resolved_objections=["show work"], unresolved_objections=[])
    judge = JudgeDecision(
        verdict="accept",
        confidence=0.9,
        scores=JudgeScores(correctness=1, completeness=1, reasoning_quality=1, factuality=1, robustness=1),
        justification=["ok"],
        key_failures=[],
        required_fixes=[],
        consistency_checks={"no_unresolved_objections": True},
        tags=["unit"],
    )
    rr = RoundRecord(round_idx=0, candidate=cand, prosecutor=pros, defense=defense, judge=judge, llm_calls=[])
    case = CaseRecord(
        case_id="c1",
        experiment_id="e1",
        created_at="2025-01-01T00:00:00Z",
        seed=123,
        question={"question_id": "q1", "question": "2+2?"},
        rounds=[rr],
        final_answer="4",
        final_verdict="accept",
        final_overall_score=1.0,
        metadata={"x": 1},
    )

    s = case.model_dump_json()
    case2 = CaseRecord.model_validate_json(s)
    assert case2.case_id == "c1"
    assert case2.rounds[0].judge.verdict == "accept"

