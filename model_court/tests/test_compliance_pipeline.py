from __future__ import annotations

from model_court.agents.contracts import AppliedRule, CandidateAnswer
from model_court.compliance_orchestrator import run_compliance_pipeline
from model_court.rule_trace import RuleDefinition, RuleEvaluation, RuleRegistry, register_rule


def _rule_answer_present(candidate: CandidateAnswer) -> RuleEvaluation:
    passed = bool(candidate.answer.strip())
    evidence = ["answer"] if passed else ["answer_missing"]
    return RuleEvaluation(rule_id="answer_present", passed=passed, evidence=evidence)


def test_low_confidence_requires_uncertainty():
    registry = RuleRegistry()
    register_rule(registry, RuleDefinition("answer_present", "Answer is non-empty.", _rule_answer_present))

    candidate = CandidateAnswer(
        answer="Paris.",
        reasoning="The capital of France is Paris.",
        applied_rules=[AppliedRule(rule_id="answer_present", passed=True, evidence=["answer"])],
        policy_decisions={},
        risk_flags=[],
        confidence_by_claim={"claim_1": 0.4},
        evidence_map={"claim_1": ["world_knowledge"]},
        uncertainty=None,
    )

    output = run_compliance_pipeline(candidate, registry=registry, confidence_threshold=0.6)
    assert output.merged_final_answer.uncertainty is not None
    assert output.merged_final_answer.uncertainty.required is True
    assert output.merged_final_answer.uncertainty.reason == "low confidence"


def test_contradiction_triggers_regeneration():
    registry = RuleRegistry()
    register_rule(registry, RuleDefinition("answer_present", "Answer is non-empty.", _rule_answer_present))

    candidate = CandidateAnswer(
        answer="Yes.",
        reasoning="No, that is not correct.",
        applied_rules=[AppliedRule(rule_id="answer_present", passed=True, evidence=["answer"])],
        policy_decisions={},
        risk_flags=[],
        confidence_by_claim={"claim_1": 0.9},
        evidence_map={"claim_1": ["model_assertion"]},
        uncertainty=None,
    )

    output = run_compliance_pipeline(candidate, registry=registry)
    assert output.deterministic_validation.enforcement_status == "regenerate"
    assert "contradiction_detected" in output.deterministic_validation.error_codes
