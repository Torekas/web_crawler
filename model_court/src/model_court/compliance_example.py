from __future__ import annotations

import json

from model_court.agents.contracts import AppliedRule, CandidateAnswer
from model_court.compliance_orchestrator import run_compliance_pipeline
from model_court.rule_trace import RuleDefinition, RuleEvaluation, RuleRegistry, register_rule


def _rule_answer_present(candidate: CandidateAnswer) -> RuleEvaluation:
    passed = bool(candidate.answer.strip())
    evidence = ["answer"] if passed else ["answer_missing"]
    return RuleEvaluation(rule_id="answer_present", passed=passed, evidence=evidence)


def _rule_reasoning_present(candidate: CandidateAnswer) -> RuleEvaluation:
    passed = bool(candidate.reasoning.strip())
    evidence = ["reasoning"] if passed else ["reasoning_missing"]
    return RuleEvaluation(rule_id="reasoning_present", passed=passed, evidence=evidence)


def main() -> None:
    registry = RuleRegistry()
    register_rule(registry, RuleDefinition("answer_present", "Answer is non-empty.", _rule_answer_present))
    register_rule(registry, RuleDefinition("reasoning_present", "Reasoning is non-empty.", _rule_reasoning_present))

    candidate = CandidateAnswer(
        answer="Paris.",
        reasoning="The capital of France is Paris.",
        applied_rules=[
            AppliedRule(rule_id="answer_present", passed=True, evidence=["answer"]),
            AppliedRule(rule_id="reasoning_present", passed=True, evidence=["reasoning"]),
        ],
        policy_decisions={},
        risk_flags=[],
        confidence_by_claim={"claim_1": 0.92},
        evidence_map={"claim_1": ["world_knowledge"]},
        uncertainty=None,
    )

    final_output = run_compliance_pipeline(candidate, registry=registry)
    print(json.dumps(final_output.model_dump(), indent=2))


if __name__ == "__main__":
    main()
