from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, Field

from model_court.agents.contracts import AppliedRule, CandidateAnswer, UncertaintyInfo
from model_court.contradiction_detection import ContradictionResult, detect_contradictions
from model_court.policy_gate import PolicyEnforcementResult, PolicyGateConfig, enforce
from model_court.rule_trace import RuleRegistry, attach_rule_evidence, evaluate_rules, validate_applied_rules


class DeterministicValidation(BaseModel):
    applied_rules: list[AppliedRule] = Field(default_factory=list)
    policy_decisions: dict[str, str] = Field(default_factory=dict)
    risk_flags: list[str] = Field(default_factory=list)
    required_redactions: list[str] = Field(default_factory=list)
    contradiction_flags: list[str] = Field(default_factory=list)
    conflicting_claims: list[str] = Field(default_factory=list)
    missing_required_fields: list[str] = Field(default_factory=list)
    enforcement_status: str = "approved"
    error_codes: list[str] = Field(default_factory=list)


class FinalOutput(BaseModel):
    model_proposal: CandidateAnswer
    deterministic_validation: DeterministicValidation
    merged_final_answer: CandidateAnswer


def run_compliance_pipeline(
    candidate: CandidateAnswer,
    *,
    registry: RuleRegistry | None = None,
    confidence_threshold: float = 0.6,
    required_fields: Iterable[str] | None = None,
    policy_config: PolicyGateConfig | None = None,
) -> FinalOutput:
    model_proposal = candidate.model_copy(deep=True)

    if registry is not None:
        evaluations = evaluate_rules(candidate, registry)
        candidate = attach_rule_evidence(candidate, evaluations)
    else:
        validate_applied_rules(candidate.applied_rules)

    contradictions = detect_contradictions(candidate, required_fields=required_fields)
    enforcement = enforce(
        candidate.answer,
        contradictions=contradictions,
        config=policy_config,
    )

    merged = _merge_candidate(candidate, enforcement, contradictions, confidence_threshold)
    validation = DeterministicValidation(
        applied_rules=merged.applied_rules,
        policy_decisions=enforcement.policy_decisions,
        risk_flags=enforcement.risk_flags,
        required_redactions=enforcement.required_redactions,
        contradiction_flags=contradictions.contradiction_flags,
        conflicting_claims=contradictions.conflicting_claims,
        missing_required_fields=contradictions.missing_required_fields,
        enforcement_status=enforcement.status,
        error_codes=enforcement.error_codes,
    )

    return FinalOutput(
        model_proposal=model_proposal,
        deterministic_validation=validation,
        merged_final_answer=merged,
    )


def _merge_candidate(
    candidate: CandidateAnswer,
    enforcement: PolicyEnforcementResult,
    contradictions: ContradictionResult,
    confidence_threshold: float,
) -> CandidateAnswer:
    updated_flags = _merge_flags(candidate.risk_flags, enforcement.risk_flags, contradictions)
    updated_policy = dict(candidate.policy_decisions)
    updated_policy.update(enforcement.policy_decisions)
    if contradictions.contradiction_flags:
        updated_policy.setdefault("contradictions", "fail")
    if contradictions.missing_required_fields:
        updated_policy.setdefault("missing_fields", "fail")

    updated_evidence_map = _merge_evidence_map(candidate.evidence_map, candidate.applied_rules)
    uncertainty = _resolve_uncertainty(candidate, confidence_threshold)

    updated = {
        "policy_decisions": updated_policy,
        "risk_flags": updated_flags,
        "evidence_map": updated_evidence_map,
        "uncertainty": uncertainty,
    }

    if enforcement.status == "redacted" and enforcement.answer is not None:
        updated["answer"] = enforcement.answer
    elif enforcement.status == "regenerate":
        updated["risk_flags"] = updated_flags + ["regeneration_required"]

    return candidate.model_copy(update=updated)


def _resolve_uncertainty(candidate: CandidateAnswer, confidence_threshold: float) -> UncertaintyInfo | None:
    low_confidence = any(score < confidence_threshold for score in candidate.confidence_by_claim.values())
    rule_failures = [rule.rule_id for rule in candidate.applied_rules if rule.passed is False]

    if low_confidence:
        return UncertaintyInfo(required=True, reason="low confidence")
    if rule_failures:
        return UncertaintyInfo(required=True, reason="rule verification failure")
    return candidate.uncertainty


def _merge_flags(
    existing: list[str],
    enforcement_flags: list[str],
    contradictions: ContradictionResult,
) -> list[str]:
    merged = list(dict.fromkeys(existing + enforcement_flags))
    for flag in contradictions.contradiction_flags:
        if flag not in merged:
            merged.append(flag)
    if contradictions.missing_required_fields and "missing_required_fields" not in merged:
        merged.append("missing_required_fields")
    return merged


def _merge_evidence_map(
    evidence_map: dict[str, list[str]],
    applied_rules: list[AppliedRule],
) -> dict[str, list[str]]:
    merged = {key: list(values) for key, values in evidence_map.items()}
    for rule in applied_rules:
        if rule.rule_id not in merged:
            merged[rule.rule_id] = list(rule.evidence)
    return merged
