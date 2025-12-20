from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from model_court.agents.contracts import AppliedRule, CandidateAnswer


class RuleTraceError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuleEvaluation:
    rule_id: str
    passed: bool
    evidence: list[str]


@dataclass
class RuleDefinition:
    rule_id: str
    description: str
    evaluator: Callable[[CandidateAnswer], RuleEvaluation]


@dataclass
class RuleRegistry:
    rules: dict[str, RuleDefinition] = field(default_factory=dict)


def register_rule(registry: RuleRegistry, rule: RuleDefinition) -> None:
    if rule.rule_id in registry.rules:
        raise RuleTraceError(f"Rule already registered: {rule.rule_id}")
    registry.rules[rule.rule_id] = rule


def validate_applied_rules(applied_rules: list[AppliedRule]) -> None:
    if not applied_rules:
        raise RuleTraceError("applied_rules is empty")
    for rule in applied_rules:
        if not rule.rule_id.strip():
            raise RuleTraceError("applied_rules has empty rule_id")
        if not rule.evidence:
            raise RuleTraceError(f"applied_rules[{rule.rule_id}] missing evidence")


def evaluate_rules(candidate: CandidateAnswer, registry: RuleRegistry) -> list[RuleEvaluation]:
    validate_applied_rules(candidate.applied_rules)
    evaluations: list[RuleEvaluation] = []
    for applied in candidate.applied_rules:
        rule = registry.rules.get(applied.rule_id)
        if rule is None:
            raise RuleTraceError(f"Rule not registered: {applied.rule_id}")
        result = rule.evaluator(candidate)
        if not result.evidence:
            raise RuleTraceError(f"Rule {applied.rule_id} produced no evidence")
        evaluations.append(result)
    return evaluations


def attach_rule_evidence(candidate: CandidateAnswer, evaluations: list[RuleEvaluation]) -> CandidateAnswer:
    by_id = {ev.rule_id: ev for ev in evaluations}
    updated: list[AppliedRule] = []
    for applied in candidate.applied_rules:
        evaluation = by_id.get(applied.rule_id)
        if evaluation is None:
            raise RuleTraceError(f"Missing evaluation for rule: {applied.rule_id}")
        updated.append(
            AppliedRule(rule_id=evaluation.rule_id, passed=evaluation.passed, evidence=evaluation.evidence)
        )
    return candidate.model_copy(update={"applied_rules": updated})
