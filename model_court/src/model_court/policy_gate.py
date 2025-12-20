from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal

from pydantic import BaseModel, Field

from model_court.contradiction_detection import ContradictionResult


class PolicyCheckResult(BaseModel):
    policy_decisions: dict[str, str] = Field(default_factory=dict)
    risk_flags: list[str] = Field(default_factory=list)
    required_redactions: list[str] = Field(default_factory=list)


class PolicyEnforcementResult(BaseModel):
    status: Literal["approved", "redacted", "regenerate"]
    answer: str | None = None
    error_codes: list[str] = Field(default_factory=list)
    policy_decisions: dict[str, str] = Field(default_factory=dict)
    risk_flags: list[str] = Field(default_factory=list)
    required_redactions: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class PolicyGateConfig:
    domain_blocklist: Iterable[str] = field(default_factory=tuple)
    safety_terms: Iterable[str] = field(default_factory=lambda: ("kill", "harm", "bomb", "suicide", "weapon"))


_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"\\b\\+?\\d[\\d\\s().-]{7,}\\d\\b")
_SSN_RE = re.compile(r"\\b\\d{3}-\\d{2}-\\d{4}\\b")


def check_compliance(answer: str, *, config: PolicyGateConfig | None = None) -> PolicyCheckResult:
    cfg = config or PolicyGateConfig()
    required_redactions = _detect_pii(answer)
    safety_hits = _detect_safety(answer, cfg.safety_terms)
    domain_hits = _detect_domain_violations(answer, cfg.domain_blocklist)

    policy_decisions = {
        "pii": "fail" if required_redactions else "pass",
        "safety": "fail" if safety_hits else "pass",
        "domain": "fail" if domain_hits else "pass",
    }
    risk_flags: list[str] = []
    if required_redactions:
        risk_flags.append("pii_detected")
    if safety_hits:
        risk_flags.append("safety_violation")
    if domain_hits:
        risk_flags.append("domain_violation")

    return PolicyCheckResult(
        policy_decisions=policy_decisions,
        risk_flags=risk_flags + safety_hits + domain_hits,
        required_redactions=required_redactions,
    )


def enforce(
    answer: str,
    *,
    contradictions: ContradictionResult | None = None,
    config: PolicyGateConfig | None = None,
) -> PolicyEnforcementResult:
    check = check_compliance(answer, config=config)
    error_codes: list[str] = []
    policy_decisions = dict(check.policy_decisions)
    risk_flags = list(check.risk_flags)

    missing_required = list(contradictions.missing_required_fields) if contradictions else []
    contradiction_flags = list(contradictions.contradiction_flags) if contradictions else []

    if missing_required:
        policy_decisions["missing_fields"] = "fail"
        error_codes.append("missing_required_fields")
        risk_flags.append("missing_required_fields")
    if contradiction_flags:
        policy_decisions["contradictions"] = "fail"
        error_codes.append("contradiction_detected")
        risk_flags.extend(contradiction_flags)

    if check.policy_decisions.get("safety") == "fail":
        error_codes.append("safety_violation")
    if check.policy_decisions.get("domain") == "fail":
        error_codes.append("domain_violation")

    if error_codes:
        return PolicyEnforcementResult(
            status="regenerate",
            answer=None,
            error_codes=error_codes,
            policy_decisions=policy_decisions,
            risk_flags=risk_flags,
            required_redactions=check.required_redactions,
        )

    if check.required_redactions:
        redacted = _apply_redactions(answer, check.required_redactions)
        return PolicyEnforcementResult(
            status="redacted",
            answer=redacted,
            error_codes=[],
            policy_decisions=policy_decisions,
            risk_flags=risk_flags,
            required_redactions=check.required_redactions,
        )

    return PolicyEnforcementResult(
        status="approved",
        answer=answer,
        error_codes=[],
        policy_decisions=policy_decisions,
        risk_flags=risk_flags,
        required_redactions=check.required_redactions,
    )


def _detect_pii(answer: str) -> list[str]:
    matches = _EMAIL_RE.findall(answer) + _PHONE_RE.findall(answer) + _SSN_RE.findall(answer)
    return list(dict.fromkeys(matches))


def _detect_safety(answer: str, terms: Iterable[str]) -> list[str]:
    text = answer.lower()
    hits = [term for term in terms if term.lower() in text]
    return hits


def _detect_domain_violations(answer: str, blocklist: Iterable[str]) -> list[str]:
    text = answer.lower()
    hits = [term for term in blocklist if term.lower() in text]
    return hits


def _apply_redactions(answer: str, redactions: Iterable[str]) -> str:
    redacted = answer
    for token in redactions:
        if token:
            redacted = redacted.replace(token, "[REDACTED]")
    return redacted
