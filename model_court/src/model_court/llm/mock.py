from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from model_court.llm.base import LLM


@dataclass(frozen=True)
class MockLLM(LLM):
    """
    Deterministic-ish stand-in for real LLMs so the pipeline is runnable without API keys.
    It generates JSON matching the role schema by inspecting the prompt.
    """

    def complete(self, *, prompt: str, model_id: str, temperature: float, seed: int | None) -> str:
        rng = random.Random((seed or 0) ^ (hash(model_id) & 0xFFFFFFFF))
        role = _infer_role(prompt)
        q = _extract_question(prompt)

        if role == "candidate":
            final = _mock_answer(q)
            return json.dumps(
                {
                    "answer": final,
                    "reasoning": (
                        "Parse the question and identify the target quantity. "
                        "Apply the relevant rule or calculation. "
                        "Return the result with correct formatting."
                    ),
                    "applied_rules": [
                        {
                            "rule_id": "answer_present",
                            "pass": True,
                            "evidence": ["answer"],
                        },
                        {
                            "rule_id": "reasoning_present",
                            "pass": True,
                            "evidence": ["reasoning"],
                        },
                    ],
                    "policy_decisions": {},
                    "risk_flags": [],
                    "confidence_by_claim": {"claim_1": 0.82},
                    "evidence_map": {"claim_1": ["question_text"]},
                    "uncertainty": None,
                }
            )
        if role == "prosecutor":
            return json.dumps(
                {
                    "objections": [
                        "The answer does not justify why the chosen method applies.",
                        "Edge cases and alternative interpretations are not addressed.",
                    ],
                    "counterexamples": ["If the question intends a different unit or definition, the result changes."],
                    "missing_assumptions": ["Define any implied constraints (integer-only? rounding?)."],
                    "factuality_flags": [],
                    "targeted_questions": ["What assumptions are required for the calculation to be valid?"],
                }
            )
        if role == "defense":
            return json.dumps(
                {
                    "rebuttals": [
                        "The method matches the most direct reading of the problem.",
                        "Assumptions can be made explicit without changing the result.",
                    ],
                    "corrections": [],
                    "strengthened_answer": None,
                    "resolved_objections": [
                        "The answer does not justify why the chosen method applies.",
                        "Edge cases and alternative interpretations are not addressed.",
                    ],
                    "unresolved_objections": [],
                }
            )
        if role == "judge":
            # Slight stochasticity via seed to allow different thresholds in tests.
            base = 0.78 + (rng.random() - 0.5) * 0.05
            correctness = max(0.0, min(1.0, base))
            scores = {
                "correctness": correctness,
                "completeness": correctness - 0.05,
                "reasoning_quality": correctness - 0.03,
                "factuality": 0.9,
                "robustness": correctness - 0.08,
            }
            overall = 0.30 * scores["correctness"] + 0.20 * scores["completeness"] + 0.20 * scores["reasoning_quality"] + 0.20 * scores["factuality"] + 0.10 * scores["robustness"]
            verdict = "accept" if overall >= 0.75 else "revise"
            return json.dumps(
                {
                    "verdict": verdict,
                    "confidence": float(max(0.0, min(1.0, overall))),
                    "scores": scores,
                    "justification": [
                        "Defense addressed the prosecutor's objections.",
                        "Answer is plausible and internally consistent for the given question.",
                    ],
                    "key_failures": [] if verdict == "accept" else ["Needs clearer assumptions and edge-case handling."],
                    "required_fixes": [] if verdict == "accept" else ["State assumptions and clarify interpretation."],
                    "consistency_checks": {"no_unresolved_objections": True},
                    "tags": ["mock", "demo"],
                }
            )

        return json.dumps({"text": "unrecognized role"})


def _infer_role(prompt: str) -> str:
    p = prompt.lower()
    if "you are the prosecutor" in p:
        return "prosecutor"
    if "you are the defense attorney" in p:
        return "defense"
    if "you are the judge" in p:
        return "judge"
    if "you are the candidate model" in p:
        return "candidate"
    return "unknown"


def _extract_question(prompt: str) -> str:
    m = re.search(r"\"question\"\\s*:\\s*\"(.*?)\"", prompt, re.DOTALL)
    if not m:
        return ""
    return m.group(1)


def _mock_answer(question: str) -> str:
    q = question.strip().lower()
    if "2+2" in q or "two plus two" in q:
        return "4"
    if "capital of france" in q:
        return "Paris."
    return "Answer: (mock) See reasoning summary."
