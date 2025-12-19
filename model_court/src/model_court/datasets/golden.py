from __future__ import annotations

import json
from pathlib import Path

from model_court.storage.schema import CaseRecord


def export_golden(
    *,
    cases_path: Path,
    out_path: Path,
    accept_threshold: float,
    confidence_threshold: float,
    require_no_unresolved_objections: bool = True,
) -> dict[str, int]:
    """
    Golden inclusion rules (case_v1):
    - final_verdict == "accept"
    - final_overall_score >= accept_threshold
    - final round judge.confidence >= confidence_threshold
    - (optional) defense.unresolved_objections empty
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = 0
    n_out = 0
    with cases_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            n_in += 1
            case = CaseRecord.model_validate_json(line)
            last = case.rounds[-1]
            if case.final_verdict != "accept":
                continue
            if float(case.final_overall_score) < float(accept_threshold):
                continue
            if float(last.judge.confidence) < float(confidence_threshold):
                continue
            if require_no_unresolved_objections and last.defense.unresolved_objections:
                continue
            f_out.write(json.dumps(case.model_dump(), ensure_ascii=False) + "\n")
            n_out += 1
    return {"in": n_in, "out": n_out}

