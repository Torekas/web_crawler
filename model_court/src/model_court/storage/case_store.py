from __future__ import annotations

import json
from pathlib import Path

from model_court.storage.schema import CaseRecord


class CaseStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def append(self, case: CaseRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(case.model_dump(), ensure_ascii=False) + "\n")

    def iter_cases(self) -> list[CaseRecord]:
        out: list[CaseRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            out.append(CaseRecord.model_validate_json(line))
        return out

