from __future__ import annotations

import uuid
from datetime import datetime, timezone


def new_experiment_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"exp_{ts}_{uuid.uuid4().hex[:8]}"


def new_case_id(experiment_id: str, idx: int) -> str:
    return f"{experiment_id}_case_{idx:05d}"

