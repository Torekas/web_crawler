from __future__ import annotations

import json
from pathlib import Path

from model_court.simulation.orchestrator import run_experiment
from model_court.vector_store.sqlite_store import SQLiteVectorStore


def test_run_experiment_creates_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {
        "experiment": {
            "seed": 1,
            "max_rounds": 1,
            "n_questions": 2,
            "output_dir": "datasets/raw",
            "runs_dir": "runs",
        },
        "models": {
            "provider": "mock",
            "candidate_id": "mock-candidate",
            "prosecutor_id": "mock-prosecutor",
            "defense_id": "mock-defense",
            "judge_id": "mock-judge",
        },
        "question_source": {"type": "synthetic_math_v1", "difficulty": "easy"},
        "vector_store": {"dim": 64, "chunk_chars": 200, "chunk_overlap": 20},
        "judge": {"accept_threshold": 0.75, "confidence_threshold": 0.75},
    }

    result = run_experiment(cfg)
    cases_path = Path(result["cases_path"])
    db_path = Path(result["vector_db_path"])
    assert cases_path.exists()
    assert db_path.exists()

    # JSONL parses
    lines = cases_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    obj = json.loads(lines[0])
    assert obj["schema_version"] == "case_v1"

    # Vector query returns something
    store = SQLiteVectorStore(db_path)
    hits = store.query(text="math addition", k=3)
    assert len(hits) >= 1

