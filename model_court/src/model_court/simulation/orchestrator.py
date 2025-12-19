from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from model_court.agents.factory import build_agents
from model_court.question_gen import JSONLBenchmarkAdapter, SyntheticMathV1
from model_court.storage.case_store import CaseStore
from model_court.storage.schema import CaseRecord, RoundRecord
from model_court.utils.ids import new_case_id, new_experiment_id
from model_court.utils.time import now_iso
from model_court.vector_store.indexing import CaseIndexer
from model_court.vector_store.sqlite_store import SQLiteVectorStore


def run_experiment(cfg: dict[str, Any]) -> dict[str, Any]:
    exp_cfg = cfg.get("experiment") or {}
    seed = int(exp_cfg.get("seed", 0))
    max_rounds = int(exp_cfg.get("max_rounds", 1))
    n_questions = int(exp_cfg.get("n_questions", 1))

    experiment_id = new_experiment_id()
    started_at = now_iso()

    project_root = Path.cwd()
    output_root = project_root / str(exp_cfg.get("output_dir", "datasets/raw"))
    runs_root = project_root / str(exp_cfg.get("runs_dir", "runs"))
    case_dir = output_root / experiment_id
    case_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    cases_path = case_dir / "cases.jsonl"

    vector_db_path = runs_root / experiment_id / "vectors.sqlite"
    vector_db_path.parent.mkdir(parents=True, exist_ok=True)
    vector_store = SQLiteVectorStore(vector_db_path)
    indexer = CaseIndexer(vector_store=vector_store, cfg=cfg.get("vector_store") or {})

    agents = build_agents(cfg)
    question_source = _build_question_source(cfg, project_root=project_root)

    questions = question_source.sample(n=n_questions, seed=seed)
    store = CaseStore(cases_path)

    accepted = 0
    for idx, q in enumerate(tqdm(questions, desc="ModelCourt", unit="case")):
        case_id = new_case_id(experiment_id, idx)
        case_seed = seed + idx
        case = _run_case(
            cfg=cfg,
            experiment_id=experiment_id,
            case_id=case_id,
            case_seed=case_seed,
            max_rounds=max_rounds,
            question=q.model_dump(),
            agents=agents,
        )
        store.append(case)
        indexer.index_case(case)
        if case.final_verdict == "accept":
            accepted += 1

    finished_at = now_iso()
    return {
        "experiment_id": experiment_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "n_questions": n_questions,
        "accepted": accepted,
        "cases_path": str(cases_path),
        "vector_db_path": str(vector_db_path),
    }


def _build_question_source(cfg: dict[str, Any], *, project_root: Path):
    qs = cfg.get("question_source") or {}
    t = qs.get("type", "synthetic_math_v1")
    if t == "synthetic_math_v1":
        return SyntheticMathV1(difficulty=str(qs.get("difficulty", "mixed")))
    if t == "jsonl_benchmark":
        path = project_root / str(qs["path"])
        return JSONLBenchmarkAdapter(path=path, source_name=str(qs.get("source_name", "benchmark")))
    raise ValueError(f"Unknown question_source.type={t!r}")


def _run_case(
    *,
    cfg: dict[str, Any],
    experiment_id: str,
    case_id: str,
    case_seed: int,
    max_rounds: int,
    question: dict[str, Any],
    agents: dict[str, Any],
) -> CaseRecord:
    rounds: list[RoundRecord] = []
    candidate_answer_text: str | None = None

    for round_idx in range(max_rounds):
        candidate_input = {
            "question": question["question"],
            "prior_rounds": [r.model_dump() for r in rounds],
            "instruction": (
                "Provide the best answer."
                if round_idx == 0
                else "Revise your answer using the judge's required_fixes and prosecutor objections."
            ),
        }
        cand, cand_call = agents["candidate"].run(input_json=candidate_input, seed=case_seed + 10 * round_idx)
        candidate_answer_text = cand.final_answer

        pros_input = {
            "question": question["question"],
            "candidate_answer": cand.model_dump(),
        }
        pros, pros_call = agents["prosecutor"].run(input_json=pros_input, seed=case_seed + 10 * round_idx + 1)

        def_input = {
            "question": question["question"],
            "candidate_answer": cand.model_dump(),
            "prosecutor": pros.model_dump(),
        }
        defense, def_call = agents["defense"].run(input_json=def_input, seed=case_seed + 10 * round_idx + 2)

        judge_input = {
            "question": question["question"],
            "candidate_answer": (defense.strengthened_answer or cand.final_answer),
            "candidate_structured": cand.model_dump(),
            "prosecutor": pros.model_dump(),
            "defense": defense.model_dump(),
            "rubric": cfg.get("judge") or {},
        }
        judge, judge_call = agents["judge"].run(input_json=judge_input, seed=case_seed + 10 * round_idx + 3)

        rounds.append(
            RoundRecord(
                round_idx=round_idx,
                candidate=cand,
                prosecutor=pros,
                defense=defense,
                judge=judge,
                llm_calls=[cand_call, pros_call, def_call, judge_call],
            )
        )

        if judge.verdict == "accept":
            break

    final_round = rounds[-1]
    final_answer = final_round.defense.strengthened_answer or final_round.candidate.final_answer
    overall = final_round.judge.scores.overall()
    return CaseRecord(
        case_id=case_id,
        experiment_id=experiment_id,
        created_at=now_iso(),
        seed=case_seed,
        question=question,
        rounds=rounds,
        final_answer=final_answer,
        final_verdict=final_round.judge.verdict,
        final_overall_score=overall,
        metadata={
            "config_snapshot": json.loads(json.dumps(cfg)),  # ensure JSON-serializable
        },
    )

