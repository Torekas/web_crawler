from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_court.simulation.orchestrator import run_experiment
from model_court.storage.case_store import CaseStore
from model_court.utils.config import load_config
from model_court.vector_store.sqlite_store import SQLiteVectorStore


def _parse_args_demo(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("model-court-demo")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--n", type=int, default=3, help="Number of questions")
    return p.parse_args(argv)


def _parse_args_run(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("model-court-run")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--n", type=int, default=None, help="Override n_questions from config")
    return p.parse_args(argv)


def _parse_args_export(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("model-court-export-golden")
    p.add_argument("--cases", required=True, help="Path to cases.jsonl")
    p.add_argument("--out", required=True, help="Output JSONL path")
    p.add_argument("--accept-threshold", type=float, default=0.75)
    p.add_argument("--confidence-threshold", type=float, default=0.75)
    p.add_argument("--allow-unresolved-objections", action="store_true")
    return p.parse_args(argv)


def _parse_args_query(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("model-court-query")
    p.add_argument("--db", required=True, help="Path to vectors.sqlite")
    p.add_argument("--text", required=True, help="Query text")
    p.add_argument("--k", type=int, default=5)
    return p.parse_args(argv)


def main_demo(argv: list[str] | None = None) -> int:
    args = _parse_args_demo(argv)
    cfg = load_config(Path(args.config))
    cfg["experiment"]["n_questions"] = int(args.n)
    result = run_experiment(cfg)
    print(json.dumps(result, indent=2))
    return 0


def main_run(argv: list[str] | None = None) -> int:
    args = _parse_args_run(argv)
    cfg = load_config(Path(args.config))
    if args.n is not None:
        cfg["experiment"]["n_questions"] = int(args.n)
    result = run_experiment(cfg)
    print(json.dumps(result, indent=2))
    return 0


def main_export_golden(argv: list[str] | None = None) -> int:
    from model_court.datasets.golden import export_golden

    args = _parse_args_export(argv)
    export_golden(
        cases_path=Path(args.cases),
        out_path=Path(args.out),
        accept_threshold=float(args.accept_threshold),
        confidence_threshold=float(args.confidence_threshold),
        require_no_unresolved_objections=not bool(args.allow_unresolved_objections),
    )
    return 0


def main_query(argv: list[str] | None = None) -> int:
    args = _parse_args_query(argv)
    store = SQLiteVectorStore(Path(args.db))
    hits = store.query(text=args.text, k=int(args.k))
    for h in hits:
        print(json.dumps(h, ensure_ascii=False))
    return 0
