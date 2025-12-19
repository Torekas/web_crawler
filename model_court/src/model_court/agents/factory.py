from __future__ import annotations

from pathlib import Path
from typing import Any

from model_court.agents.base import Agent
from model_court.agents.contracts import CandidateAnswer, DefenseRebuttal, JudgeDecision, ProsecutorCritique
from model_court.llm import MockLLM, OpenAIChatLLM
from model_court.llm.base import LLM


def build_llm(cfg: dict[str, Any]) -> LLM:
    provider = (cfg.get("models") or {}).get("provider", "mock")
    if provider == "mock":
        return MockLLM()
    if provider == "openai_chat":
        return OpenAIChatLLM()
    raise ValueError(f"Unknown models.provider={provider!r}")


def build_agents(cfg: dict[str, Any]) -> dict[str, Agent]:
    llm = build_llm(cfg)
    models = cfg.get("models") or {}
    prompt_dir = Path(__file__).resolve().parent / "prompts"

    return {
        "candidate": Agent(
            role="candidate",
            model_id=str(models.get("candidate_id", "mock-candidate")),
            llm=llm,
            output_model=CandidateAnswer,
            prompt_path=prompt_dir / "candidate.txt",
            temperature=0.0,
        ),
        "prosecutor": Agent(
            role="prosecutor",
            model_id=str(models.get("prosecutor_id", "mock-prosecutor")),
            llm=llm,
            output_model=ProsecutorCritique,
            prompt_path=prompt_dir / "prosecutor.txt",
            temperature=0.0,
        ),
        "defense": Agent(
            role="defense",
            model_id=str(models.get("defense_id", "mock-defense")),
            llm=llm,
            output_model=DefenseRebuttal,
            prompt_path=prompt_dir / "defense.txt",
            temperature=0.0,
        ),
        "judge": Agent(
            role="judge",
            model_id=str(models.get("judge_id", "mock-judge")),
            llm=llm,
            output_model=JudgeDecision,
            prompt_path=prompt_dir / "judge.txt",
            temperature=0.0,
        ),
    }

