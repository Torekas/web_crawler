from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from model_court.storage.schema import CaseRecord
from model_court.vector_store.embedder import HashingEmbedder
from model_court.vector_store.sqlite_store import SQLiteVectorStore


@dataclass(frozen=True)
class CaseIndexer:
    vector_store: SQLiteVectorStore
    cfg: dict[str, Any]
    embedder: HashingEmbedder = field(init=False)

    def __post_init__(self) -> None:
        dim = int(self.cfg.get("dim", 1024))
        object.__setattr__(self, "embedder", HashingEmbedder(dim=dim))

    def embed_query(self, text: str):
        return self.embedder.embed(text)

    def index_case(self, case: CaseRecord) -> None:
        for doc_id, content, metadata in self._case_documents(case):
            chunks = _chunk_text(
                content,
                chunk_chars=int(self.cfg.get("chunk_chars", 1400)),
                chunk_overlap=int(self.cfg.get("chunk_overlap", 120)),
            )
            for i, ch in enumerate(chunks):
                emb = self.embedder.embed(ch)
                self.vector_store.upsert(
                    id=f"{doc_id}::chunk_{i:03d}",
                    content=ch,
                    metadata={**metadata, "chunk_idx": i, "case_id": case.case_id},
                    embedding=emb,
                )

    def _case_documents(self, case: CaseRecord):
        # Embed fields useful for retrieval: question, final answer, objections, judge justification.
        last = case.rounds[-1]
        yield (
            f"{case.case_id}::question",
            str(case.question.get("question", "")),
            {"kind": "question", "experiment_id": case.experiment_id, "source": case.question.get("source", "")},
        )
        yield (
            f"{case.case_id}::final_answer",
            str(case.final_answer),
            {"kind": "final_answer", "experiment_id": case.experiment_id},
        )
        if last.prosecutor.objections:
            yield (
                f"{case.case_id}::objections",
                "\n".join(last.prosecutor.objections),
                {"kind": "objections", "experiment_id": case.experiment_id},
            )
        if last.judge.justification:
            yield (
                f"{case.case_id}::judge_justification",
                "\n".join(last.judge.justification),
                {"kind": "judge_justification", "experiment_id": case.experiment_id, "verdict": last.judge.verdict},
            )


def _chunk_text(text: str, *, chunk_chars: int, chunk_overlap: int) -> list[str]:
    s = text.strip()
    if not s:
        return []
    if len(s) <= chunk_chars:
        return [s]
    out: list[str] = []
    start = 0
    while start < len(s):
        end = min(len(s), start + chunk_chars)
        out.append(s[start:end])
        if end == len(s):
            break
        start = max(0, end - chunk_overlap)
    return out
