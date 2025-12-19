from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from model_court.vector_store.embedder import HashingEmbedder


@dataclass
class SQLiteVectorStore:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.path))
        con.execute("PRAGMA journal_mode=WAL;")
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                  id TEXT PRIMARY KEY,
                  content TEXT NOT NULL,
                  metadata_json TEXT NOT NULL,
                  embedding BLOB NOT NULL,
                  dim INTEGER NOT NULL
                );
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_vectors_dim ON vectors(dim);")

    def upsert(self, *, id: str, content: str, metadata: dict[str, Any], embedding: np.ndarray) -> None:
        emb = np.asarray(embedding, dtype=np.float32)
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO vectors(id, content, metadata_json, embedding, dim)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  content=excluded.content,
                  metadata_json=excluded.metadata_json,
                  embedding=excluded.embedding,
                  dim=excluded.dim;
                """,
                (id, content, json.dumps(metadata, ensure_ascii=False), emb.tobytes(), int(emb.shape[0])),
            )

    def query(self, *, text: str, k: int = 5) -> list[dict[str, Any]]:
        dim = self._infer_dim()
        emb = HashingEmbedder(dim=dim).embed(text)
        return self.query_embedded(embedding=emb, k=k)

    def query_embedded(self, *, embedding: np.ndarray, k: int = 5, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        emb = np.asarray(embedding, dtype=np.float32)
        dim = int(emb.shape[0])
        where = where or {}
        rows: list[tuple[str, str, str, bytes]] = []
        with self._connect() as con:
            cur = con.execute("SELECT id, content, metadata_json, embedding FROM vectors WHERE dim=?", (dim,))
            rows = list(cur.fetchall())

        sims: list[tuple[float, str, str, dict[str, Any]]] = []
        for rid, content, meta_json, blob in rows:
            meta = json.loads(meta_json)
            if not _meta_matches(meta, where):
                continue
            v = np.frombuffer(blob, dtype=np.float32)
            score = float(np.dot(emb, v))  # cosine if both L2-normalized
            sims.append((score, rid, content, meta))
        sims.sort(key=lambda x: x[0], reverse=True)
        out: list[dict[str, Any]] = []
        for score, rid, content, meta in sims[:k]:
            out.append({"id": rid, "score": score, "content": content, "metadata": meta})
        return out

    def _infer_dim(self) -> int:
        with self._connect() as con:
            cur = con.execute("SELECT dim FROM vectors LIMIT 1;")
            row = cur.fetchone()
        return int(row[0]) if row else 1024


def _meta_matches(meta: dict[str, Any], where: dict[str, Any]) -> bool:
    for k, v in where.items():
        if meta.get(k) != v:
            return False
    return True
