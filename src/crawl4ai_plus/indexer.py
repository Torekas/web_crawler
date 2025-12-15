"""ChromaDB indexer abstraction with pluggable embedding backends."""

from __future__ import annotations

import hashlib
import logging
from typing import Iterable, List, Sequence

import chromadb
import httpx

logger = logging.getLogger(__name__)


class EmbeddingBackend:
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError


class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return self.model.encode(list(texts), convert_to_numpy=False, show_progress_bar=False).tolist()


class OllamaEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")
        self.client = httpx.Client(timeout=120)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            resp = self.client.post(f"{self.host}/api/embeddings", json={"model": self.model, "prompt": text})
            resp.raise_for_status()
            payload = resp.json()
            vectors.append(payload.get("embedding", []))
        return vectors


class HashEmbeddingBackend(EmbeddingBackend):
    """Deterministic lightweight embedding for tests when real models are unavailable."""

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [int(b) / 255.0 for b in h[:32]]
            vectors.append(vec)
        return vectors


def resolve_backend(name: str, model: str, ollama_host: str | None = None) -> EmbeddingBackend:
    if name == "sentence-transformers":
        return SentenceTransformerBackend(model)
    if name == "ollama":
        return OllamaEmbeddingBackend(model=model, host=ollama_host or "http://localhost:11434")
    return HashEmbeddingBackend()


class ChromaIndexer:
    def __init__(
        self,
        path: str,
        collection_name: str,
        embedding_backend: EmbeddingBackend,
    ) -> None:
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.backend = embedding_backend

    def add_chunks(self, chunks: Iterable[dict]) -> None:
        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        metadata = [c.get("metadata", {}) for c in chunks]
        embeddings = self.backend.embed(texts)
        self.collection.upsert(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadata)

    def query(self, query: str, top_k: int = 4) -> List[dict]:
        embeddings = self.backend.embed([query])
        results = self.collection.query(query_embeddings=embeddings, n_results=top_k)
        hits: List[dict] = []
        for idx, _id in enumerate(results.get("ids", [[]])[0]):
            hits.append(
                {
                    "id": _id,
                    "text": results["documents"][0][idx],
                    "score": results["distances"][0][idx] if "distances" in results else None,
                    "metadata": results["metadatas"][0][idx],
                }
            )
        return hits


__all__ = ["ChromaIndexer", "EmbeddingBackend", "resolve_backend", "HashEmbeddingBackend", "SentenceTransformerBackend"]
