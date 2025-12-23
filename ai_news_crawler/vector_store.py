from __future__ import annotations

from pathlib import Path
from typing import Iterable
from uuid import uuid4

import chromadb
from sentence_transformers import SentenceTransformer


class VectorArchive:
    """Store and query article chunks in a persistent ChromaDB collection."""

    def __init__(
        self,
        db_dir: Path,
        collection_name: str = "ai_news",
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        self.collection = self.client.get_or_create_collection(collection_name)
        self.model = SentenceTransformer(model_name)
        self.max_batch_size = 5000

    def add_documents(self, documents: Iterable[dict]) -> int:
        """Chunk documents, embed them, and add them to the vector store."""
        texts: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        for doc in documents:
            content = doc.get("content", "")
            if not content:
                continue
            summary = self._trim_summary(doc.get("summary", ""))
            chunks = self._chunk_text(content)
            for idx, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append(
                    {
                        "url": doc.get("url", ""),
                        "date": doc.get("date", ""),
                        "title": doc.get("title", ""),
                        "summary": summary,
                        "chunk": idx,
                    }
                )
                ids.append(str(uuid4()))

        if not texts:
            return 0

        total_added = 0
        for start in range(0, len(texts), self.max_batch_size):
            end = start + self.max_batch_size
            batch_texts = texts[start:end]
            batch_metadatas = metadatas[start:end]
            batch_ids = ids[start:end]
            embeddings = self.model.encode(batch_texts, batch_size=64).tolist()
            self.collection.add(
                documents=batch_texts,
                metadatas=batch_metadatas,
                embeddings=embeddings,
                ids=batch_ids,
            )
            total_added += len(batch_texts)
        return total_added

    def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        """Return the most similar chunks for a given query."""
        embedding = self.model.encode([query_text]).tolist()
        results = self.collection.query(
            query_embeddings=embedding, n_results=n_results
        )
        return self._format_results(results)

    def _format_results(self, results: dict) -> list[dict]:
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])

        if not documents:
            return []

        output: list[dict] = []
        for idx, doc in enumerate(documents[0]):
            meta = metadatas[0][idx] if metadatas else {}
            distance = distances[0][idx] if distances else None
            output.append({"text": doc, "metadata": meta, "distance": distance})
        return output

    def _trim_summary(self, text: str, max_chars: int = 400) -> str:
        if not text:
            return ""
        text = text.strip()
        if len(text) <= max_chars:
            return text
        trimmed = text[:max_chars].rsplit(" ", 1)[0]
        return trimmed or text[:max_chars]

    def _chunk_text(
        self, text: str, max_chars: int = 1000, overlap: int = 200
    ) -> list[str]:
        """Split text into overlapping chunks for embedding."""
        if not text:
            return []

        text = text.strip()
        if len(text) <= max_chars:
            return [text]

        if overlap >= max_chars:
            overlap = max_chars // 3

        chunks: list[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(length, start + max_chars)
            if end < length:
                split = text.rfind(" ", start, end)
                if split != -1 and split > start + max_chars * 0.5:
                    end = split
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= length:
                break
            start = max(0, end - overlap)
            if start >= end:
                start = end

        return chunks
