"""Chunking utilities for markdown/text content."""

from __future__ import annotations

import hashlib
from typing import List, Sequence


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200, min_size: int = 200) -> List[dict]:
    """Chunk text into overlapping segments."""
    if not text:
        return []
    segments: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        segment = text[start:end]
        if len(segment) >= min_size:
            segments.append(segment.strip())
        start = end - overlap if end - overlap > start else end
    chunks: List[dict] = []
    for idx, seg in enumerate(segments):
        chunk_id = hashlib.sha256(f"{idx}:{seg[:80]}".encode("utf-8")).hexdigest()[:16]
        chunks.append({"chunk_id": chunk_id, "text": seg, "metadata": {"order": idx}})
    return chunks


__all__ = ["chunk_text"]
