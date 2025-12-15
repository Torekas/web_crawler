"""Simple RAG chat over Chroma using Ollama mixtral 8x7b."""

from __future__ import annotations

import json
import logging
from typing import List

import httpx

from .indexer import ChromaIndexer

logger = logging.getLogger(__name__)


def build_prompt(query: str, contexts: List[dict]) -> str:
    context_block = "\n\n".join(
        [f"[{idx+1}] {c['text']}\nMETA: {json.dumps(c.get('metadata', {}))}" for idx, c in enumerate(contexts)]
    )
    return (
        "You are an AI assistant.\n"
        "Answer the question using only the context. Cite sources as [n].\n"
        f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"
    )


def chat_once(
    query: str,
    indexer: ChromaIndexer,
    ollama_host: str = "http://localhost:11434",
    model: str = "mixtral:8x7b",
    top_k: int = 4,
) -> str:
    contexts = indexer.query(query, top_k=top_k)
    prompt = build_prompt(query, contexts)
    try:
        with httpx.Client(timeout=120) as client:
            # Explicitly disable streaming so the response is a single JSON object (avoids JSON decode errors).
            payload = {"model": model, "stream": False, "messages": [{"role": "user", "content": prompt}]}
            resp = client.post(f"{ollama_host.rstrip('/')}/api/chat", json=payload)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            return content
    except Exception as exc:  # noqa: BLE001
        logger.error("Chat call failed, returning fallback: %s", exc)
        return prompt


__all__ = ["chat_once", "build_prompt"]
