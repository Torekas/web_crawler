import gzip
import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}


@dataclass
class Chunk:
    text: str
    url: str
    title: str
    fetched_at: str


def load_pages(pages_path: Path) -> List[Dict]:
    pages = []
    with pages_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                pages.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pages


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> Iterable[str]:
    words = text.split()
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        yield " ".join(words[start:end])
        if end == len(words):
            break
        start = max(0, end - overlap)


def get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embed_texts(texts: List[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    model = get_model(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def build_index(
    pages_path: Path,
    index_path: Path,
    model_name: str = DEFAULT_MODEL,
    chunk_size: int = 800,
    overlap: int = 150,
    min_words: int = 60,
) -> None:
    pages = load_pages(pages_path)
    chunks: List[Chunk] = []
    for page in pages:
        text = page.get("text", "")
        if len(text.split()) < min_words:
            continue
        for chunk in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            if len(chunk.split()) < min_words:
                continue
            chunks.append(
                Chunk(
                    text=chunk,
                    url=page.get("url", ""),
                    title=page.get("title", ""),
                    fetched_at=page.get("fetched_at", ""),
                )
            )

    if not chunks:
        raise ValueError("No chunks to index. Did the crawl fetch any content?")

    logger.info("Embedding %d chunks with %s", len(chunks), model_name)
    embeddings = embed_texts([c.text for c in chunks], model_name=model_name)

    payload = {
        "model_name": model_name,
        "embeddings": embeddings,
        "chunks": chunks,
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(index_path, "wb") as f:
        pickle.dump(payload, f)
    logger.info("Index saved to %s", index_path)


def load_index(index_path: Path) -> Dict:
    with gzip.open(index_path, "rb") as f:
        return pickle.load(f)


def search_index(
    index: Dict,
    query: str,
    top_k: int = 5,
    model_name: str = None,
) -> List[Tuple[float, Chunk]]:
    model_to_use = model_name or index["model_name"]
    stored_embeddings: np.ndarray = index["embeddings"]
    chunks: List[Chunk] = index["chunks"]

    query_vec = embed_texts([query], model_name=model_to_use)[0]
    scores = np.dot(stored_embeddings, query_vec)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), chunks[i]) for i in top_indices]


def search(index_path: Path, query: str, top_k: int = 5, model_name: str = None) -> List[Tuple[float, Chunk]]:
    index = load_index(index_path)
    return search_index(index, query, top_k=top_k, model_name=model_name)


def format_context(results: List[Tuple[float, Chunk]]) -> str:
    lines = []
    for idx, (score, chunk) in enumerate(results, start=1):
        lines.append(f"[{idx}] {chunk.title} - {chunk.url}")
        lines.append(chunk.text)
        lines.append("")  # spacer
    return "\n".join(lines)


def _build_chat_context(results: List[Tuple[float, Chunk]]) -> str:
    return "\n\n".join(
        f"[{idx}] {chunk.title or 'Untitled'} ({chunk.url})\n{chunk.text}"
        for idx, (_, chunk) in enumerate(results, start=1)
    )


def answer_with_ollama(
    results: List[Tuple[float, Chunk]],
    question: str,
    model: str = "mixtral:8x7b",
) -> str:
    try:
        import ollama
    except ImportError:
        raise RuntimeError("ollama package not installed; cannot use Ollama chat")

    context = _build_chat_context(results)
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions using the provided context. "
                "Cite sources by their [number] when you reference them. "
                "Keep answers concise."
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    response = ollama.chat(model=model, messages=messages, options={"temperature": 0.2})
    return response["message"]["content"]


def answer_with_openai(
    results: List[Tuple[float, Chunk]],
    question: str,
    model: str = "gpt-4o-mini",
) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed; cannot use OpenAI chat")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    context = _build_chat_context(results)

    prompt = (
        "You are an assistant that answers questions using the provided context. "
        "Cite sources by their [number] when relevant and keep responses concise.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return completion.choices[0].message.content


def interactive_chat(
    index_path: Path,
    model_name: str = None,
    top_k: int = 4,
    llm: str = "ollama",
    ollama_model: str = "mixtral:8x7b",
    openai_model: str = "gpt-4o-mini",
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    index = load_index(index_path)
    print("RAG chat ready. Type 'exit' to quit.")
    while True:
        question = input("\nQ: ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            break
        results = search_index(index, question, top_k=top_k, model_name=model_name)
        try:
            if llm == "ollama":
                answer = answer_with_ollama(results, question, model=ollama_model)
            elif llm == "openai":
                answer = answer_with_openai(results, question, model=openai_model)
            else:
                raise ValueError(f"Unknown llm backend: {llm}")
            print(f"\nA: {answer}")
        except Exception as exc:
            logger.info("LLM unavailable (%s). Showing top contexts instead.", exc)
            for idx, (score, chunk) in enumerate(results, start=1):
                print(f"[{idx}] score={score:.3f} {chunk.title} - {chunk.url}")
                print(chunk.text[:700] + ("..." if len(chunk.text) > 700 else ""))


if __name__ == "__main__":
    interactive_chat(Path("data/index.pkl.gz"))
