import gzip
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

from . import prompts
from .llm import LLMClient
from .memory import MemoryStore, ShortTermMemory

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
XAI_KEYWORDS = [
    "xai",
    "explain",
    "explanation",
    "explainability",
    "interpretability",
    "interpretable",
    "saliency",
    "attribution",
    "lime",
    "shap",
    "counterfactual",
    "transparency",
    "faithfulness",
    "trustworthy",
]
NEWS_TRIGGERS = ("latest", "today", "recent", "news", "update", "updates", "release", "released", "announced")


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


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
    device = resolve_device()
    cache_key = f"{model_name}:{device}"
    if cache_key not in _MODEL_CACHE:
        logger.info("Loading embeddings model %s on device %s", model_name, device)
        _MODEL_CACHE[cache_key] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[cache_key]


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

    logger.info("Embedding %d chunks with %s on %s", len(chunks), model_name, resolve_device())
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
    model_name: Optional[str] = None,
    query_embedding: Optional[np.ndarray] = None,
) -> List[Tuple[float, Chunk]]:
    model_to_use = model_name or index["model_name"]
    stored_embeddings: np.ndarray = index["embeddings"]
    chunks: List[Chunk] = index["chunks"]

    query_vec = query_embedding if query_embedding is not None else embed_texts([query], model_name=model_to_use)[0]
    scores = np.dot(stored_embeddings, query_vec)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), chunks[i]) for i in top_indices]


def search(index_path: Path, query: str, top_k: int = 5, model_name: Optional[str] = None) -> List[Tuple[float, Chunk]]:
    index = load_index(index_path)
    return search_index(index, query, top_k=top_k, model_name=model_name)


def format_context(results: List[Tuple[float, Chunk]]) -> str:
    lines = []
    for idx, (score, chunk) in enumerate(results, start=1):
        lines.append(f"[{idx}] {chunk.title} - {chunk.url} (fetched {chunk.fetched_at})")
        lines.append(chunk.text)
        lines.append("")  # spacer
    return "\n".join(lines)


def _build_chat_context(results: List[Tuple[float, Chunk]]) -> str:
    return "\n\n".join(
        f"[{idx}] {chunk.title or 'Untitled'} ({chunk.url})\nFetched: {chunk.fetched_at}\n{chunk.text}"
        for idx, (_, chunk) in enumerate(results, start=1)
    )


def _is_xai_relevant(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in XAI_KEYWORDS)


def _strip_html(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines), title


class LiveNewsFetcher:
    """Best-effort live fetcher to add fresh AI news snippets to retrieval."""

    def __init__(
        self,
        model_name: str,
        max_results: int = 4,
        min_words: int = 80,
        max_words: int = 1200,
        timeout: int = 10,
    ) -> None:
        self.model_name = model_name
        self.max_results = max_results
        self.min_words = min_words
        self.max_words = max_words
        self.timeout = timeout
        self.session = requests.Session()
        self.headers = {"User-Agent": "AI-NewsRAG/0.4 (+https://github.com/Torekas/web_crawler)"}

    def _decode_link(self, href: str) -> Optional[str]:
        if "uddg=" in href:
            qs = urlparse(href).query
            decoded = parse_qs(qs).get("uddg", [])
            if decoded:
                return decoded[0]
        return href

    def _search_urls(self, query: str) -> List[str]:
        try:
            resp = self.session.get(
                "https://duckduckgo.com/html",
                params={"q": query, "ia": "web"},
                headers=self.headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.debug("Live search failed: %s", exc)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        urls: List[str] = []
        for anchor in soup.select("a.result__a"):
            href = anchor.get("href", "")
            if not href:
                continue
            decoded = self._decode_link(href)
            if not decoded or not decoded.startswith(("http://", "https://")):
                continue
            if decoded in urls:
                continue
            urls.append(decoded)
            if len(urls) >= self.max_results:
                break
        return urls

    def _fetch_chunk(self, url: str) -> Optional[Chunk]:
        try:
            resp = self.session.get(url, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.debug("Live fetch failed for %s: %s", url, exc)
            return None

        text, title = _strip_html(resp.text)
        words = text.split()
        if len(words) < self.min_words:
            return None
        limited_text = " ".join(words[: self.max_words])
        fetched_at = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        return Chunk(text=limited_text, url=url, title=title, fetched_at=fetched_at)

    def search(self, query: str, query_vec: np.ndarray) -> List[Tuple[float, Chunk]]:
        urls = self._search_urls(query)
        if not urls:
            return []

        chunks: List[Chunk] = []
        for url in urls:
            chunk = self._fetch_chunk(url)
            if chunk:
                chunks.append(chunk)

        if not chunks:
            return []

        embeddings = embed_texts([c.text for c in chunks], model_name=self.model_name)
        scores = np.dot(embeddings, query_vec)
        scored = [(float(score), chunk) for score, chunk in zip(scores, chunks)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored


class ConversationalRAGAgent:
    def __init__(
        self,
        index: Dict,
        model_name: Optional[str],
        llm_client: LLMClient,
        top_k: int = 4,
        min_score: float = 0.18,
        freshness_half_life_days: float = 60.0,
        validate_urls: bool = True,
        memory_store: Optional[MemoryStore] = None,
        display_char_limit: Optional[int] = None,
        enable_live_news: bool = True,
    ) -> None:
        self.index = index
        self.model_name = model_name or index["model_name"]
        self.llm_client = llm_client
        self.top_k = top_k
        self.min_score = min_score
        self.freshness_half_life_days = freshness_half_life_days
        self.validate_urls = validate_urls
        self.memory_store = memory_store or MemoryStore()
        self.short_memory = ShortTermMemory()
        self.last_results: List[Tuple[float, Chunk]] = []
        self.display_char_limit = display_char_limit
        self.live_fetcher = LiveNewsFetcher(self.model_name) if enable_live_news else None

    def _build_messages(self, question: str) -> List[dict]:
        reflections = self.memory_store.recent(kind="reflection", limit=5)
        long_term = "\n".join(f"- {e.content}" for e in reflections) or "none"
        conversation = self.short_memory.conversation_text() or "none"
        context_block = _build_chat_context(self.last_results)
        user_content = (
            f"Retrieved context (with [n] references):\n{context_block}\n\n"
            f"Long-term memory (reflections/facts):\n{long_term}\n\n"
            f"Short-term conversation:\n{conversation}\n\n"
            f"Question: {question}\n"
            "Use brief chain-of-thought internally, then write a concise but fact-rich answer (2-6 sentences) weaving in concrete details from the references. "
            "Cite sources inline as [n] and only rely on the provided references. Prefer fresher fetched_at entries and avoid unreachable links. "
            "If context is insufficient, say so and propose what to crawl or verify next."
        )
        return [
            {"role": "system", "content": prompts.build_answer_system_prompt()},
            {"role": "user", "content": user_content},
        ]

    def retrieve(self, question: str) -> List[Tuple[float, Chunk]]:
        query_vec = embed_texts([question], model_name=self.model_name)[0]
        raw_results = search_index(
            self.index,
            question,
            top_k=max(self.top_k * 2, 6),
            model_name=self.model_name,
            query_embedding=query_vec,
        )
        rescored: List[Tuple[float, Chunk]] = []
        now = datetime.now(timezone.utc)

        def recency_weight(fetched_at: str) -> float:
            try:
                dt = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except Exception:
                return 0.0
            age_days = max((now - dt).total_seconds() / 86400.0, 0.0)
            # Exponential decay with configurable half-life
            return np.exp(-age_days / self.freshness_half_life_days)

        for score, chunk in raw_results:
            freshness = recency_weight(chunk.fetched_at)
            combined = float(score + 0.15 * freshness)
            rescored.append((combined, chunk))

        rescored.sort(key=lambda x: x[0], reverse=True)
        rescored = rescored[: self.top_k]

        # Drop clearly irrelevant hits for XAI-focused queries
        filtered: List[Tuple[float, Chunk]] = []
        for sc, ch in rescored:
            if sc < self.min_score:
                continue
            if "xai" in question.lower() or "explain" in question.lower():
                if not _is_xai_relevant(ch.text):
                    continue
            filtered.append((sc, ch))

        need_live = False
        lower_q = question.lower()
        if not filtered:
            need_live = True
        elif any(term in lower_q for term in NEWS_TRIGGERS):
            need_live = True
        else:
            top_chunk = filtered[0][1] if filtered else rescored[0][1]
            try:
                dt = datetime.fromisoformat(top_chunk.fetched_at.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_days = (now - dt).total_seconds() / 86400.0
                if age_days > 45:
                    need_live = True
            except Exception:
                need_live = True

        live_results: List[Tuple[float, Chunk]] = []
        if need_live and self.live_fetcher:
            try:
                live_results = self.live_fetcher.search(question, query_vec=query_vec)
            except Exception as exc:
                logger.debug("Live news search failed: %s", exc)

        combined: List[Tuple[float, Chunk]] = []
        for score, chunk in filtered or rescored:
            combined.append((score, chunk))
        for score, chunk in live_results:
            freshness = recency_weight(chunk.fetched_at)
            combined.append((float(score + 0.15 * freshness), chunk))

        deduped: Dict[str, Tuple[float, Chunk]] = {}
        for score, chunk in combined:
            if chunk.url in deduped:
                if score > deduped[chunk.url][0]:
                    deduped[chunk.url] = (score, chunk)
            else:
                deduped[chunk.url] = (score, chunk)

        merged = list(deduped.values())
        merged.sort(key=lambda x: x[0], reverse=True)
        merged = merged[: self.top_k]

        if self.validate_urls:
            validated: List[Tuple[float, Chunk]] = []
            for sc, ch in merged:
                if self._is_url_reachable(ch.url):
                    validated.append((sc, ch))
            if validated:
                merged = validated

        self.last_results = merged
        return self.last_results

    def _is_url_reachable(self, url: str, timeout: float = 4.0) -> bool:
        if not url:
            return False
        try:
            resp = requests.head(url, allow_redirects=True, timeout=timeout)
            if resp.status_code >= 400:
                return False
            return True
        except Exception:
            return False

    def _format_sources(self) -> str:
        if not self.last_results:
            return ""
        lines = ["References:"]
        for idx, (_, chunk) in enumerate(self.last_results, start=1):
            title = chunk.title.strip() if chunk.title else "Untitled"
            fetched = chunk.fetched_at or "unknown date"
            lines.append(f"[{idx}] {title} - {chunk.url} (fetched {fetched})")
        return "\n".join(lines)

    def answer(self, question: str) -> str:
        results = self.retrieve(question)
        if not results or results[0][0] < self.min_score:
            msg = (
                "I couldn't find strong, relevant matches for your query in the indexed pages. "
                "Crawl fresher XAI sources (e.g., arXiv cs.IR/cs.AI \"new\" lists, HF blog) and rebuild the index."
            )
            self.short_memory.add("user", question)
            self.short_memory.add("assistant", msg)
            self.memory_store.add("conversation", f"Q: {question}\nA: {msg}", {"sources": []})
            return msg

        messages = self._build_messages(question)
        answer = self.llm_client.chat(messages, temperature=0.2)
        sources_block = self._format_sources()
        full_answer = f"Answer:\n{answer}\n\n{sources_block}" if sources_block else f"Answer:\n{answer}"

        display_answer = full_answer
        if self.display_char_limit and len(display_answer) > self.display_char_limit:
            display_answer = display_answer[: self.display_char_limit].rstrip() + " ..."

        self.short_memory.add("user", question)
        self.short_memory.add("assistant", display_answer)
        self.memory_store.add(
            "conversation",
            f"Q: {question}\nA: {full_answer}",
            {
                "sources": [chunk.url for _, chunk in self.last_results],
                "display_preview": display_answer,
            },
        )
        return display_answer


def interactive_chat(
    index_path: Path,
    model_name: Optional[str] = None,
    top_k: int = 4,
    llm: str = "ollama",
    ollama_model: str = "mixtral:8x7b",
    openai_model: str = "gpt-4o-mini",
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    index = load_index(index_path)
    llm_client = LLMClient(backend=llm, model=ollama_model, openai_model=openai_model)
    memory_store = MemoryStore()
    agent = ConversationalRAGAgent(
        index=index,
        model_name=model_name,
        llm_client=llm_client,
        top_k=top_k,
        memory_store=memory_store,
    )

    print("Conversational RAG ready. Type 'exit' to quit.")
    while True:
        question = input("\nQ: ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            break
        try:
            answer = agent.answer(question)
            print(f"\nA: {answer}")
        except Exception as exc:
            logger.info("LLM unavailable (%s). Showing top contexts instead.", exc)
            if not agent.last_results:
                agent.retrieve(question)
            for idx, (score, chunk) in enumerate(agent.last_results, start=1):
                print(f"[{idx}] score={score:.3f} {chunk.title} - {chunk.url}")
                print(chunk.text[:700] + ("..." if len(chunk.text) > 700 else ""))


if __name__ == "__main__":
    interactive_chat(Path("data/index.pkl.gz"))
