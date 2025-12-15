import asyncio
import json
import logging
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple
from urllib import robotparser
from urllib.parse import parse_qs, urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
import requests

from . import prompts
from .llm import LLMClient, safe_llm_call
from .memory import MemoryStore

logger = logging.getLogger(__name__)

# --- SEEDS: only reachable, crawlable entry pages to bootstrap depth ---
DEFAULT_SEEDS = [
    "https://huggingface.co/blog",
    "https://cohere.com/blog",
    "https://openai.com/blog",
    "https://www.anthropic.com/news",
    "https://x.ai/blog",
    "https://mistral.ai/news",
    "https://ai.meta.com/blog/",
    "https://stability.ai/news",
    "https://blog.langchain.dev/",
    "https://www.llamaindex.ai/blog",
    "https://www.anyscale.com/blog",
    "https://www.databricks.com/blog/category/ai",
    "https://ai.googleblog.com/",
    "https://research.google/blog/",
    "https://deepmind.google/discover/blog",
    "https://blog.google/technology/ai/",
    "https://news.microsoft.com/ai/",
    "https://aws.amazon.com/blogs/machine-learning/",
    "https://developer.nvidia.com/blog/tag/ai/",
]

DEFAULT_NEWS_QUERIES = [
    "latest AI news December 2025",
    "OpenAI December 2025 announcements",
    "Anthropic December 2025 news",
    "xAI Grok December 2025",
    "Google DeepMind December 2025",
    "Google AI December 2025",
    "Meta AI December 2025",
    "Mistral AI December 2025 release",
    "Hugging Face December 2025 model",
    "Llama 3 December 2025",
    "Claude December 2025 update",
]

# --- KEYWORDS: Rozszerzono o terminy agentowe i reasoning ---
AI_KEYWORDS = [
    "artificial intelligence",
    " ai ",
    "machine learning",
    " ml ",
    "deep learning",
    "llm",
    "language model",
    "foundation model",
    "genai",
    # Nowe:
    "agentic workflows",
    "chain of thought",
    "reasoning models",
    "model alignment",
    "retrieval augmented generation",
    "vector database",
    "embedding model",
    "reranking",
    "multi-agent",
    "policy gradient",
    "agent alignment",
    "tool use",
    "function calling",
    "retriever selection",
    "query rewriting",
    "active learning",
    "distillation",
    "model compression",
]

# --- DOMAIN TERMS: Głębokie techniki z solutions_LLM.xlsx ---
DOMAIN_TERMS = [
    # Frameworki i Metody
    "dspy",
    "reflexion", "self-reflection", "self-repair",
    "rag framework", "knowledge-oriented rag", 
    "tok-rag", "uncertaintyrag",
    "knowledge injection", "prompt distillation",
    "knowledge graph",
    "sft", "supervised fine-tuning",
    "dragan", "donadello",
    "retrieval-augmented generation",
    "vector store", "faiss", "weaviate", "milvus",
    "pgvector", "chroma", "elastic vector search",
    "hybrid search", "bm25",
    
    # Reward Modeling & Saliency
    "process reward model", "prm",
    "outcome reward model", "orm",
    "shapley values", "tokenshapley",
    "saliency map", "salsa attribution",
    "mirage attribution",
    
    # Prompting & Logic
    "miprov2", "bootstrapfewshot",
    "cot", "chain-of-thought",
    "program-guided reasoning", "programfc",
    "deductive verification",
    "rationalization trap",
    "verifier-solver loop",
    "retriever-augmenter",
    "double-checking",
    "consistency checking",
    "self-healing agent",
    "experience replay",
    "curriculum learning",
]

# --- EVAL TERMS: Konkretne metryki i narzędzia z notatek ---
EVAL_TERMS = [
    "llm-as-a-judge", "judge model",
    "g-eval",
    "opik",
    "prometheus",
    "mt-bench", "chatbot arena",
    "ragtruth",
    "selfcheck", "selfcheckgpt",
    "nli", "natural language inference",
    "faithfulness", "hallucination detection",
    "ragbench", "trace metrics",
    "context adherence", "context relevance",
    "normalized_diff",
    "pairwise preference", "win-rate",
    "human eval", "auto-eval",
    "judge consistency",
    "disagreement rate",
    "factual precision",
    "coverage",
]

# --- DATASETS: Pełna lista z plików notatki_testowe.txt i solutions_LLM.xlsx ---
DATASETS = [
    # Reasoning & Math
    "gsm8k", "math", "svamp", "aqua", "mawps", "addsub",
    
    # General & Science
    "arc", "mmlu", "mmlu-pro", "hellaswag", "sciq",
    
    # QA & RAG & Multi-hop
    "truthfulqa", "hotpotqa", "squad", "2wikimultihopqa",
    "natural questions", "nq", "triviaqa", 
    "musique", "qasper", "narrativeqa", "qmsum",
    
    # Agent & Code
    "alfworld", "humaneval", "mbpp", "leetcode",
    
    # Fact Checking & Hallucination
    "fever", "hover", "bingcheck", "ragtruth",
    "bioasq",

    # Benchmarks & Safety & Robustness
    "bbh", "mmlu-pro", "truthfulqa-longform", "toxigen", "civilcomments",
    "advbench", "real-toxicity-prompts",
]

# --- PLIKI DO POMINIĘCIA: Dodano formaty binarne i webowe śmieci ---
SKIP_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".zip", ".gz", ".mp4", ".mp3", 
    ".wav", ".avi", ".mov", ".exe", ".dmg", ".iso", ".bin", 
    ".css", ".js", ".ico", ".woff", ".woff2", ".ttf",
    ".tar", ".7z", ".rar", ".apk", ".bz2",
    ".ppt", ".pptx", ".doc", ".docx", ".xls", ".xlsx",
    ".epub",
}

# --- USER AGENTS: Rotacja, aby uniknąć blokowania (429 Too Many Requests) ---
DEFAULT_USER_AGENT = "AI-KnowledgeCrawler/0.3-GPU (+https://github.com/Torekas/web_crawler; bot@example.com)"
MAX_HTTP_RETRIES = 3
USER_AGENTS = [
    DEFAULT_USER_AGENT,
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36",
]

COMMON_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Pragma": "no-cache",
    "Upgrade-Insecure-Requests": "1",
}


def build_headers(user_agent: str) -> dict:
    headers = COMMON_HEADERS.copy()
    headers["User-Agent"] = user_agent
    return headers


@dataclass
class Page:
    url: str
    title: str
    text: str
    fetched_at: str
    relevance_score: float = 0.0
    decision: str = "keep"
    verifier_note: Optional[str] = None


class RobotsCache:
    """Cache robots.txt lookups so we do not hammer the same site."""

    def __init__(self) -> None:
        self.parsers = {}

    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self.parsers:
            rp = robotparser.RobotFileParser()
            rp.set_url(urljoin(base, "/robots.txt"))
            try:
                rp.read()
            except Exception:
                self.parsers[base] = None
            else:
                self.parsers[base] = rp
        parser = self.parsers.get(base)
        if parser is None:
            return True
        try:
            return parser.can_fetch(DEFAULT_USER_AGENT, url)
        except Exception:
            return True


def clean_text(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines), title


def extract_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href:
            continue
        absolute = urljoin(base_url, href)
        links.append(absolute.split("#")[0])
    return links


def _decode_duckduckgo_link(href: str) -> str:
    if "uddg=" in href:
        qs = urlparse(href).query
        decoded = parse_qs(qs).get("uddg", [])
        if decoded:
            return decoded[0]
    return href


def discover_news_seeds(
    news_queries: Iterable[str],
    per_query: int = 6,
    timeout: int = 10,
    user_agents: Optional[List[str]] = None,
) -> List[str]:
    headers_list = user_agents or USER_AGENTS
    session = requests.Session()
    discovered: List[str] = []
    for query in news_queries:
        headers = build_headers(random.choice(headers_list))
        try:
            resp = session.get(
                "https://duckduckgo.com/html",
                params={"q": query, "ia": "web"},
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.debug("Discovery failed for '%s': %s", query, exc)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        for anchor in soup.select("a.result__a"):
            href = anchor.get("href", "")
            if not href:
                continue
            decoded = _decode_duckduckgo_link(href)
            if not decoded.startswith(("http://", "https://")):
                continue
            if any(decoded.lower().endswith(ext) for ext in SKIP_EXTENSIONS):
                continue
            if decoded in discovered:
                continue
            discovered.append(decoded)
            if len(discovered) >= per_query * len(news_queries):
                break
        if len(discovered) >= per_query * len(news_queries):
            break
    return discovered


def should_visit(url: str, allowed_domains: Set[str]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if any(parsed.path.lower().endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    host = parsed.netloc.lower()
    return any(host.endswith(domain) for domain in allowed_domains)


def load_existing(output_path: Path) -> Set[str]:
    if not output_path.exists():
        return set()
    seen = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                payload = json.loads(line)
                seen.add(payload.get("url", ""))
            except json.JSONDecodeError:
                continue
    return seen


def save_page(page: Page, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(page.__dict__, ensure_ascii=False) + "\n")


def heuristic_relevance_score(text: str) -> int:
    lowered = text.lower()
    hits = 0
    for keyword in AI_KEYWORDS + DOMAIN_TERMS + EVAL_TERMS + DATASETS:
        if keyword in lowered:
            hits += 1
    return hits


class ReflexionLogger:
    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    def record(self, url: str, reason: str, action: str) -> None:
        note = f"{reason} -> {action}"
        logger.debug("Reflexion on %s: %s", url, note)
        self.memory.add("reflection", note, {"url": url})


class DomainJudge:
    def __init__(self, llm_client: Optional[LLMClient]) -> None:
        self.llm_client = llm_client

    async def judge(self, snippet: str, heuristic_score: int) -> Tuple[bool, str]:
        """Return (keep, note)."""
        if heuristic_score >= 2:
            return True, "heuristic-high"
        if self.llm_client is None:
            return heuristic_score > 0, "heuristic-low"

        def _call() -> str:
            return self.llm_client.chat([{"role": "user", "content": prompts.build_judge_prompt(snippet[:1200])}], 0.0)

        response = await asyncio.to_thread(safe_llm_call, "judge", _call, "")
        normalized = response.strip().lower()
        if "keep" in normalized:
            return True, f"llm-keep:{normalized[:60]}"
        if "unsure" in normalized or "unclear" in normalized:
            return False, f"llm-unsure:{normalized[:60]}"
        if not response:
            return heuristic_score > 0, "llm-unavailable"
        return False, f"llm-skip:{normalized[:60]}"

    async def verify(self, snippet: str) -> Tuple[bool, Optional[str]]:
        if self.llm_client is None:
            return True, None
        prompt = prompts.build_verification_prompt(snippet[:1200])

        def _call() -> str:
            return self.llm_client.chat([{"role": "user", "content": prompt}], temperature=0.0)

        raw = await asyncio.to_thread(safe_llm_call, "verify", _call, "")
        lowered = raw.lower()
        if "\"status\"" in lowered and "unclear" in lowered:
            return False, raw
        if not raw:
            return True, None
        return True, raw


class AsyncCrawler:
    def __init__(
        self,
        seeds: Iterable[str],
        max_pages: int,
        max_depth: int,
        delay_seconds: float,
        output_path: Path,
        timeout: int,
        concurrency: int,
        llm_client: Optional[LLMClient],
    ) -> None:
        self.seeds = seeds
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay_seconds = delay_seconds
        self.output_path = output_path
        self.timeout = timeout
        self.concurrency = concurrency
        self.llm_client = llm_client

        self.allowed_domains = {urlparse(seed).netloc.lower() for seed in seeds}
        # Track URLs already persisted on disk (for deduped writes) separately
        # from URLs seen during this crawl run.
        self.persisted: Set[str] = load_existing(output_path)
        self.seen: Set[str] = set()
        self.robots = RobotsCache()
        self.memory = MemoryStore()
        self.reflexion = ReflexionLogger(self.memory)
        self.judge = DomainJudge(llm_client)
        self.write_lock = asyncio.Lock()
        self.keep_count = 0
        self.skip_count = 0

    async def crawl(self) -> List[Page]:
        queue: deque[Tuple[str, int]] = deque((seed, 0) for seed in self.seeds)
        pages: List[Page] = []

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        headers = build_headers(DEFAULT_USER_AGENT)
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            while queue and len(pages) < self.max_pages:
                batch: List[Tuple[str, int]] = []
                while queue and len(batch) < self.concurrency and len(pages) + len(batch) < self.max_pages:
                    batch.append(queue.popleft())
                results = await asyncio.gather(*(self._process_url(url, depth, session) for url, depth in batch))
                for result in results:
                    if not result:
                        continue
                    page, links, next_depth = result
                    if page.decision == "keep":
                        pages.append(page)
                    for link in links:
                        queue.append((link, next_depth))

                if self.delay_seconds:
                    await asyncio.sleep(self.delay_seconds)

        logger.info("Crawl finished: %d pages kept", len(pages))
        logger.info(
            "\nCrawl summary\n-------------\nVisited: %d (depth <= %d)\nKept: %d\nSkipped (judged): %d\nOutput: %s",
            len(self.seen),
            self.max_depth,
            self.keep_count,
            self.skip_count,
            self.output_path,
        )
        return pages

    async def _process_url(
        self,
        url: str,
        depth: int,
        session: aiohttp.ClientSession,
    ) -> Optional[Tuple[Page, List[str], int]]:
        if url in self.seen or depth > self.max_depth:
            return None
        self.seen.add(url)
        if not should_visit(url, self.allowed_domains):
            return None
        allowed = await asyncio.to_thread(self.robots.allowed, url)
        if not allowed:
            self.reflexion.record(url, "robots.txt", "skipped")
            return None

        html = await self._fetch_html(url, session)
        if not html:
            return None

        text, title = clean_text(html)
        word_count = len(text.split())
        if word_count < 60 and depth > 0:
            self.reflexion.record(url, "short-text", "skipped")
            return None

        score = heuristic_relevance_score(text)
        keep, note = await self.judge.judge(text, heuristic_score=score)
        verifier_note = None
        if not keep and score > 0:
            keep, verifier_note = await self.judge.verify(text)

        next_depth = depth + 1
        links: List[str] = []
        if depth < self.max_depth:
            soup = BeautifulSoup(html, "html.parser")
            for link in extract_links(soup, url):
                if link not in self.seen and should_visit(link, self.allowed_domains):
                    links.append(link)

        decision = "keep" if keep else "skip"
        page = Page(
            url=url,
            title=title,
            text=text,
            fetched_at=datetime.utcnow().isoformat(),
            relevance_score=float(score),
            decision=decision,
            verifier_note=verifier_note,
        )

        if decision == "keep":
            if page.url in self.persisted:
                logger.info("Captured (existing, not re-saved): %s", url)
            else:
                await self._persist_page(page)
                self.persisted.add(page.url)
                logger.info("Captured (%s): %s", decision, url)
            self.keep_count += 1
        else:
            self.reflexion.record(url, "filtered", note)
            self.skip_count += 1

        return page, links, next_depth

    async def _fetch_html(self, url: str, session: aiohttp.ClientSession, attempt: int = 1) -> Optional[str]:
        user_agent = USER_AGENTS[min(attempt - 1, len(USER_AGENTS) - 1)]
        headers = build_headers(user_agent)
        try:
            async with session.get(url, headers=headers) as resp:
                if resp.status in {403, 429} and attempt < MAX_HTTP_RETRIES:
                    retry_after_header = resp.headers.get("Retry-After")
                    try:
                        retry_after = float(retry_after_header) if retry_after_header is not None else 0.0
                    except ValueError:
                        retry_after = 0.0
                    backoff_seconds = max(self.delay_seconds, 0.5 * attempt, retry_after)
                    self.reflexion.record(
                        url,
                        f"http {resp.status}",
                        f"backoff {backoff_seconds:.1f}s then retry UA {user_agent}",
                    )
                    if backoff_seconds > 0:
                        await asyncio.sleep(backoff_seconds)
                    return await self._fetch_html(url, session, attempt + 1)
                if resp.status >= 400:
                    self.reflexion.record(url, f"http {resp.status}", "skipped")
                    return None
                text = await resp.text()
                if text:
                    return text
        except aiohttp.ClientError as exc:
            self.reflexion.record(url, "network", f"{exc}")
        except Exception as exc:  # pragma: no cover - safety net
            self.reflexion.record(url, "network-unknown", f"{exc}")

        # Fallback to synchronous requests if aiohttp path failed or returned empty.
        try:
            import requests

            resp = requests.get(url, headers=headers, timeout=self.timeout)
            if resp.status_code >= 400:
                self.reflexion.record(url, f"http {resp.status_code}", "fallback-sync-skipped")
                return None
            return resp.text
        except Exception as exc:  # pragma: no cover - best-effort fallback
            self.reflexion.record(url, "network-fallback", f"{exc}")
            return None
        except aiohttp.ClientError as exc:
            self.reflexion.record(url, "network", f"{exc}")
            return None

    async def _persist_page(self, page: Page) -> None:
        async with self.write_lock:
            save_page(page, self.output_path)


async def crawl_async(
    seeds: Iterable[str] = DEFAULT_SEEDS,
    max_pages: int = 50,
    max_depth: int = 2,
    delay_seconds: float = 1.0,
    output_path: Path = Path("data/pages.jsonl"),
    timeout: int = 12,
    concurrency: int = 4,
    llm_backend: Optional[str] = None,
    llm_model: str = "mixtral:8x7b",
    openai_model: str = "gpt-4o-mini",
    discover_news: bool = True,
    news_queries: Optional[Iterable[str]] = None,
    news_per_query: int = 6,
) -> List[Page]:
    llm_client = None
    if llm_backend in {"ollama", "openai"}:
        llm_client = LLMClient(backend=llm_backend, model=llm_model, openai_model=openai_model)
    seeds_list = list(seeds)
    if discover_news:
        extra = discover_news_seeds(
            news_queries=news_queries or DEFAULT_NEWS_QUERIES,
            per_query=news_per_query,
            timeout=timeout,
            user_agents=USER_AGENTS,
        )
        for url in extra:
            if url not in seeds_list:
                seeds_list.append(url)
    crawler = AsyncCrawler(
        seeds=seeds_list,
        max_pages=max_pages,
        max_depth=max_depth,
        delay_seconds=delay_seconds,
        output_path=output_path,
        timeout=timeout,
        concurrency=concurrency,
        llm_client=llm_client,
    )
    return await crawler.crawl()


def crawl(
    seeds: Iterable[str] = DEFAULT_SEEDS,
    max_pages: int = 50,
    max_depth: int = 2,
    delay_seconds: float = 1.0,
    output_path: Path = Path("data/pages.jsonl"),
    timeout: int = 12,
    concurrency: int = 4,
    llm_backend: Optional[str] = None,
    llm_model: str = "mixtral:8x7b",
    openai_model: str = "gpt-4o-mini",
    discover_news: bool = True,
    news_queries: Optional[Iterable[str]] = None,
    news_per_query: int = 6,
) -> List[Page]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return asyncio.run(
        crawl_async(
            seeds=seeds,
            max_pages=max_pages,
            max_depth=max_depth,
            delay_seconds=delay_seconds,
            output_path=output_path,
            timeout=timeout,
            concurrency=concurrency,
            llm_backend=llm_backend,
            llm_model=llm_model,
            openai_model=openai_model,
            discover_news=discover_news,
            news_queries=news_queries,
            news_per_query=news_per_query,
        )
    )


def run_from_cli(
    seeds: Optional[List[str]] = None,
    max_pages: int = 50,
    max_depth: int = 2,
    delay_seconds: float = 1.0,
    output_path: str = "data/pages.jsonl",
    concurrency: int = 4,
    llm_backend: Optional[str] = None,
    llm_model: str = "mixtral:8x7b",
    openai_model: str = "gpt-4o-mini",
    discover_news: bool = True,
    news_queries: Optional[List[str]] = None,
    news_per_query: int = 6,
) -> None:
    crawl(
        seeds=seeds or DEFAULT_SEEDS,
        max_pages=max_pages,
        max_depth=max_depth,
        delay_seconds=delay_seconds,
        output_path=Path(output_path),
        concurrency=concurrency,
        llm_backend=llm_backend,
        llm_model=llm_model,
        openai_model=openai_model,
        discover_news=discover_news,
        news_queries=news_queries,
        news_per_query=news_per_query,
    )


if __name__ == "__main__":
    run_from_cli()
