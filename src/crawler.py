import asyncio
import json
import logging
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple
from urllib import robotparser
from urllib.parse import parse_qs, urljoin, urlparse

import aiohttp
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from . import prompts
from .llm import LLMClient, safe_llm_call
from .memory import MemoryStore

logger = logging.getLogger(__name__)

# --- SEEDS: only reachable, crawlable entry pages to bootstrap depth ---
DEFAULT_SEEDS = [
    "https://huggingface.co/blog",
    "https://cohere.com/blog",
    "https://ai.meta.com/blog/",
    "https://stability.ai/news",
    "https://blog.langchain.dev/",
    "https://www.llamaindex.ai/blog",
    "https://www.anyscale.com/blog",
    "https://www.databricks.com/blog/category/ai",
    "https://ai.googleblog.com/",
    "https://research.google/blog/",
]

DEFAULT_DISCOVERY_QUERIES = [
    "retrieval augmented generation blog",
    "rag framework updates",
    "agentic workflows llm",
    "self reflection llm agents",
    "rag evaluation metrics blog",
    "vector database rag use cases",
    "prompt distillation knowledge injection",
    "explainable ai interpretability xai",
    "llm as a judge evaluation",
    "faithfulness hallucination detection llm",
    "compliant ai governance deterministic rules",
    "probabilistic llm deterministic rules hybrid",
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
    "explainable ai",
    "xai",
    "interpretability",
    "model governance",
    "compliance",
    "deterministic rules",
    "probabilistic models",
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
USER_AGENTS = [
    DEFAULT_USER_AGENT,
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36",
    "Googlebot/2.1 (+http://www.google.com/bot.html)",
    "DuckDuckBot/1.0; (+http://duckduckgo.com/duckduckbot.html)",
]


@dataclass
class Page:
    url: str
    title: str
    text: str
    fetched_at: str
    relevance_score: float = 0.0
    confidence: float = 0.0
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


def _decode_duckduckgo_link(href: str) -> str:
    if "uddg=" in href:
        qs = urlparse(href).query
        decoded = parse_qs(qs).get("uddg", [])
        if decoded:
            return decoded[0]
    return href


def _safe_get(url: str, timeout: int = 8) -> Optional[requests.Response]:
    try:
        resp = requests.get(url, headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=timeout)
        if resp.status_code < 400:
            return resp
    except Exception as exc:
        logger.debug("auto-seed fetch failed for %s: %s", url, exc)
    return None


def _discover_from_sitemap(base_url: str, limit: int = 30) -> List[Tuple[str, str]]:
    parsed = urlparse(base_url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    sitemap_urls = [f"{root}/sitemap.xml", f"{root}/sitemap_index.xml", f"{root}/sitemap-index.xml"]
    found: List[Tuple[str, str]] = []
    for sitemap_url in sitemap_urls:
        resp = _safe_get(sitemap_url)
        if not resp:
            continue
        try:
            soup = BeautifulSoup(resp.text, "xml")
        except Exception:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
                soup = BeautifulSoup(resp.text, "html.parser")
        for loc in soup.find_all("loc"):
            url = loc.get_text(strip=True)
            if url:
                found.append((url, "sitemap"))
            if len(found) >= limit:
                return found
    return found


def _discover_from_rss(base_url: str, limit: int = 30) -> List[Tuple[str, str]]:
    parsed = urlparse(base_url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    feeds = [f"{root}/feed", f"{root}/rss.xml", f"{root}/blog/feed", f"{root}/blog/rss"]
    found: List[Tuple[str, str]] = []
    for feed in feeds:
        resp = _safe_get(feed)
        if not resp:
            continue
        try:
            soup = BeautifulSoup(resp.text, "xml")
        except Exception:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
                soup = BeautifulSoup(resp.text, "html.parser")
        for item in soup.find_all("item"):
            link = item.find("link")
            title = item.find("title")
            url = link.get_text(strip=True) if link else ""
            if url:
                context = title.get_text(strip=True) if title else "rss"
                found.append((url, context))
            if len(found) >= limit:
                return found
    return found


def _discover_from_search(query: str, limit: int = 10) -> List[Tuple[str, str]]:
    try:
        resp = requests.get(
            "https://duckduckgo.com/html",
            params={"q": query, "ia": "web"},
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.debug("auto-seed search failed for %s: %s", query, exc)
        return []

    found: List[Tuple[str, str]] = []
    soup = BeautifulSoup(resp.text, "html.parser")
    for anchor in soup.select("a.result__a"):
        href = anchor.get("href", "").strip()
        if not href:
            continue
        url = _decode_duckduckgo_link(href)
        title = anchor.get_text(" ", strip=True)
        found.append((url, f"{title} {query}"))
        if len(found) >= limit:
            return found

    if not found:
        # fallback: look for DuckDuckGo redirect links
        for anchor in soup.select("a[href]"):
            href = anchor.get("href", "")
            if "uddg=" in href:
                found.append((_decode_duckduckgo_link(href), query))
            if len(found) >= limit:
                break
    return found


def discover_seeds(
    seeds: Iterable[str],
    max_new: int = 25,
    search_queries: Optional[List[str]] = None,
    per_source_limit: int = 30,
) -> List[str]:
    seeds_list = list(seeds)
    candidates: List[Tuple[str, str]] = []
    for seed in seeds_list:
        candidates.extend(_discover_from_sitemap(seed, limit=per_source_limit))
        candidates.extend(_discover_from_rss(seed, limit=per_source_limit))

    queries = search_queries or DEFAULT_DISCOVERY_QUERIES
    for query in queries:
        candidates.extend(_discover_from_search(query, limit=max(5, per_source_limit // 2)))

    seen_urls = set(seeds_list)
    scored: List[Tuple[int, str]] = []
    for url, context in candidates:
        if not url or url in seen_urls:
            continue
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if any(parsed.path.lower().endswith(ext) for ext in SKIP_EXTENSIONS):
            continue
        score = heuristic_relevance_score(f"{url} {context}")
        if score <= 0:
            continue
        scored.append((score, url))
        seen_urls.add(url)

    scored.sort(key=lambda tup: tup[0], reverse=True)
    selected = [url for _, url in scored[:max_new]]
    if selected:
        logger.info("Auto-discovered %d seeds (kept top %d)", len(selected), len(selected))
    else:
        logger.info("Auto-discovery found no additional seeds")
    return selected


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
        memory_store: Optional[MemoryStore] = None,
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
        self.persisted_urls: Set[str] = load_existing(output_path)
        self.seen: Set[str] = set()
        self.robots = RobotsCache()
        self.memory = memory_store or MemoryStore()
        self.reflexion = ReflexionLogger(self.memory)
        self.judge = DomainJudge(llm_client)
        self.write_lock = asyncio.Lock()

    async def crawl(self) -> List[Page]:
        queue: deque[Tuple[str, int]] = deque((seed, 0) for seed in self.seeds)
        pages: List[Page] = []
        saved_count = 0

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        headers = {"User-Agent": DEFAULT_USER_AGENT}
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            while queue and saved_count < self.max_pages:
                batch: List[Tuple[str, int]] = []
                while queue and len(batch) < self.concurrency:
                    batch.append(queue.popleft())
                results = await asyncio.gather(*(self._process_url(url, depth, session) for url, depth in batch))
                for result in results:
                    if not result:
                        continue
                    page, links, next_depth, saved = result
                    if page.decision == "keep":
                        pages.append(page)
                        if saved:
                            saved_count += 1
                    for link in links:
                        queue.append((link, next_depth))

                if saved_count >= self.max_pages:
                    break

                if self.delay_seconds:
                    await asyncio.sleep(self.delay_seconds)

        logger.info("Crawl finished: %d pages kept (%d new)", len(pages), saved_count)
        return pages

    async def _process_url(
        self,
        url: str,
        depth: int,
        session: aiohttp.ClientSession,
    ) -> Optional[Tuple[Page, List[str], int, bool]]:
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
        if len(text.split()) < 60:
            return None

        score = heuristic_relevance_score(text)
        keep, note = await self.judge.judge(text, heuristic_score=score)
        verifier_note = None
        if not keep and score > 0:
            keep, verifier_note = await self.judge.verify(text)

        # Build a simple confidence estimate combining heuristics and judge signal.
        confidence = min(1.0, 0.15 * score)
        if note.startswith("heuristic-high"):
            confidence = min(1.0, confidence + 0.25)
        if note.startswith("llm-keep"):
            confidence = min(1.0, confidence + 0.25)
        if not keep:
            confidence = max(0.05, confidence * 0.4)

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
            confidence=float(confidence),
            decision=decision,
            verifier_note=verifier_note,
        )

        saved = False
        if decision == "keep":
            saved = await self._persist_page(page)
            status = "new" if saved else "existing"
            logger.info("Captured (%s): %s", status, url)
        else:
            self.reflexion.record(url, "filtered", note)

        return page, links, next_depth, decision == "keep" and saved

    async def _fetch_html(self, url: str, session: aiohttp.ClientSession, attempt: int = 1) -> Optional[str]:
        try:
            async with session.get(url) as resp:
                if resp.status in {403, 429} and attempt < len(USER_AGENTS):
                    new_agent = USER_AGENTS[attempt % len(USER_AGENTS)]
                    session.headers.update({"User-Agent": new_agent})
                    self.reflexion.record(url, f"http {resp.status}", f"retry with UA {new_agent}")
                    return await self._fetch_html(url, session, attempt + 1)
                if resp.status >= 400:
                    self.reflexion.record(url, f"http {resp.status}", "skipped")
                    return None
                return await resp.text()
        except aiohttp.ClientError as exc:
            self.reflexion.record(url, "network", f"{exc}")
            return None

    async def _persist_page(self, page: Page) -> bool:
        if page.url in self.persisted_urls:
            return False
        async with self.write_lock:
            if page.url in self.persisted_urls:
                return False
            save_page(page, self.output_path)
            self.persisted_urls.add(page.url)
            self.memory.add_page_entry(page)
        return True


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
    auto_seed: bool = False,
    auto_seed_max: int = 25,
    auto_seed_queries: Optional[List[str]] = None,
    auto_seed_per_source: int = 30,
) -> List[Page]:
    seeds_input = list(seeds)
    memory_store = MemoryStore()
    llm_client = None
    if llm_backend in {"ollama", "openai"}:
        llm_client = LLMClient(backend=llm_backend, model=llm_model, openai_model=openai_model)

    seed_list = list(dict.fromkeys(seeds_input + memory_store.seed_urls()))
    if seed_list != seeds_input:
        logger.info("Loaded %d seeds from long-term memory (total seeds: %d)", len(seed_list) - len(seeds_input), len(seed_list))
    if auto_seed:
        new_seeds = discover_seeds(seed_list, max_new=auto_seed_max, search_queries=auto_seed_queries, per_source_limit=auto_seed_per_source)
        if new_seeds:
            seed_list = list(dict.fromkeys(seed_list + new_seeds))
            logger.info("Expanded seeds from %d to %d", len(seeds), len(seed_list))
            for url in new_seeds:
                memory_store.add_seed(url, {"source": "auto-discovery"})
    crawler = AsyncCrawler(
        seeds=seed_list,
        max_pages=max_pages,
        max_depth=max_depth,
        delay_seconds=delay_seconds,
        output_path=output_path,
        timeout=timeout,
        concurrency=concurrency,
        llm_client=llm_client,
        memory_store=memory_store,
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
    auto_seed: bool = False,
    auto_seed_max: int = 25,
    auto_seed_queries: Optional[List[str]] = None,
    auto_seed_per_source: int = 30,
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
            auto_seed=auto_seed,
            auto_seed_max=auto_seed_max,
            auto_seed_queries=auto_seed_queries,
            auto_seed_per_source=auto_seed_per_source,
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
    auto_seed: bool = False,
    auto_seed_max: int = 25,
    auto_seed_queries: Optional[List[str]] = None,
    auto_seed_per_source: int = 30,
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
        auto_seed=auto_seed,
        auto_seed_max=auto_seed_max,
        auto_seed_queries=auto_seed_queries,
        auto_seed_per_source=auto_seed_per_source,
    )


if __name__ == "__main__":
    run_from_cli()
