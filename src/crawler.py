import asyncio
import json
import logging
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple
from urllib import robotparser
from urllib.parse import urljoin, urlparse

# Zmieniamy aiohttp na playwright
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup

from . import prompts
from .llm import LLMClient, safe_llm_call
from .memory import MemoryStore

logger = logging.getLogger(__name__)

# --- SEEDS ---
DEFAULT_SEEDS = [
    "https://huggingface.co/blog",
    "https://arxiv.org/list/cs.AI/new",
    "https://arxiv.org/list/cs.IR/new",
    "https://cohere.com/blog",
    "https://ai.meta.com/blog/",
    "https://openai.com/blog",
    "https://deepmind.google/discover/blog",
    "https://www.anthropic.com/news",
    "https://research.ibm.com/blog/ai",
    "https://www.microsoft.com/en-us/research/blog/",
    "https://www.nvidia.com/en-us/research/ai-playground/",
    "https://stability.ai/news",
    "https://blog.langchain.dev/",
    "https://www.llamaindex.ai/blog",
    "https://www.anyscale.com/blog",
    "https://www.databricks.com/blog/category/ai",
    "https://research.google/blog/",
    "https://openai.com/news/",
    "https://aws.amazon.com/blogs/machine-learning/",
]

# --- KEYWORDS ---
AI_KEYWORDS = [
    "artificial intelligence", " ai ", "machine learning", " ml ",
    "deep learning", "llm", "language model", "foundation model", "genai",
    "agentic workflows", "chain of thought", "reasoning models",
    "rag", "retrieval augmented", "vector database", "embedding",
    "multi-agent", "reinforcement learning", "fine-tuning", "lora",
    "quantization", "inference", "gpu", "transformer", "attention mechanism"
]

# --- DOMAIN TERMS ---
DOMAIN_TERMS = [
    "dspy", "reflexion", "self-reflection", "self-repair",
    "knowledge graph", "sft", "ppo", "dpo", "orpo",
    "vector store", "faiss", "weaviate", "milvus", "qdrant", "chroma",
    "hybrid search", "bm25", "reranking", "colbert",
    "cot", "chain-of-thought", "react agent", "autogen", "crewai",
    "langgraph", "semantic router", "guardrails"
]

# --- EVAL TERMS ---
EVAL_TERMS = [
    "llm-as-a-judge", "g-eval", "prometheus", "mt-bench", "chatbot arena",
    "hallucination", "faithfulness", "context relevance", "answer relevance",
    "ragas", "truelens", "arize", "confusion matrix", "f1 score", "exact match"
]

# --- DATASETS ---
DATASETS = [
    "gsm8k", "mmlu", "arc-c", "hellaswag", "truthfulqa", "hotpotqa",
    "squad", "humaneval", "mbpp", "swe-bench"
]

# --- PLIKI DO POMINIĘCIA ---
SKIP_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".zip", ".gz", ".mp4", ".mp3",
    ".wav", ".avi", ".mov", ".exe", ".dmg", ".iso", ".bin",
    ".css", ".js", ".ico", ".woff", ".woff2", ".ttf", ".xml", ".rss", ".json",
    ".tar", ".7z", ".rar", ".apk", ".bz2",
    ".ppt", ".pptx", ".doc", ".docx", ".xls", ".xlsx", ".epub",
}

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
USER_AGENTS = [
    DEFAULT_USER_AGENT,
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36",
]

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
    """Prosta obsługa robots.txt, aby nie spamować stron, które sobie tego nie życzą."""
    def __init__(self) -> None:
        self.parsers = {}

    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self.parsers:
            rp = robotparser.RobotFileParser()
            rp.set_url(urljoin(base, "/robots.txt"))
            try:
                # Używamy prostego requests tutaj, bo robots.txt jest zazwyczaj statyczny
                # W wersji w pełni asynchronicznej można by to zmienić, ale to rzadka operacja
                import requests
                resp = requests.get(urljoin(base, "/robots.txt"), timeout=5)
                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                else:
                    rp.allow_all = True
            except Exception:
                rp.allow_all = True # Fail open jeśli nie można pobrać
            self.parsers[base] = rp
        
        parser = self.parsers.get(base)
        try:
            return parser.can_fetch(DEFAULT_USER_AGENT, url)
        except Exception:
            return True


def clean_text(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    
    # Usuwamy śmieci
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "header", "footer", "nav", "aside", "form", "button"]):
        tag.decompose()
    
    # Próba wyciągnięcia głównej treści (heurystyka)
    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=lambda x: x and 'content' in x)
    
    target = main_content if main_content else soup
    
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text = target.get_text(separator="\n")
    
    # Czyszczenie białych znaków
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_text = "\n".join(lines)
    
    return cleaned_text, title


def extract_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:")):
            continue
        try:
            absolute = urljoin(base_url, href)
            # Usuwamy fragmenty (#)
            clean_url = absolute.split("#")[0]
            if clean_url != base_url: # Unikamy pętli na samej sobie
                links.append(clean_url)
        except Exception:
            continue
    return list(set(links)) # Unikalne


def should_visit(url: str, allowed_domains: Set[str]) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in {"http", "https"}:
        return False
    
    path_lower = parsed.path.lower()
    if any(path_lower.endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    
    # Sprawdzenie czy domena jest w dozwolonych
    host = parsed.netloc.lower()
    # Usuwamy 'www.' dla porównania
    host_clean = host.replace("www.", "")
    
    # Zezwalamy jeśli host kończy się na którąś z dozwolonych domen
    # np. blog.google.com kończy się na google.com
    is_allowed = any(host_clean.endswith(domain.replace("www.", "")) for domain in allowed_domains)
    return is_allowed


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
    # Prosta heurystyka, liczy wystąpienia słów kluczowych
    # Można zoptymalizować (np. Aho-Corasick), ale dla małej skali wystarczy pętla
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
        """Zwraca (keep, note)."""
        # Jeśli wynik heurystyczny jest bardzo wysoki, akceptujemy bez LLM (oszczędność)
        if heuristic_score >= 5:
            return True, "heuristic-high-confidence"
        
        # Jeśli wynik jest zerowy, odrzucamy od razu
        if heuristic_score == 0:
            return False, "heuristic-zero"

        if self.llm_client is None:
            # Tryb bez LLM: akceptuj wszystko co ma > 1 słowo kluczowe
            return heuristic_score >= 2, "heuristic-only"

        # LLM Judge
        def _call() -> str:
            return self.llm_client.chat([{"role": "user", "content": prompts.build_judge_prompt(snippet[:1500])}], 0.0)

        response = await asyncio.to_thread(safe_llm_call, "judge", _call, "")
        normalized = response.strip().upper()
        
        if "KEEP" in normalized:
            return True, f"llm-keep"
        if "UNSURE" in normalized:
            # Jeśli LLM nie jest pewny, ale heurystyka była ok, zostawiamy
            return heuristic_score >= 3, "llm-unsure-fallback"
        
        return False, f"llm-skip"

    async def verify(self, snippet: str) -> Tuple[bool, Optional[str]]:
        if self.llm_client is None:
            return True, None
        
        prompt = prompts.build_verification_prompt(snippet[:1500])
        def _call() -> str:
            return self.llm_client.chat([{"role": "user", "content": prompt}], temperature=0.0)

        raw = await asyncio.to_thread(safe_llm_call, "verify", _call, "")
        if "\"status\": \"unclear\"" in raw.lower():
             return False, raw
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
        render_wait_ms: int = 1200,
        goto_timeout_ms: int = 15000,
    ) -> None:
        self.seeds = seeds
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay_seconds = delay_seconds
        self.output_path = output_path
        self.timeout = timeout
        self.concurrency = concurrency
        self.llm_client = llm_client
        self.render_wait_ms = render_wait_ms
        self.goto_timeout_ms = goto_timeout_ms

        self.allowed_domains = {urlparse(seed).netloc.lower().replace("www.", "") for seed in seeds}
        self.seen: Set[str] = load_existing(output_path)
        self.robots = RobotsCache()
        self.memory = MemoryStore()
        self.reflexion = ReflexionLogger(self.memory)
        self.judge = DomainJudge(llm_client)
        self.write_lock = asyncio.Lock()
        self.contexts = []
        self.context_cycle = None

    async def _build_context_pool(self, browser):
        pool = []
        uas = USER_AGENTS[:]
        random.shuffle(uas)
        for idx in range(max(1, self.concurrency)):
            ua = uas[idx % len(uas)]
            ctx = await browser.new_context(
                user_agent=ua,
                viewport={"width": 1280, "height": 900},
            )
            await ctx.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")
            pool.append(ctx)
        return pool

    async def crawl(self) -> List[Page]:
        queue: deque[Tuple[str, int]] = deque((seed, 0) for seed in self.seeds)
        pages: List[Page] = []
        
        logger.info(f"Starting crawler with Playwright. Concurrency: {self.concurrency}")

        async with async_playwright() as p:
            # Uruchamiamy przeglądarkę (Chromium)
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )

            self.contexts = await self._build_context_pool(browser)
            self.context_cycle = cycle(self.contexts)
            sem = asyncio.Semaphore(self.concurrency)
            
            while queue and len(pages) < self.max_pages:
                # Pobieramy partię URLi do przetworzenia
                batch_tasks = []
                while queue and len(batch_tasks) < self.concurrency and (len(pages) + len(batch_tasks)) < self.max_pages:
                    url, depth = queue.popleft()
                    batch_tasks.append(self._process_url_safe(sem, url, depth))
                
                if not batch_tasks:
                    break

                results = await asyncio.gather(*batch_tasks)
                
                for result in results:
                    if not result:
                        continue
                    
                    page_obj, links, next_depth = result
                    
                    if page_obj.decision == "keep":
                        pages.append(page_obj)
                        logger.info(f"CAPTURED [{len(pages)}/{self.max_pages}]: {page_obj.title} ({page_obj.url})")
                    
                    if next_depth <= self.max_depth:
                        for link in links:
                            if link not in self.seen:
                                queue.append((link, next_depth))

            for ctx in self.contexts:
                await ctx.close()
            await browser.close()

        logger.info("Crawl finished: %d pages kept", len(pages))
        return pages

    async def _process_url_safe(self, sem, url, depth):
        """Wrapper z semaforem, aby nie zarżnąć przeglądarki."""
        async with sem:
            # Losowe opóźnienie, aby wyglądać bardziej "ludzko"
            if self.delay_seconds > 0:
                await asyncio.sleep(self.delay_seconds * random.uniform(0.5, 1.5))
            context = next(self.context_cycle)
            return await self._process_url(context, url, depth)

    async def _process_url(
        self,
        context,
        url: str,
        depth: int,
    ) -> Optional[Tuple[Page, List[str], int]]:
        if url in self.seen:
            return None
        self.seen.add(url)
        
        if not should_visit(url, self.allowed_domains):
            return None

        # Robots check (opcjonalnie, można wyłączyć jeśli crawler jest "agresywny")
        if not self.robots.allowed(url):
            self.reflexion.record(url, "robots.txt", "skipped")
            return None

        # Fetch HTML via Playwright
        html, final_url = await self._fetch_html_playwright(context, url)
        if not html:
            return None

        text, title = clean_text(html)
        
        word_count = len(text.split())

        # Ocena strony
        score = heuristic_relevance_score(text)

        # Bardzo krótkie strony tylko jeśli brak sygnałów relewantności
        if word_count < 20 and score == 0:
            self.reflexion.record(url, "content-too-short", f"len={word_count}")
            return None
        keep, note = await self.judge.judge(text, heuristic_score=score)
        
        verifier_note = None
        if not keep and score > 2:
            # Double check dla obiecujących stron odrzuconych przez LLM
            keep, verifier_note = await self.judge.verify(text)

        next_depth = depth + 1
        links: List[str] = []
        if depth < self.max_depth:
            extracted = extract_links(html, final_url)
            for link in extracted:
                if link not in self.seen and should_visit(link, self.allowed_domains):
                    links.append(link)

        decision = "keep" if keep else "skip"
        page = Page(
            url=final_url,
            title=title,
            text=text,
            fetched_at=datetime.utcnow().isoformat(),
            relevance_score=float(score),
            decision=decision,
            verifier_note=verifier_note,
        )

        if decision == "keep":
            await self._persist_page(page)
        else:
            self.reflexion.record(url, "filtered", note)

        return page, links, next_depth

    async def _fetch_html_playwright(self, context, url: str, attempt: int = 0) -> Tuple[Optional[str], str]:
        page = await context.new_page()
        try:
            # Timeout na nawigację
            try:
                response = await page.goto(url, wait_until="domcontentloaded", timeout=self.goto_timeout_ms)
            except PlaywrightTimeoutError as e:
                self.reflexion.record(url, "timeout-domcontent", str(e))
                # fallback na prostsze ładowanie
                try:
                    response = await page.goto(url, wait_until="load", timeout=self.goto_timeout_ms)
                except Exception as ee:
                    self.reflexion.record(url, "timeout-load", str(ee))
                    return None, url
            except Exception as e:
                self.reflexion.record(url, "playwright-error", str(e))
                return None, url

            if not response:
                return None, url
            
            if response.status >= 400:
                if response.status in {403, 429} and attempt + 1 < len(USER_AGENTS):
                    self.reflexion.record(url, f"http {response.status}", "retry-with-ua")
                    alt_ctx = next(self.context_cycle)
                    await page.close()
                    return await self._fetch_html_playwright(alt_ctx, url, attempt + 1)
                logger.warning(f"HTTP {response.status} for {url}")
                return None, url

            # Czekamy chwilę na JS (opcjonalne, ale pomaga na "lazy loading")
            await page.wait_for_timeout(self.render_wait_ms)
            # Krótki scroll dla lazy load
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(200)

            content = await page.content()
            final_url = page.url
            return content, final_url

        except Exception as exc:
            self.reflexion.record(url, "playwright-error", str(exc))
            return None, url
        finally:
            await page.close()

    async def _persist_page(self, page: Page) -> None:
        async with self.write_lock:
            save_page(page, self.output_path)


# Funkcje pomocnicze kompatybilne z main.py
async def crawl_async(
    seeds: Iterable[str] = DEFAULT_SEEDS,
    max_pages: int = 50,
    max_depth: int = 2,
    delay_seconds: float = 1.0,
    output_path: Path = Path("data/pages.jsonl"),
    timeout: int = 30, # Zwiększony timeout dla Playwright
    concurrency: int = 4, # Playwright zużywa więcej RAM, ostrożnie z concurrency
    llm_backend: Optional[str] = None,
    llm_model: str = "mixtral:8x7b",
    openai_model: str = "gpt-4o-mini",
    render_wait_ms: int = 1200,
    goto_timeout_ms: int = 15000,
) -> List[Page]:
    llm_client = None
    if llm_backend in {"ollama", "openai"}:
        llm_client = LLMClient(backend=llm_backend, model=llm_model, openai_model=openai_model)
    crawler = AsyncCrawler(
        seeds=seeds,
        max_pages=max_pages,
        max_depth=max_depth,
        delay_seconds=delay_seconds,
        output_path=output_path,
        timeout=timeout,
        concurrency=concurrency,
        llm_client=llm_client,
        render_wait_ms=render_wait_ms,
        goto_timeout_ms=goto_timeout_ms,
    )
    return await crawler.crawl()

def crawl(*args, **kwargs):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return asyncio.run(crawl_async(*args, **kwargs))
