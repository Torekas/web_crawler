"""Crawler orchestration: fetch, clean, chunk, dedup, and index content."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from bs4 import BeautifulSoup
import httpx

from .config import Config
from .dedup import Deduper, content_hash
from .http_fetcher import PoliteHttpFetcher
from .browser_fetcher import BrowserFetcher
from .markdown_cleaner import html_to_markdown
from .chunker import chunk_text
from .storage import Storage, ensure_dirs
from .url_utils import canonicalize_url, domain_from_url, should_crawl
from .extraction import extract_with_selectors, extract_with_llm_stub

logger = logging.getLogger(__name__)


def _extract_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        if href.startswith("http"):
            links.append(href)
        elif href.startswith("/"):
            links.append(base_url.rstrip("/") + href)
    return links


def relevance_score(text: str, keywords: List[str]) -> float:
    lowered = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lowered)
    return hits / max(len(keywords), 1)


class CrawlService:
    """High-level crawler that coordinates fetching, cleaning, dedup, and indexing."""

    def __init__(self, config: Config, storage: Storage, indexer, use_browser: bool | None = None) -> None:
        self.config = config
        self.storage = storage
        self.indexer = indexer
        self.deduper = Deduper()
        self.use_browser = config.crawler.use_browser if use_browser is None else use_browser
        self.http = PoliteHttpFetcher(
            user_agent=config.crawler.user_agent,
            cache_dir=config.crawler.cache_dir,
            obey_robots=config.crawler.obey_robots,
            per_domain_delay=config.rate_limits.per_domain_delay,
            max_retries=config.rate_limits.max_retries,
            backoff_base=config.rate_limits.backoff_base,
            circuit_breaker_failures=config.rate_limits.circuit_breaker_failures,
            circuit_breaker_reset=config.rate_limits.circuit_breaker_reset,
        )
        self.browser = BrowserFetcher(
            headless=config.browser.headless,
            proxy=config.browser.proxy,
            cookies_path=config.browser.cookies_path,
            screenshot_dir=config.browser.screenshot_dir,
            wait_seconds=config.crawler.dynamic_wait_seconds,
            scroll_pause=config.browser.scroll_pause,
        )
        ensure_dirs(config.storage.raw_html_dir, config.storage.markdown_dir, config.storage.diagnostics_dir)
        self.domain_counts: Dict[str, int] = defaultdict(int)
        self.seen_keywords: Set[str] = set()
        self.no_gain_streak = 0

    async def close(self) -> None:
        await self.http.close()
        await self.browser.close()

    async def crawl(self, seeds: List[str], job_id: Optional[str] = None) -> Dict[str, int]:
        job_id = job_id or str(uuid.uuid4())
        self.storage.upsert_job(job_id, status="running", params={"seeds": seeds})
        queue: deque[Tuple[str, int]] = deque()
        visited: Set[str] = set()
        for seed in seeds:
            queue.append((seed, 0))
        processed = 0

        try:
            while queue:
                if processed >= self.config.crawler.max_pages:
                    break
                if self.config.crawler.strategy == "dfs":
                    url, depth = queue.pop()
                else:
                    url, depth = queue.popleft()
                canonical = canonicalize_url(url)
                if canonical in visited:
                    continue
                if depth > self.config.crawler.max_depth:
                    continue
                if not should_crawl(
                    canonical,
                    allowed_domains=self.config.crawler.allowed_domains or None,
                    denied_domains=self.config.crawler.denied_domains or None,
                    allow_patterns=self.config.crawler.allow_url_patterns or None,
                    deny_patterns=self.config.crawler.deny_url_patterns or None,
                ):
                    continue
                domain = domain_from_url(canonical)
                if self.domain_counts[domain] >= self.config.crawler.max_urls_per_domain:
                    continue
                visited.add(canonical)
                result = await self.http.fetch(canonical, cache_ttl=self.config.crawler.content_cache_ttl)
                html_content = result.content
                status_code = result.status_code
                screenshot_path = None
                if (status_code is None or status_code >= 400) and self.use_browser:
                    browser_result = await self.browser.fetch(canonical, infinite_scroll=self.config.crawler.infinite_scroll, scroll_timeout=self.config.crawler.scroll_timeout)
                    html_content = browser_result.content
                    status_code = browser_result.status
                    screenshot_path = browser_result.screenshot_path
                if not html_content:
                    continue
                md, text = html_to_markdown(
                    html_content,
                    keywords=self.config.crawler.topic_keywords,
                    use_bm25=self.config.index.use_bm25_filter,
                    bm25_top_k=self.config.index.bm25_top_k,
                )
                if self.deduper.seen(text):
                    is_dup = True
                else:
                    is_dup = False
                chash = content_hash(text)
                topic_score = relevance_score(text, self.config.crawler.topic_keywords)
                page = self.storage.save_page(
                    canonical_url=canonical,
                    url=canonical,
                    domain=domain,
                    html=html_content,
                    markdown=md,
                    text=text,
                    metadata={"title": None},
                    status_code=status_code,
                    content_hash=chash,
                    topic_score=topic_score,
                    is_duplicate=is_dup,
                    diagnostics={
                        "timing_ms": result.elapsed,
                        "redirect_chain": result.redirect_chain,
                        "screenshot_path": screenshot_path,
                    },
                )
                page_existing = getattr(page, "_was_existing", False)
                if (self.config.extraction.css_selectors or self.config.extraction.xpath_selectors) and not page_existing:
                    data = extract_with_selectors(html_content, self.config.extraction.css_selectors, self.config.extraction.xpath_selectors)
                    self.storage.save_extraction(page.id, schema={"selectors": True}, data=data)
                if self.config.extraction.enable_llm_extraction and not page_existing:
                    llm_data = extract_with_llm_stub(html_content, schema=self.config.extraction.llm_schema)
                    self.storage.save_extraction(page.id, schema=self.config.extraction.llm_schema, data=llm_data)
                skip_indexing = is_dup or page_existing
                if not skip_indexing and md:
                    chunks = chunk_text(md, chunk_size=self.config.index.chunk_size, overlap=self.config.index.chunk_overlap, min_size=self.config.index.min_chunk_chars)
                    if chunks:
                        for chunk in chunks:
                            chunk.setdefault("metadata", {}).update({"page_id": page.id, "url": canonical})
                        self.storage.save_chunks(page.id, chunks)
                        self.indexer.add_chunks(chunks)
                self.domain_counts[domain] += 1
                processed += 1
                page_keywords = {kw for kw in self.config.crawler.topic_keywords if kw.lower() in text.lower()}
                before = len(self.seen_keywords)
                self.seen_keywords.update(page_keywords)
                if len(self.seen_keywords) == before:
                    self.no_gain_streak += 1
                else:
                    self.no_gain_streak = 0
                if self._should_stop():
                    break
                for link in self._filter_links(_extract_links(html_content, canonical)):
                    if self.config.crawler.strategy == "best-first" and topic_score >= 0.2:
                        queue.appendleft((link, depth + 1))
                    elif self.config.crawler.strategy == "dfs":
                        queue.append((link, depth + 1))
                    else:
                        queue.append((link, depth + 1))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Crawl failed: %s", exc)
            self.storage.upsert_job(job_id, status="failed", error=str(exc))
            await self._notify_webhook(job_id, {"status": "failed", "error": str(exc)})
            raise
        else:
            self.storage.upsert_job(job_id, status="completed", params={"processed": processed})
            await self._notify_webhook(job_id, {"status": "completed", "processed": processed})
        finally:
            await self.close()
        return {"processed": processed, "unique_keywords": len(self.seen_keywords)}

    def _filter_links(self, links: Iterable[str]) -> List[str]:
        filtered: List[str] = []
        for link in links:
            if should_crawl(
                link,
                allowed_domains=self.config.crawler.allowed_domains or None,
                denied_domains=self.config.crawler.denied_domains or None,
                allow_patterns=self.config.crawler.allow_url_patterns or None,
                deny_patterns=self.config.crawler.deny_url_patterns or None,
            ):
                filtered.append(link)
        return filtered

    def _should_stop(self) -> bool:
        if self.no_gain_streak >= 5:
            return True
        return False

    async def _notify_webhook(self, job_id: str, payload: Dict[str, object]) -> None:
        if not self.config.webhook.enabled or not self.config.webhook.url:
            return
        attempt = 0
        url = self.config.webhook.url
        while attempt < self.config.webhook.max_attempts:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(url, json={"job_id": job_id, **payload})
                return
            except Exception:  # noqa: BLE001
                attempt += 1
                await asyncio.sleep(self.config.webhook.backoff_seconds * (2**attempt))


__all__ = ["CrawlService", "relevance_score"]
