"""Production-ready web crawler with smart discovery and relevance filtering.

Features:
- OOP design with a reusable WebCrawler class.
- Robust HTTP handling with requests.Session, user-agent rotation, and polite delays.
- Priority queue scheduling with keyword-aware prioritization.
- Topic relevance filtering to skip off-topic pages early.
- Search-based discovery (DuckDuckGo HTML endpoint) to keep the queue warm.
- Concurrent fetching via ThreadPoolExecutor.
- Expanded parsing (titles, meta descriptions, links, images, tables).
- CSV/JSON persistence helpers.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from queue import PriorityQueue
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36",
]

SKIP_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".zip",
    ".gz",
    ".mp4",
    ".mp3",
    ".css",
    ".js",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
}

PROMISING_HINTS = ("tutorial", "guide", "how-to", "docs", "blog", "article")


@dataclass
class CrawlResult:
    url: str
    title: str
    meta_description: str
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    tables: List[List[str]] = field(default_factory=list)
    text_snippet: str = ""
    fetched_at: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class WebCrawler:
    """Keyword-aware crawler with smart discovery and prioritized scheduling."""

    def __init__(
        self,
        start_urls: Iterable[str],
        keywords: Iterable[str],
        max_depth: int = 2,
        max_pages: int = 100,
        delay_range: Tuple[float, float] = (1.0, 3.0),
        timeout: int = 10,
        max_workers: int = 8,
        relevance_threshold: int = 3,
        discovery_batch_size: int = 10,
        user_agents: Optional[List[str]] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.keywords = [kw.lower() for kw in keywords]
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.delay_range = delay_range
        self.timeout = timeout
        self.max_workers = max_workers
        self.relevance_threshold = relevance_threshold
        self.discovery_batch_size = discovery_batch_size
        self.user_agents = user_agents or DEFAULT_USER_AGENTS

        self.session = session or self._build_session()
        self.queue: PriorityQueue[Tuple[int, int, str, int]] = PriorityQueue()
        self.visited: Set[str] = set()
        self.results: List[CrawlResult] = []
        self._counter = 0  # keeps ordering stable in PriorityQueue
        self._lock = threading.Lock()

        for url in start_urls:
            self._enqueue(url, depth=0)

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _enqueue(self, url: str, depth: int) -> None:
        if depth > self.max_depth or not self._should_visit(url):
            return
        normalized = self._normalize(url)
        with self._lock:
            if normalized in self.visited:
                return
            priority = self._priority_score(normalized, depth)
            self._counter += 1
            self.queue.put((priority, self._counter, normalized, depth))

    @staticmethod
    def _normalize(url: str) -> str:
        parsed = urlparse(url)
        clean = parsed._replace(fragment="", query=parsed.query).geturl()
        return clean

    def _priority_score(self, url: str, depth: int) -> int:
        score = 0
        lowered = url.lower()
        for hint in PROMISING_HINTS:
            if hint in lowered:
                score += 3
        for kw in self.keywords:
            if kw in lowered:
                score += 1
        score -= depth  # shallow links preferred
        return -score  # PriorityQueue returns smallest first

    def _should_visit(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False
        if any(parsed.path.lower().endswith(ext) for ext in SKIP_EXTENSIONS):
            return False
        return True

    def discover_new_seeds(self, topic_keyword: str, limit: int = 15) -> List[str]:
        """Search DuckDuckGo HTML endpoint to replenish seeds."""
        headers = {"User-Agent": random.choice(self.user_agents)}
        try:
            resp = self.session.get(
                "https://duckduckgo.com/html",
                params={"q": topic_keyword, "ia": "web"},
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Discovery failed for '%s': %s", topic_keyword, exc)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        found: List[str] = []
        for anchor in soup.select("a.result__a"):
            href = anchor.get("href", "")
            if not href:
                continue
            decoded = self._decode_duckduckgo_link(href)
            if decoded and self._should_visit(decoded):
                found.append(decoded)
            if len(found) >= limit:
                break
        return found

    @staticmethod
    def _decode_duckduckgo_link(href: str) -> str:
        if "uddg=" in href:
            qs = urlparse(href).query
            decoded = parse_qs(qs).get("uddg", [])
            if decoded:
                return decoded[0]
        return href

    def _random_user_agent(self) -> str:
        return random.choice(self.user_agents)

    def _sleep_briefly(self) -> None:
        low, high = self.delay_range
        if high <= 0:
            return
        time.sleep(random.uniform(low, high))

    def fetch(self, url: str) -> Optional[str]:
        self._sleep_briefly()
        headers = {"User-Agent": self._random_user_agent()}
        try:
            response = self.session.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            logger.info("Fetch failed for %s: %s", url, exc)
            return None

    def is_relevant(self, html: str) -> bool:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True).lower()
        hits = sum(text.count(kw) for kw in self.keywords)
        return hits >= self.relevance_threshold

    def parse_static_content(self, html: str, url: str) -> CrawlResult:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        meta = soup.find("meta", attrs={"name": "description"}) or soup.find(
            "meta", attrs={"property": "og:description"}
        )
        meta_desc = meta["content"].strip() if meta and meta.get("content") else ""

        links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]
        images = [urljoin(url, img["src"]) for img in soup.find_all("img", src=True)]

        tables: List[List[str]] = []
        for table in soup.find_all("table"):
            rows = []
            for row in table.find_all("tr"):
                cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                tables.append(rows)

        full_text = soup.get_text(" ", strip=True)
        snippet = (full_text[:500] + "...") if len(full_text) > 500 else full_text

        return CrawlResult(
            url=url,
            title=title,
            meta_description=meta_desc,
            links=links,
            images=images,
            tables=tables,
            text_snippet=snippet,
            fetched_at=datetime.utcnow().isoformat(),
        )

    def parse_dynamic_content(self, url: str) -> Optional[str]:
        """Placeholder for JS-heavy pages (e.g., Selenium/Playwright)."""
        # Implement headless browser fetch here when ready.
        return None

    def _process_url(self, url: str, depth: int, topic_keyword: Optional[str]) -> None:
        with self._lock:
            if url in self.visited:
                return
            self.visited.add(url)

        html = self.fetch(url)
        if not html:
            return
        if not self.is_relevant(html):
            logger.debug("Irrelevant page skipped: %s", url)
            return

        result = self.parse_static_content(html, url)
        with self._lock:
            if len(self.results) >= self.max_pages:
                return
            self.results.append(result)

        next_depth = depth + 1
        if next_depth > self.max_depth:
            return

        for link in result.links:
            self._enqueue(link, next_depth)

        # Keep discovery queue warm when it starts draining.
        if self.queue.qsize() < self.discovery_batch_size and topic_keyword:
            for seed in self.discover_new_seeds(topic_keyword, limit=self.discovery_batch_size):
                self._enqueue(seed, depth=0)

    def crawl(self, topic_keyword: Optional[str] = None) -> List[CrawlResult]:
        """Run the crawl loop until max_pages is reached or the queue is exhausted."""
        if self.queue.empty() and topic_keyword:
            for seed in self.discover_new_seeds(topic_keyword, limit=self.discovery_batch_size):
                self._enqueue(seed, depth=0)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while not self.queue.empty() and len(self.results) < self.max_pages:
                batch = []
                while not self.queue.empty() and len(batch) < self.max_workers:
                    priority, _, url, depth = self.queue.get()
                    batch.append((priority, url, depth))
                futures = [
                    executor.submit(self._process_url, url, depth, topic_keyword) for _, url, depth in batch
                ]
                for future in as_completed(futures):
                    future.result()
        return self.results

    def save_results(self, path: Path, fmt: str = "json") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt.lower() == "csv":
            fieldnames = [
                "url",
                "title",
                "meta_description",
                "links",
                "images",
                "tables",
                "text_snippet",
                "fetched_at",
            ]
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for item in self.results:
                    writer.writerow(
                        {
                            "url": item.url,
                            "title": item.title,
                            "meta_description": item.meta_description,
                            "links": ";".join(item.links),
                            "images": ";".join(item.images),
                            "tables": json.dumps(item.tables, ensure_ascii=False),
                            "text_snippet": item.text_snippet,
                            "fetched_at": item.fetched_at,
                        }
                    )
            return

        with path.open("w", encoding="utf-8") as f:
            for item in self.results:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


__all__ = ["WebCrawler", "CrawlResult"]
