import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from urllib import robotparser

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_SEEDS = [
    "https://openai.com/blog",
    "https://deepmind.google/",
    "https://ai.googleblog.com/",
    "https://www.anthropic.com/news",
    "https://stability.ai/blog",
    "https://www.microsoft.com/en-us/research/theme/artificial-intelligence/",
    "https://huggingface.co/blog",
    "https://cohere.com/blog",
]

AI_KEYWORDS = [
    "artificial intelligence",
    "ai ",
    "machine learning",
    "ml ",
    "deep learning",
    "llm",
    "language model",
    "foundation model",
    "genai",
]

SKIP_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".zip", ".gz", ".mp4", ".mp3"}

USER_AGENT = "AI-KnowledgeCrawler/0.1 (+https://github.com/Torekas/web_crawler)"


@dataclass
class Page:
    url: str
    title: str
    text: str
    fetched_at: str


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
            return parser.can_fetch(USER_AGENT, url)
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


def is_ai_relevant(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in AI_KEYWORDS)


def extract_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
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


def crawl(
    seeds: Iterable[str] = DEFAULT_SEEDS,
    max_pages: int = 50,
    max_depth: int = 2,
    delay_seconds: float = 1.0,
    output_path: Path = Path("data/pages.jsonl"),
    timeout: int = 12,
) -> List[Page]:
    allowed_domains = {urlparse(seed).netloc for seed in seeds}
    queue: deque[Tuple[str, int]] = deque((seed, 0) for seed in seeds)
    seen: Set[str] = load_existing(output_path)
    robots = RobotsCache()
    pages: List[Page] = []

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    logger.info("Starting crawl: %d seeds, max_pages=%d, depth=%d", len(queue), max_pages, max_depth)

    while queue and len(pages) < max_pages:
        url, depth = queue.popleft()
        if url in seen or depth > max_depth:
            continue
        seen.add(url)
        if not should_visit(url, allowed_domains):
            continue
        if not robots.allowed(url):
            logger.debug("Blocked by robots.txt: %s", url)
            continue

        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
        except Exception as exc:
            logger.debug("Failed to fetch %s: %s", url, exc)
            continue

        text, title = clean_text(resp.text)
        if len(text.split()) < 60:
            continue
        if not is_ai_relevant(text):
            continue

        page = Page(
            url=url,
            title=title,
            text=text,
            fetched_at=datetime.utcnow().isoformat(),
        )
        save_page(page, output_path)
        pages.append(page)
        logger.info("Captured (%d/%d): %s", len(pages), max_pages, url)

        if depth < max_depth:
            soup = BeautifulSoup(resp.text, "html.parser")
            for link in extract_links(soup, url):
                if link not in seen:
                    queue.append((link, depth + 1))

        if delay_seconds:
            time.sleep(delay_seconds)

    return pages


def run_from_cli(
    seeds: Optional[List[str]] = None,
    max_pages: int = 50,
    max_depth: int = 2,
    delay_seconds: float = 1.0,
    output_path: str = "data/pages.jsonl",
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    crawl(
        seeds=seeds or DEFAULT_SEEDS,
        max_pages=max_pages,
        max_depth=max_depth,
        delay_seconds=delay_seconds,
        output_path=Path(output_path),
    )


if __name__ == "__main__":
    run_from_cli()
