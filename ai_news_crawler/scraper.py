from __future__ import annotations

import re
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


class AINewsScraper:
    """Scrape AI news articles from a list of source URLs."""

    def __init__(
        self,
        urls: Iterable[str],
        timeout: int = 15,
        max_articles_per_source: int = 12,
    ) -> None:
        self.urls = list(urls)
        self.timeout = timeout
        self.max_articles_per_source = max_articles_per_source
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/118.0 Safari/537.36"
                )
            }
        )
        self.min_article_length = 200
        self.min_feed_length = 120

    def scrape(self) -> list[dict]:
        """Fetch and parse articles from the configured sources."""
        articles: list[dict] = []
        seen_urls: set[str] = set()

        for source_url in self.urls:
            html = self._fetch(source_url)
            if not html:
                continue

            if self._looks_like_feed(html):
                feed_entries = self._parse_feed_entries(html, source_url)
                for entry in feed_entries[: self.max_articles_per_source]:
                    entry_url = entry.get("url", "")
                    if not entry_url or entry_url in seen_urls:
                        continue
                    if entry.get("content") and len(entry["content"]) >= self.min_feed_length:
                        articles.append(entry)
                        seen_urls.add(entry_url)
                        continue

                    article_html = self._fetch(entry_url)
                    if not article_html:
                        continue
                    article_soup = BeautifulSoup(article_html, "html.parser")
                    article = self._parse_article(article_soup, entry_url)
                    if article:
                        if entry.get("title") and article.get("title") == "Untitled":
                            article["title"] = entry["title"]
                        if entry.get("date") and not article.get("date"):
                            article["date"] = entry["date"]
                        articles.append(article)
                        seen_urls.add(entry_url)
                continue

            soup = BeautifulSoup(html, "html.parser")
            if self._looks_like_article(soup):
                article = self._parse_article(soup, source_url)
                if article and article["url"] not in seen_urls:
                    articles.append(article)
                    seen_urls.add(article["url"])
                continue

            links = self._extract_article_links(soup, source_url)
            for link in links[: self.max_articles_per_source]:
                if link in seen_urls:
                    continue
                article_html = self._fetch(link)
                if not article_html:
                    continue
                article_soup = BeautifulSoup(article_html, "html.parser")
                article = self._parse_article(article_soup, link)
                if article:
                    articles.append(article)
                    seen_urls.add(link)

        return articles

    def _fetch(self, url: str) -> str | None:
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"[scraper] Request failed for {url}: {exc}")
            return None
        return response.text

    def _looks_like_feed(self, text: str) -> bool:
        head = text.lstrip()[:500].lower()
        return "<rss" in head or "<feed" in head

    def _looks_like_article(self, soup: BeautifulSoup) -> bool:
        return bool(soup.find("article")) and bool(soup.find("h1"))

    def _parse_feed_entries(self, text: str, base_url: str) -> list[dict]:
        soup = BeautifulSoup(text, "xml")
        items = soup.find_all("item")
        if not items:
            items = soup.find_all("entry")

        entries: list[dict] = []
        for item in items:
            title = item.title.get_text(" ", strip=True) if item.title else "Untitled"
            link = ""
            link_tags = item.find_all("link")
            link_tag = None
            for candidate in link_tags:
                rel = candidate.get("rel") or []
                if isinstance(rel, str):
                    rel = [rel]
                if "alternate" in rel:
                    link_tag = candidate
                    break
            if not link_tag and link_tags:
                link_tag = link_tags[0]
            if link_tag:
                link = link_tag.get("href") or link_tag.get_text(" ", strip=True)
            if link and link.endswith(".pdf"):
                link = ""
            link = urljoin(base_url, link)
            date_tag = item.find("pubDate") or item.find("published") or item.find(
                "updated"
            )
            date = date_tag.get_text(" ", strip=True) if date_tag else ""
            summary_tag = (
                item.find("summary") or item.find("description") or item.find("content")
            )
            content = summary_tag.get_text(" ", strip=True) if summary_tag else ""
            content = re.sub(r"\s+", " ", content).strip()
            summary = self._summarize_text(content)
            if not link:
                continue
            entries.append(
                {
                    "title": title,
                    "url": link,
                    "date": date,
                    "content": content,
                    "summary": summary,
                }
            )
        return entries

    def _extract_article_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        base_netloc = urlparse(base_url).netloc
        raw_links: list[str] = []

        for article in soup.find_all("article"):
            for anchor in article.find_all("a", href=True):
                raw_links.append(urljoin(base_url, anchor["href"]))

        if not raw_links:
            for anchor in soup.find_all("a", href=True):
                raw_links.append(urljoin(base_url, anchor["href"]))

        unique_links: list[str] = []
        seen: set[str] = set()
        for link in raw_links:
            normalized = self._normalize_url(link)
            if not normalized or normalized in seen:
                continue
            if urlparse(normalized).netloc != base_netloc:
                continue
            if not self._is_probable_article(normalized):
                continue
            unique_links.append(normalized)
            seen.add(normalized)

        return unique_links

    def _normalize_url(self, url: str) -> str:
        if not url or url.startswith(("mailto:", "javascript:")):
            return ""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return ""
        return parsed._replace(query="", fragment="").geturl()

    def _is_probable_article(self, url: str) -> bool:
        parsed = urlparse(url)
        path = parsed.path.lower()
        if "arxiv.org" in parsed.netloc:
            return path.startswith(("/abs/", "/html/"))
        if path.endswith((".pdf", ".jpg", ".png", ".svg")):
            return False
        if path.count("/") >= 2:
            return True
        keywords = ("blog", "news", "article", "posts", "story", "research", "ai")
        return any(keyword in path for keyword in keywords)

    def _parse_article(self, soup: BeautifulSoup, url: str) -> dict | None:
        title = self._extract_title(soup)
        date = self._extract_date(soup)
        content = self._extract_text(soup)
        if not content or len(content) < self.min_article_length:
            return None
        summary = self._summarize_text(content)
        return {
            "title": title,
            "url": url,
            "date": date,
            "content": content,
            "summary": summary,
        }

    def _summarize_text(
        self, text: str, max_sentences: int = 3, max_chars: int = 500
    ) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        summary = " ".join(sentences[:max_sentences]).strip()
        if not summary:
            summary = cleaned
        if len(summary) > max_chars:
            summary = summary[:max_chars].rsplit(" ", 1)[0] or summary[:max_chars]
        return summary

    def _extract_title(self, soup: BeautifulSoup) -> str:
        meta_title = soup.find("meta", property="og:title") or soup.find(
            "meta", attrs={"name": "title"}
        )
        if meta_title and meta_title.get("content"):
            return meta_title["content"].strip()
        header = soup.find("h1")
        if header:
            return header.get_text(" ", strip=True)
        if soup.title:
            return soup.title.get_text(" ", strip=True)
        return "Untitled"

    def _extract_date(self, soup: BeautifulSoup) -> str:
        time_tag = soup.find("time")
        if time_tag:
            value = time_tag.get("datetime") or time_tag.get_text(" ", strip=True)
            if value:
                return value.strip()

        meta_candidates = [
            ("property", "article:published_time"),
            ("property", "og:pubdate"),
            ("name", "pubdate"),
            ("name", "date"),
            ("name", "dc.date"),
            ("name", "dc.date.issued"),
        ]
        for attr, key in meta_candidates:
            meta = soup.find("meta", attrs={attr: key})
            if meta and meta.get("content"):
                return meta["content"].strip()
        return ""

    def _extract_text(self, soup: BeautifulSoup) -> str:
        for tag in soup(
            ["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]
        ):
            tag.decompose()

        content = soup.find("article") or soup.body or soup
        text = content.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return text
