"""HTML -> Markdown cleaning with fit-markdown style pruning and optional BM25 filter."""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

from bs4 import BeautifulSoup
from readability import Document
from markdownify import markdownify as md
from rank_bm25 import BM25Okapi


NOISE_PATTERNS = [
    r"cookie(s)? policy",
    r"subscribe",
    r"sign in",
    r"all rights reserved",
    r"copyright",
    r"newsletter",
    r"advert",
]


def _strip_boilerplate(lines: Iterable[str], keywords: List[str]) -> List[str]:
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) < 25 and not any(k.lower() in stripped.lower() for k in keywords):
            continue
        if any(re.search(p, stripped, flags=re.IGNORECASE) for p in NOISE_PATTERNS):
            continue
        # drop lines that are mostly links
        link_ratio = stripped.count("http") / max(len(stripped.split()), 1)
        if link_ratio > 0.4:
            continue
        cleaned.append(stripped)
    return cleaned


def _bm25_filter(paragraphs: List[str], keywords: List[str], top_k: int) -> List[str]:
    tokens = [p.lower().split() for p in paragraphs]
    if not tokens:
        return []
    bm25 = BM25Okapi(tokens)
    scores = bm25.get_scores(keywords)
    ranked = sorted(zip(paragraphs, scores), key=lambda x: x[1], reverse=True)
    return [p for p, _ in ranked[:top_k]]


def html_to_markdown(
    html: str,
    keywords: List[str],
    use_bm25: bool = False,
    bm25_top_k: int = 24,
) -> Tuple[str, str]:
    """Convert HTML to cleaned markdown and plain text."""
    doc = Document(html)
    summary_html = doc.summary()
    soup = BeautifulSoup(summary_html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    markdown = md(str(soup), heading_style="ATX")
    lines = markdown.splitlines()
    pruned = _strip_boilerplate(lines, keywords)
    if use_bm25:
        pruned = _bm25_filter(pruned, keywords, bm25_top_k)
    cleaned_md = "\n".join(pruned)
    text = "\n".join([BeautifulSoup(summary_html, "html.parser").get_text(separator="\n")])
    return cleaned_md.strip(), text.strip()


__all__ = ["html_to_markdown"]
