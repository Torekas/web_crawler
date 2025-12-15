"""URL utilities: canonicalization, domain filtering, and helpers."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Optional
from urllib.parse import parse_qsl, urlparse, urlunparse


DEFAULT_QUERY_FILTERS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_content",
    "utm_term",
    "fbclid",
    "gclid",
    "yclid",
    "mc_eid",
}


def canonicalize_url(url: str) -> str:
    """Canonicalize URLs for deduplication."""
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower() or "http"
    netloc = parsed.hostname.lower() if parsed.hostname else ""
    if not netloc:
        return url
    port = parsed.port
    if port and ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        port = None
    if port:
        netloc = f"{netloc}:{port}"
    path = re.sub(r"/+", "/", parsed.path) or "/"
    path = path.rstrip("/") or "/"
    query_pairs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=False) if k not in DEFAULT_QUERY_FILTERS]
    query_pairs.sort()
    query = "&".join(f"{k}={v}" for k, v in query_pairs)
    normalized = urlunparse((scheme, netloc, path, "", query, ""))
    return normalized


def url_hash(url: str) -> str:
    return hashlib.sha256(canonicalize_url(url).encode("utf-8")).hexdigest()


def domain_from_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed.hostname.lower() if parsed.hostname else ""


def matches_any(patterns: Iterable[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def should_crawl(
    url: str,
    allowed_domains: Optional[List[str]] = None,
    denied_domains: Optional[List[str]] = None,
    allow_patterns: Optional[List[str]] = None,
    deny_patterns: Optional[List[str]] = None,
) -> bool:
    domain = domain_from_url(url)
    if allowed_domains and domain and domain not in allowed_domains:
        return False
    if denied_domains and domain in denied_domains:
        return False
    if deny_patterns and matches_any(deny_patterns, url):
        return False
    if allow_patterns:
        return matches_any(allow_patterns, url)
    return True


__all__ = ["canonicalize_url", "domain_from_url", "url_hash", "should_crawl", "matches_any"]
