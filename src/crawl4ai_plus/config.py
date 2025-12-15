"""Configuration loader for the crawl4ai-inspired crawler.

Reads YAML configuration, applies environment variable overrides, and returns typed
dataclasses consumed across the crawler stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def _merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_overrides(config: Dict[str, Any], prefix: str = "CRAWL") -> Dict[str, Any]:
    """Override config using env vars like CRAWL_CRAWLER__MAX_PAGES=200."""
    overrides: Dict[str, Any] = {}
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix + "_"):
            continue
        trimmed = env_key[len(prefix) + 1 :]
        keys = trimmed.lower().split("__")
        cursor = overrides
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = _coerce_env_value(env_val)
    return _merge_dicts(config, overrides)


def _coerce_env_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if "," in value:
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


@dataclass
class RateLimitSettings:
    per_domain_delay: float = 1.0
    max_concurrency: int = 4
    max_retries: int = 3
    backoff_base: float = 0.6
    circuit_breaker_failures: int = 5
    circuit_breaker_reset: int = 900


@dataclass
class CrawlerSettings:
    seeds: List[str] = field(default_factory=lambda: [])
    allowed_domains: List[str] = field(default_factory=list)
    denied_domains: List[str] = field(default_factory=list)
    allow_url_patterns: List[str] = field(default_factory=list)
    deny_url_patterns: List[str] = field(default_factory=list)
    max_depth: int = 3
    max_pages: int = 120
    strategy: str = "bfs"  # bfs | dfs | best-first
    user_agent: str = "crawl4ai-plus/0.1"
    obey_robots: bool = True
    cache_dir: str = "data/cache"
    content_cache_ttl: int = 86400
    use_browser: bool = False
    topic_keywords: List[str] = field(default_factory=lambda: [
        "artificial intelligence",
        " ai ",
        "machine learning",
        " ml ",
        "deep learning",
        "llm",
        "language model",
        "foundation model",
        "genai",
        "agentic workflows",
        "chain of thought",
        "reasoning models",
        "model alignment",
        "retrieval augmented generation",
        "vector database",
        "embedding model",
        "reranking",
        "multi-agent",
    ])
    dynamic_wait_seconds: float = 2.0
    infinite_scroll: bool = False
    scroll_timeout: float = 8.0
    max_urls_per_domain: int = 400


@dataclass
class BrowserSettings:
    headless: bool = True
    proxy: Optional[str] = None
    viewport: str = "1280,720"
    cookies_path: str = "data/cookies.json"
    stealth: bool = True
    js_wait: float = 2.0
    scroll_pause: float = 0.2
    screenshot_dir: str = "data/screenshots"


@dataclass
class StorageSettings:
    db_url: str = "sqlite:///data/memory.db"
    raw_html_dir: str = "data/raw_html"
    markdown_dir: str = "data/markdown"
    diagnostics_dir: str = "logs"
    chroma_path: str = "data/chroma"


@dataclass
class ExtractionSettings:
    css_selectors: List[str] = field(default_factory=list)
    xpath_selectors: List[str] = field(default_factory=list)
    llm_schema: Optional[Dict[str, Any]] = None
    enable_llm_extraction: bool = False


@dataclass
class IndexSettings:
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    min_chunk_chars: int = 200
    use_bm25_filter: bool = False
    bm25_top_k: int = 24


@dataclass
class ApiSettings:
    host: str = "0.0.0.0"
    port: int = 8000
    jwt_secret: Optional[str] = None
    enable_jwt: bool = False


@dataclass
class ChatSettings:
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "mixtral:8x7b"
    top_k: int = 4


@dataclass
class WebhookSettings:
    enabled: bool = False
    url: Optional[str] = None
    max_attempts: int = 3
    backoff_seconds: float = 1.5


@dataclass
class Config:
    crawler: CrawlerSettings = field(default_factory=CrawlerSettings)
    browser: BrowserSettings = field(default_factory=BrowserSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    extraction: ExtractionSettings = field(default_factory=ExtractionSettings)
    index: IndexSettings = field(default_factory=IndexSettings)
    rate_limits: RateLimitSettings = field(default_factory=RateLimitSettings)
    api: ApiSettings = field(default_factory=ApiSettings)
    chat: ChatSettings = field(default_factory=ChatSettings)
    webhook: WebhookSettings = field(default_factory=WebhookSettings)


def load_config(path: Optional[Path | str] = None, env_prefix: str = "CRAWL") -> Config:
    """Load YAML config and merge env overrides."""
    config_path = Path(path) if path else Path("config.yaml")
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            data = _expand_env(yaml.safe_load(f) or {})
    else:
        data = {}
    merged_dict = _apply_env_overrides(data, prefix=env_prefix)
    return map_dict_to_config(merged_dict)


def map_dict_to_config(data: Dict[str, Any]) -> Config:
    crawler = CrawlerSettings(**data.get("crawler", {}))
    browser = BrowserSettings(**data.get("browser", {}))
    storage = StorageSettings(**data.get("storage", {}))
    extraction = ExtractionSettings(**data.get("extraction", {}))
    index = IndexSettings(**data.get("index", {}))
    rate_limits = RateLimitSettings(**data.get("rate_limits", {}))
    api = ApiSettings(**data.get("api", {}))
    chat = ChatSettings(**data.get("chat", {}))
    webhook = WebhookSettings(**data.get("webhook", {}))
    return Config(
        crawler=crawler,
        browser=browser,
        storage=storage,
        extraction=extraction,
        index=index,
        rate_limits=rate_limits,
        api=api,
        chat=chat,
        webhook=webhook,
    )


__all__ = [
    "Config",
    "CrawlerSettings",
    "BrowserSettings",
    "StorageSettings",
    "IndexSettings",
    "ApiSettings",
    "ChatSettings",
    "WebhookSettings",
    "RateLimitSettings",
    "ExtractionSettings",
    "load_config",
    "map_dict_to_config",
]
