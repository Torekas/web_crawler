"""HTTP fetcher with caching, retries, rate limiting, and robots.txt politeness."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib import robotparser

import httpx

from .url_utils import canonicalize_url, domain_from_url, url_hash


@dataclass
class FetchResult:
    url: str
    status_code: int | None
    content: str | None
    elapsed: float
    final_url: str | None
    from_cache: bool
    error: str | None = None
    redirect_chain: list[str] | None = None


class PoliteHttpFetcher:
    """Async HTTP fetcher with per-domain rate limits, retry/backoff, cache, and robots check."""

    def __init__(
        self,
        user_agent: str,
        cache_dir: str,
        obey_robots: bool = True,
        per_domain_delay: float = 1.0,
        max_retries: int = 3,
        backoff_base: float = 0.6,
        timeout: float = 20.0,
        circuit_breaker_failures: int = 5,
        circuit_breaker_reset: int = 900,
    ) -> None:
        self.user_agent = user_agent
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.obey_robots = obey_robots
        self.per_domain_delay = per_domain_delay
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.circuit_breaker_failures = circuit_breaker_failures
        self.circuit_breaker_reset = circuit_breaker_reset
        self.robot_parsers: Dict[str, robotparser.RobotFileParser] = {}
        self.domain_locks: Dict[str, asyncio.Lock] = {}
        self.last_request: Dict[str, float] = {}
        self.client = httpx.AsyncClient(follow_redirects=True, headers={"User-Agent": user_agent})
        self.domain_failures: Dict[str, Dict[str, float]] = {}

    async def close(self) -> None:
        await self.client.aclose()

    async def _allowed_by_robots(self, url: str) -> bool:
        if not self.obey_robots:
            return True
        domain = domain_from_url(url)
        if not domain:
            return True
        if domain not in self.robot_parsers:
            robots_url = f"https://{domain}/robots.txt"
            parser = robotparser.RobotFileParser()
            try:
                resp = await self.client.get(robots_url, timeout=self.timeout)
                if resp.status_code < 400:
                    parser.parse(resp.text.splitlines())
                else:
                    parser.parse([])
            except httpx.HTTPError:
                parser.parse([])
            self.robot_parsers[domain] = parser
        parser = self.robot_parsers[domain]
        return parser.can_fetch(self.user_agent, url)

    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / f"{url_hash(url)}.json"

    def _load_cache(self, url: str, ttl: int) -> Optional[FetchResult]:
        path = self._cache_path(url)
        if not path.exists():
            return None
        if ttl > 0 and time.time() - path.stat().st_mtime > ttl:
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return FetchResult(
                url=data["url"],
                status_code=data.get("status_code"),
                content=data.get("content"),
                elapsed=data.get("elapsed", 0.0),
                final_url=data.get("final_url"),
                from_cache=True,
                error=None,
                redirect_chain=data.get("redirect_chain"),
            )
        except json.JSONDecodeError:
            return None

    def _write_cache(self, url: str, result: FetchResult) -> None:
        path = self._cache_path(url)
        payload = {
            "url": result.url,
            "status_code": result.status_code,
            "content": result.content,
            "elapsed": result.elapsed,
            "final_url": result.final_url,
            "redirect_chain": result.redirect_chain,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    async def fetch(self, url: str, cache_ttl: int = 3600) -> FetchResult:
        canonical = canonicalize_url(url)
        cached = self._load_cache(canonical, ttl=cache_ttl)
        if cached:
            return cached
        if not await self._allowed_by_robots(canonical):
            return FetchResult(
                url=canonical,
                status_code=None,
                content=None,
                elapsed=0.0,
                final_url=None,
                from_cache=False,
                error="blocked_by_robots",
                redirect_chain=None,
            )
        domain = domain_from_url(canonical)
        if self._circuit_open(domain):
            return FetchResult(
                url=canonical,
                status_code=None,
                content=None,
                elapsed=0.0,
                final_url=None,
                from_cache=False,
                error="circuit_open",
                redirect_chain=None,
            )
        lock = self.domain_locks.setdefault(domain, asyncio.Lock())
        async with lock:
            wait = 0.0
            last = self.last_request.get(domain)
            now = time.time()
            if last:
                wait = max(0.0, self.per_domain_delay - (now - last))
            if wait:
                await asyncio.sleep(wait)
            attempt = 0
            error: str | None = None
            response: httpx.Response | None = None
            redirect_chain: list[str] | None = None
            start = time.perf_counter()
            while attempt <= self.max_retries:
                try:
                    response = await self.client.get(canonical, timeout=self.timeout)
                    redirect_chain = [str(r.headers.get("location")) for r in response.history] if response.history else []
                    break
                except (httpx.TimeoutException, httpx.RequestError) as exc:
                    error = str(exc)
                    await asyncio.sleep(self.backoff_base * (2**attempt))
                    attempt += 1
            elapsed = (time.perf_counter() - start) * 1000
            self.last_request[domain] = time.time()

        if response is None:
            self._record_failure(domain)
            return FetchResult(
                url=canonical,
                status_code=None,
                content=None,
                elapsed=elapsed,
                final_url=None,
                from_cache=False,
                error=error or "request_failed",
                redirect_chain=redirect_chain,
            )
        result = FetchResult(
            url=canonical,
            status_code=response.status_code,
            content=response.text,
            elapsed=elapsed,
            final_url=str(response.url),
            from_cache=False,
            error=None,
            redirect_chain=redirect_chain,
        )
        if response.status_code < 400:
            self._write_cache(canonical, result)
            self._reset_failures(domain)
        else:
            self._record_failure(domain)
        return result

    def _record_failure(self, domain: str) -> None:
        entry = self.domain_failures.setdefault(domain, {"count": 0, "time": time.time()})
        entry["count"] += 1
        entry["time"] = time.time()

    def _reset_failures(self, domain: str) -> None:
        if domain in self.domain_failures:
            self.domain_failures.pop(domain)

    def _circuit_open(self, domain: str) -> bool:
        if domain not in self.domain_failures:
            return False
        entry = self.domain_failures[domain]
        if entry["count"] >= self.circuit_breaker_failures and (time.time() - entry["time"]) < self.circuit_breaker_reset:
            return True
        if (time.time() - entry["time"]) >= self.circuit_breaker_reset:
            self._reset_failures(domain)
        return False


__all__ = ["PoliteHttpFetcher", "FetchResult"]
