"""Playwright-based fetcher with session reuse and failure screenshots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from playwright.async_api import BrowserContext, Playwright, async_playwright

from .url_utils import canonicalize_url


@dataclass
class BrowserResult:
    url: str
    content: str | None
    status: int | None
    final_url: str | None
    screenshot_path: str | None
    error: str | None = None


class BrowserFetcher:
    """Managed browser session that reuses context and cookies."""

    def __init__(
        self,
        headless: bool = True,
        proxy: Optional[str] = None,
        cookies_path: str = "data/cookies.json",
        screenshot_dir: str = "data/screenshots",
        wait_seconds: float = 2.0,
        scroll_pause: float = 0.2,
    ) -> None:
        self.headless = headless
        self.proxy = proxy
        self.cookies_path = Path(cookies_path)
        self.screenshot_dir = Path(screenshot_dir)
        self.wait_seconds = wait_seconds
        self.scroll_pause = scroll_pause
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._browser = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def start(self) -> None:
        if self._playwright:
            return
        self._playwright = await async_playwright().start()
        launch_kwargs = {"headless": self.headless}
        if self.proxy:
            launch_kwargs["proxy"] = {"server": self.proxy}
        self._browser = await self._playwright.chromium.launch(**launch_kwargs)
        self._context = await self._browser.new_context()
        if self.cookies_path.exists():
            try:
                cookies = json.loads(self.cookies_path.read_text(encoding="utf-8"))
                await self._context.add_cookies(cookies)
            except json.JSONDecodeError:
                pass

    async def close(self) -> None:
        if self._context:
            cookies = await self._context.cookies()
            self.cookies_path.parent.mkdir(parents=True, exist_ok=True)
            self.cookies_path.write_text(json.dumps(cookies), encoding="utf-8")
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._playwright = None
        self._browser = None
        self._context = None

    async def fetch(self, url: str, infinite_scroll: bool = False, scroll_timeout: float = 8.0) -> BrowserResult:
        if not self._context:
            await self.start()
        assert self._context is not None
        canonical = canonicalize_url(url)
        page = await self._context.new_page()
        screenshot_path: str | None = None
        try:
            resp = await page.goto(canonical, wait_until="load")
            await page.wait_for_timeout(self.wait_seconds * 1000)
            if infinite_scroll:
                await self._scroll_page(page, scroll_timeout)
            html = await page.content()
            status = resp.status if resp else None
            final_url = page.url
            return BrowserResult(
                url=canonical,
                content=html,
                status=status,
                final_url=final_url,
                screenshot_path=None,
            )
        except Exception as exc:  # noqa: BLE001
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
            import uuid

            screenshot_path = str(self.screenshot_dir / f"error_{uuid.uuid4().hex}.png")
            try:
                await page.screenshot(path=screenshot_path, full_page=True)
            except Exception:
                screenshot_path = None
            return BrowserResult(
                url=canonical,
                content=None,
                status=None,
                final_url=None,
                screenshot_path=screenshot_path,
                error=str(exc),
            )
        finally:
            await page.close()

    async def _scroll_page(self, page, timeout: float) -> None:
        import asyncio

        end_time = asyncio.get_event_loop().time() + timeout
        last_height = await page.evaluate("() => document.body.scrollHeight")
        while asyncio.get_event_loop().time() < end_time:
            await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(self.scroll_pause * 1000)
            new_height = await page.evaluate("() => document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height


__all__ = ["BrowserFetcher", "BrowserResult"]
