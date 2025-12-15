"""Structured extraction helpers using CSS/XPath and optional LLM stub."""

from __future__ import annotations

from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from lxml import html as lxml_html


def extract_with_selectors(
    html: str,
    css_selectors: Optional[List[str]] = None,
    xpath_selectors: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    soup = BeautifulSoup(html, "html.parser")
    output: Dict[str, List[str]] = {}
    for selector in css_selectors or []:
        output[selector] = [el.get_text(strip=True) for el in soup.select(selector)]
    if xpath_selectors:
        tree = lxml_html.fromstring(html)
        for selector in xpath_selectors:
            output[selector] = [t.strip() for t in tree.xpath(selector)]
    return output


def extract_with_llm_stub(html: str, schema: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Placeholder that can be replaced with a real LLM extraction call."""
    text = BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)
    result = {"schema": schema or {}, "preview": text[:500]}
    return result


__all__ = ["extract_with_selectors", "extract_with_llm_stub"]
