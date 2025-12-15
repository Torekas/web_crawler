"""CLI entrypoint with crawl/clean/reindex/chat/test-crawl commands."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from .chat import chat_once
from .cleaner import run_clean
from .config import Config, load_config
from .crawler import CrawlService
from .indexer import ChromaIndexer, resolve_backend
from .storage import Storage

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _build_indexer(config: Config):
    backend = resolve_backend(config.index.embedding_backend, config.index.embedding_model, ollama_host=config.chat.ollama_host)
    return ChromaIndexer(
        path=config.storage.chroma_path,
        collection_name="pages",
        embedding_backend=backend,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="crawl4ai-plus pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    crawl_p = sub.add_parser("crawl", help="Run crawler")
    crawl_p.add_argument("--seeds", nargs="+", help="Override seeds")

    sub.add_parser("clean", help="Run memory cleaner")
    sub.add_parser("reindex", help="Rebuild Chroma index from DB")

    chat_p = sub.add_parser("chat", help="Ask a question using RAG over Chroma")
    chat_p.add_argument("--query", required=True)
    chat_p.add_argument("--top-k", type=int, default=None)

    sub.add_parser("test-crawl", help="Run a short crawl with 2 pages for smoke testing")
    return parser


async def _run_crawl(config: Config, seeds: list[str] | None = None) -> None:
    storage = Storage(config.storage.db_url)
    storage.init_db()
    indexer = _build_indexer(config)
    seeds = seeds or config.crawler.seeds
    service = CrawlService(config=config, storage=storage, indexer=indexer)
    result = await service.crawl(seeds)
    logger.info("Crawl done: %s", result)


def _run_reindex(config: Config) -> None:
    storage = Storage(config.storage.db_url)
    storage.init_db()
    indexer = _build_indexer(config)
    for page in storage.iter_pages():
        if not page.markdown:
            continue
        chunks = []
        text = page.markdown
        from .chunker import chunk_text

        chunks = chunk_text(text, chunk_size=config.index.chunk_size, overlap=config.index.chunk_overlap, min_size=config.index.min_chunk_chars)
        for c in chunks:
            c.setdefault("metadata", {}).update({"page_id": page.id, "url": page.canonical_url})
        if chunks:
            indexer.add_chunks(chunks)
    logger.info("Reindex complete")


def _run_clean(config: Config) -> None:
    storage = Storage(config.storage.db_url)
    storage.init_db()
    stats = run_clean(storage)
    logger.info("Cleaned memory: %s", stats)


def _run_chat(config: Config, query: str, top_k: int | None = None) -> None:
    indexer = _build_indexer(config)
    answer = chat_once(query=query, indexer=indexer, ollama_host=config.chat.ollama_host, model=config.chat.ollama_model, top_k=top_k or config.chat.top_k)
    print(answer)


def _run_test_crawl(config: Config) -> None:
    seeds = config.crawler.seeds[:2] if config.crawler.seeds else ["https://example.com"]
    config.crawler.max_pages = 2
    asyncio.run(_run_crawl(config, seeds=seeds))


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "crawl":
        asyncio.run(_run_crawl(config, seeds=args.seeds))
    elif args.command == "clean":
        _run_clean(config)
    elif args.command == "reindex":
        _run_reindex(config)
    elif args.command == "chat":
        _run_chat(config, query=args.query, top_k=args.top_k)
    elif args.command == "test-crawl":
        _run_test_crawl(config)


if __name__ == "__main__":
    main()
