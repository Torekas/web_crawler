import asyncio
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from crawl4ai_plus.config import load_config
from crawl4ai_plus.crawler import CrawlService
from crawl4ai_plus.indexer import ChromaIndexer, resolve_backend
from crawl4ai_plus.storage import Storage


@pytest.fixture(scope="module")
def local_server():
    base_dir = Path(__file__).parent / "data"
    handler = partial(SimpleHTTPRequestHandler, directory=str(base_dir))
    server = ThreadingHTTPServer(("localhost", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join()


@pytest.mark.asyncio
async def test_crawl_local_site(tmp_path, local_server):
    port = local_server.server_address[1]
    seed = f"http://localhost:{port}/page1.html"
    config = load_config()
    config.crawler.seeds = [seed]
    config.crawler.allowed_domains = ["localhost"]
    config.crawler.max_pages = 3
    config.crawler.max_depth = 2
    config.crawler.obey_robots = False
    config.crawler.cache_dir = str(tmp_path / "cache")
    config.storage.db_url = f"sqlite:///{tmp_path/'memory.db'}"
    config.storage.chroma_path = str(tmp_path / "chroma")
    config.index.embedding_backend = "hash"

    storage = Storage(config.storage.db_url)
    storage.init_db()
    backend = resolve_backend(config.index.embedding_backend, config.index.embedding_model, ollama_host=config.chat.ollama_host)
    indexer = ChromaIndexer(path=config.storage.chroma_path, collection_name="pages", embedding_backend=backend)
    service = CrawlService(config=config, storage=storage, indexer=indexer, use_browser=False)
    result = await service.crawl(config.crawler.seeds)

    pages = storage.recent_pages()
    assert result["processed"] >= 2
    assert len(pages) >= 2

