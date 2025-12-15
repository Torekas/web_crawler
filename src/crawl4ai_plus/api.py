"""FastAPI server exposing crawl jobs and search/chat over the vector DB."""

from __future__ import annotations

import asyncio
import uuid
from typing import List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Header
from pydantic import BaseModel

from .chat import chat_once
from .config import Config, load_config
from .crawler import CrawlService
from .indexer import ChromaIndexer, resolve_backend
from .storage import Job, Storage


def _build_indexer(config: Config) -> ChromaIndexer:
    backend = resolve_backend(config.index.embedding_backend, config.index.embedding_model, ollama_host=config.chat.ollama_host)
    return ChromaIndexer(path=config.storage.chroma_path, collection_name="pages", embedding_backend=backend)


def create_app(config_path: str | None = None) -> FastAPI:
    config = load_config(config_path)
    storage = Storage(config.storage.db_url)
    storage.init_db()
    indexer = _build_indexer(config)

    app = FastAPI(title="crawl4ai-plus")
    app.state.config = config
    app.state.storage = storage
    app.state.indexer = indexer

    class CrawlRequest(BaseModel):
        seeds: Optional[List[str]] = None

    class SearchRequest(BaseModel):
        query: str
        top_k: int = 4

    class ChatRequest(BaseModel):
        query: str
        top_k: int = 4

    def get_storage() -> Storage:
        return app.state.storage

    def get_config() -> Config:
        return app.state.config

    def get_indexer() -> ChromaIndexer:
        return app.state.indexer

    async def require_auth(authorization: str | None = Header(default=None), config: Config = Depends(get_config)):
        if not config.api.enable_jwt:
            return
        if not authorization or authorization.replace("Bearer ", "") != (config.api.jwt_secret or ""):
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.post("/crawl")
    async def submit_crawl(
        req: CrawlRequest,
        background_tasks: BackgroundTasks,
        storage: Storage = Depends(get_storage),
        config: Config = Depends(get_config),
        indexer: ChromaIndexer = Depends(get_indexer),
        _: None = Depends(require_auth),
    ):
        job_id = str(uuid.uuid4())
        service = CrawlService(config=config, storage=storage, indexer=indexer)
        seeds = req.seeds or config.crawler.seeds
        background_tasks.add_task(asyncio.create_task, service.crawl(seeds=seeds, job_id=job_id))
        return {"job_id": job_id}

    @app.get("/status/{job_id}")
    async def job_status(job_id: str, storage: Storage = Depends(get_storage), _: None = Depends(require_auth)):
        with storage.session() as s:
            job = s.get(Job, job_id)
            if not job:
                raise HTTPException(status_code=404, detail="not found")
            return {"id": job.id, "status": job.status, "error": job.error, "params": job.params}

    @app.post("/search")
    async def search(req: SearchRequest, indexer: ChromaIndexer = Depends(get_indexer), _: None = Depends(require_auth)):
        return indexer.query(req.query, top_k=req.top_k)

    @app.post("/chat")
    async def chat(
        req: ChatRequest,
        config: Config = Depends(get_config),
        indexer: ChromaIndexer = Depends(get_indexer),
        _: None = Depends(require_auth),
    ):
        answer = chat_once(query=req.query, indexer=indexer, ollama_host=config.chat.ollama_host, model=config.chat.ollama_model, top_k=req.top_k)
        return {"answer": answer}

    return app


app = create_app()


__all__ = ["app", "create_app"]
