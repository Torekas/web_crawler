"""SQLite storage using SQLAlchemy for crawler state, memory, and index metadata."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True)
    status = Column(String, default="pending", index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    params = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    webhook_notified = Column(Boolean, default=False)


class Page(Base):
    __tablename__ = "pages"
    id = Column(Integer, primary_key=True)
    url = Column(String, index=True, nullable=False)
    canonical_url = Column(String, index=True, nullable=False)
    domain = Column(String, index=True, nullable=False)
    title = Column(String, nullable=True)
    html = Column(Text, nullable=True)
    markdown = Column(Text, nullable=True)
    text = Column(Text, nullable=True)
    meta = Column("metadata", JSON, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow, index=True)
    status_code = Column(Integer, nullable=True)
    content_hash = Column(String, nullable=True, index=True)
    topic_score = Column(Float, default=0.0)
    is_duplicate = Column(Boolean, default=False)
    __table_args__ = (UniqueConstraint("canonical_url", name="uq_page_canonical"),)

    chunks = relationship("Chunk", back_populates="page", cascade="all, delete-orphan")
    diagnostics = relationship("Diagnostic", back_populates="page", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    page_id = Column(Integer, ForeignKey("pages.id"), index=True)
    chunk_id = Column(String, unique=True)
    text = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=True)
    meta = Column("metadata", JSON, nullable=True)

    page = relationship("Page", back_populates="chunks")


class Extraction(Base):
    __tablename__ = "extractions"
    id = Column(Integer, primary_key=True)
    page_id = Column(Integer, ForeignKey("pages.id"))
    schema = Column(JSON, nullable=True)
    data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Diagnostic(Base):
    __tablename__ = "diagnostics"
    id = Column(Integer, primary_key=True)
    page_id = Column(Integer, ForeignKey("pages.id"))
    timing_ms = Column(Float, nullable=True)
    redirect_chain = Column(JSON, nullable=True)
    extractor_notes = Column(Text, nullable=True)
    screenshot_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    page = relationship("Page", back_populates="diagnostics")


class Storage:
    """Thin wrapper over SQLAlchemy sessions for crawler state and memory."""

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self.engine = create_engine(db_url, future=True)
        self.SessionLocal = sessionmaker(self.engine, expire_on_commit=False, future=True)

    def init_db(self) -> None:
        Base.metadata.create_all(self.engine)

    def session(self):
        return self.SessionLocal()

    def upsert_job(self, job_id: str, status: str, params: Optional[Dict[str, Any]] = None, error: str | None = None):
        with self.session() as s:
            job = s.get(Job, job_id)
            if job is None:
                job = Job(id=job_id, status=status, params=params or {})
                s.add(job)
            job.status = status
            job.error = error
            job.updated_at = datetime.utcnow()
            s.commit()

    def save_page(
        self,
        canonical_url: str,
        url: str,
        domain: str,
        html: str | None,
        markdown: str | None,
        text: str | None,
        metadata: Dict[str, Any],
        status_code: int | None,
        content_hash: str | None,
        topic_score: float,
        is_duplicate: bool,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Page:
        with self.session() as s:
            existing = s.query(Page).filter(Page.canonical_url == canonical_url).one_or_none()
            if existing:
                existing._was_existing = True  # runtime flag to signal caller to skip re-indexing
                return existing
            page = Page(
                canonical_url=canonical_url,
                url=url,
                domain=domain,
                html=html,
                markdown=markdown,
                text=text,
                meta=metadata,
                status_code=status_code,
                fetched_at=datetime.utcnow(),
                content_hash=content_hash,
                topic_score=topic_score,
                is_duplicate=is_duplicate,
                title=metadata.get("title") if metadata else None,
            )
            page._was_existing = False
            s.add(page)
            s.flush()
            if diagnostics:
                diag = Diagnostic(
                    page_id=page.id,
                    timing_ms=diagnostics.get("timing_ms"),
                    redirect_chain=diagnostics.get("redirect_chain"),
                    extractor_notes=diagnostics.get("extractor_notes"),
                    screenshot_path=diagnostics.get("screenshot_path"),
                )
                s.add(diag)
            s.commit()
            s.refresh(page)
            return page

    def save_chunks(self, page_id: int, chunks: Iterable[Dict[str, Any]]) -> List[Chunk]:
        saved: List[Chunk] = []
        with self.session() as s:
            for c in chunks:
                chunk = Chunk(
                    page_id=page_id,
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    embedding=c.get("embedding"),
                    meta=c.get("metadata"),
                )
                s.add(chunk)
                saved.append(chunk)
            s.commit()
        return saved

    def recent_pages(self, limit: int = 50) -> List[Page]:
        with self.session() as s:
            return list(s.query(Page).order_by(Page.fetched_at.desc()).limit(limit))

    def iter_pages(self, limit: int | None = None):
        with self.session() as s:
            query = s.query(Page).filter(Page.is_duplicate.is_(False)).order_by(Page.fetched_at.desc())
            if limit:
                query = query.limit(limit)
            for page in query:
                yield page

    def mark_duplicates_by_hash(self) -> int:
        """Mark pages with identical content_hash as duplicates."""
        updated = 0
        with self.session() as s:
            hashes = {}
            for page in s.query(Page).order_by(Page.fetched_at.asc()):
                if page.content_hash and page.content_hash in hashes:
                    page.is_duplicate = True
                    updated += 1
                else:
                    if page.content_hash:
                        hashes[page.content_hash] = page.id
            s.commit()
        return updated

    def purge_duplicates(self) -> int:
        """Delete duplicate pages and their chunks."""
        deleted = 0
        with self.session() as s:
            dup_pages = list(s.query(Page).filter(Page.is_duplicate.is_(True)))
            for page in dup_pages:
                s.delete(page)
                deleted += 1
            s.commit()
        return deleted

    def save_extraction(self, page_id: int, schema: Dict[str, Any] | None, data: Dict[str, Any] | None) -> Extraction:
        with self.session() as s:
            record = Extraction(page_id=page_id, schema=schema, data=data)
            s.add(record)
            s.commit()
            s.refresh(record)
            return record


def ensure_dirs(*paths: str | Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


__all__ = ["Storage", "Base", "Job", "Page", "Chunk", "Extraction", "Diagnostic", "ensure_dirs"]
