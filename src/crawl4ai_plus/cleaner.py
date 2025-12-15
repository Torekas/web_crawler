"""Memory cleaning module to deduplicate and compact crawler state."""

from __future__ import annotations

from typing import Dict

from .storage import Storage


def run_clean(storage: Storage) -> Dict[str, int]:
    marked = storage.mark_duplicates_by_hash()
    deleted = storage.purge_duplicates()
    return {"marked": marked, "deleted": deleted}


__all__ = ["run_clean"]
