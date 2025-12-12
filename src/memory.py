import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class MemoryEntry:
    kind: str
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class MemoryStore:
    def __init__(self, path: Path = Path("data/memory_longterm.jsonl")) -> None:
        self.path = path
        self.entries: List[MemoryEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    payload = json.loads(line)
                    self.entries.append(MemoryEntry(**payload))
                except Exception:
                    continue

    def _write(self, entry: MemoryEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")

    def add(self, kind: str, content: str, meta: Optional[Dict[str, Any]] = None) -> MemoryEntry:
        entry = MemoryEntry(kind=kind, content=content, meta=meta or {})
        self.entries.append(entry)
        self._write(entry)
        return entry

    def recent(self, *, kind: Optional[str] = None, limit: int = 5) -> List[MemoryEntry]:
        filtered = [e for e in self.entries if kind is None or e.kind == kind]
        return filtered[-limit:]

    def extend(self, entries: Iterable[MemoryEntry]) -> None:
        for entry in entries:
            self.add(entry.kind, entry.content, entry.meta)


class ShortTermMemory:
    def __init__(self, max_turns: int = 8) -> None:
        self.max_turns = max_turns
        self.turns: List[Dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def conversation_text(self) -> str:
        return "\n".join(f"{t['role']}: {t['content']}" for t in self.turns)
