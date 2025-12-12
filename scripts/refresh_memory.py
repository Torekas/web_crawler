import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


MEMORY_PATH = Path("data/memory_longterm.jsonl")
PAGES_PATH = Path("data/pages.jsonl")


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_page_entry(page: Dict) -> Dict:
    return {
        "kind": "page",
        "content": page.get("url"),
        "meta": {
            "title": page.get("title"),
            "fetched_at": page.get("fetched_at"),
            "relevance_score": page.get("relevance_score"),
            "confidence": page.get("confidence", 0.0),
            "decision": page.get("decision"),
        },
        "created_at": page.get("fetched_at") or datetime.utcnow().isoformat(),
    }


def main() -> None:
    memory_records = load_jsonl(MEMORY_PATH)
    page_records = load_jsonl(PAGES_PATH)

    seen: set[Tuple[str, str]] = set()
    cleaned: List[Dict] = []
    dropped = 0
    corrected = 0

    for rec in memory_records:
        if not isinstance(rec, dict):
            dropped += 1
            continue
        kind = rec.get("kind")
        content = rec.get("content")
        if not kind or not content:
            dropped += 1
            continue
        if (kind, content) in seen:
            dropped += 1
            continue
        if kind != "page" and isinstance(rec.get("meta"), dict) and rec["meta"].get("decision"):
            rec["kind"] = "page"
            kind = "page"
            corrected += 1
        seen.add((kind, content))
        cleaned.append(rec)

    added = 0
    for page in page_records:
        url = page.get("url")
        if not url:
            continue
        key = ("page", url)
        if key in seen:
            continue
        cleaned.append(build_page_entry(page))
        seen.add(key)
        added += 1

    write_jsonl(MEMORY_PATH, cleaned)
    print(f"memory cleaned: kept {len(cleaned)}, dropped {dropped} duplicates/invalid, corrected {corrected} kinds, added {added} missing pages")


if __name__ == "__main__":
    main()
