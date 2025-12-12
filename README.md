# GPU-Ready AI Knowledge Agent (Crawl + RAG + Chat)

Async, domain-aware crawler and RAG stack tuned for LLM/AI topics. Captures curated pages, builds a GPU-ready index, and serves sourced answers with short- and long-term memory.

## What’s inside
- **Crawler**: Async fetch with polite delays, UA rotation, robots.txt respect, short-page guard, heuristic + optional LLM judge/verification, and a `requests` fallback when `aiohttp` fails. Deduped writes; seeds are re-visited each run for fresh links. Crawl summary logs visited/kept/skipped and output path.
- **Index**: SentenceTransformer embeddings (GPU preferred), chunking with overlap and minimum-word filtering.
- **Chat**: Retrieval + recency re-rank + URL validation. Answers are prefixed with `Answer:` and include inline `[n]` citations plus a “References” block with titles, links, and fetched times. Full answers are stored in long-term memory even if the UI truncates the display.
- **Memory**: `data/memory_longterm.jsonl` keeps reflections and conversations; short-term memory grounds the current chat.

## Setup
- Python 3.10+
- Install PyTorch with CUDA 11.8 (critical):
  ```bash
  pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- Install the remaining deps:
  ```bash
  pip install -r requirements.txt
  ```
- GPU is auto-detected (`cuda` > `mps` > `cpu`) for embeddings.

## 1) Domain-aware crawl
```bash
py -m src.main crawl --max-pages 120 --depth 5 --concurrency 6 --delay 0.8 --judge-llm ollama --judge-model mixtral:8x7b --output data/pages.jsonl
```
- Seeds default to major AI labs/blogs; stays within allowed domains and respects robots.txt.
- Judge options: `--judge-llm openai --judge-openai-model gpt-4o-mini` or `--judge-llm none` to rely on heuristics only.
- Short pages at depth 0 (seeds) are kept to enable link expansion; deeper pages under 60 words are skipped with a reflexion note.

## 2) Build the vector index (GPU embeddings)
```bash
py -m src.main index --pages data/pages.jsonl --index data/index.pkl.gz --model sentence-transformers/all-MiniLM-L6-v2
```
- Text is chunked with overlap and embedded on GPU when available.

## 3) Conversational RAG with memory
```bash
py -m src.main chat --index data/index.pkl.gz --top-k 4 --llm ollama --ollama-model mixtral:8x7b
```
- OpenAI instead:
```bash
py -m src.main chat --index data/index.pkl.gz --top-k 4 --llm openai --openai-model gpt-4o-mini
```
- Single-shot search (no chat/memory):
```bash
py -m src.main search --index data/index.pkl.gz --query "latest reflexion-based RAG improvements"
```
- Chat output format:
  - `Answer:` section (2–6 sentences, fact-rich, cites `[n]`)
  - `References:` block listing numbered titles, URLs, and `fetched_at`
  - If display is truncated, the full answer still persists to long-term memory.

## Data & logs
- `data/pages.jsonl`: Captured pages (deduped appends).
- `data/index.pkl.gz`: Embeddings index.
- `data/memory_longterm.jsonl`: Reflections + full chat answers and sources.
- `logs/`: Crawl logs if enabled externally.

## More details
See `docs/architecture.md` for workflow diagrams and component notes.
