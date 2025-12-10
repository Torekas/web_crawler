# AI Web Crawler + RAG

Minimal crawler that harvests AI-related web pages and builds a local vector index you can query conversationally.

## Setup
- Python 3.10+ recommended (`py` launcher on Windows, or `python`/`python3` elsewhere).
- Install deps: `pip install -r requirements.txt`
- Chat LLM (default): install [Ollama](https://ollama.com/) and pull a model (e.g. `ollama pull mixtral:8x7b`), keep the service running.
- Optional OpenAI fallback: set `OPENAI_API_KEY` (`setx OPENAI_API_KEY "sk-..."` on Windows PowerShell or `export OPENAI_API_KEY=...` on Unix shells).

## 1) Crawl AI pages
```bash
py -m src.main crawl --max-pages 80 --depth 2 --delay 1.0
```
- Seeds default to well-known AI sources (OpenAI, DeepMind, Anthropic, Stability AI, Microsoft Research, HuggingFace, Cohere, Google AI Blog).
- Output is JSONL at `data/pages.jsonl`. Only AI-relevant pages are kept (keyword filter).

## 2) Build the vector index
```bash
py -m src.main index --pages data/pages.jsonl --index data/index.pkl.gz --model sentence-transformers/all-MiniLM-L6-v2
```
- Text is chunked with overlap, embedded, and saved to the compressed index.

## 3) Query / chat
- Interactive chat (defaults to Ollama + mixtral:8x7b):
```bash
py -m src.main chat --index data/index.pkl.gz --top-k 4 --llm ollama --ollama-model mixtral:8x7b
```
- Use OpenAI instead:
```bash
py -m src.main chat --index data/index.pkl.gz --top-k 4 --llm openai --openai-model gpt-4o-mini
```
- Single-shot search:
```bash
py -m src.main search --index data/index.pkl.gz --query "latest transformer efficiency tricks"
```

## Customizing
- Add/remove seed URLs: `--seeds https://example.com https://another.ai`
- Increase coverage: bump `--max-pages` and `--depth` (respect crawl politeness; default 1s delay).
- Embeddings: swap models with `--model` during indexing/search (keeps cache per model).
- Chat model: pick any Ollama model via `--ollama-model` (default mixtral:8x7b) or switch to OpenAI via `--llm openai`.

## Notes
- The crawler checks `robots.txt`, skips binary assets, and de-duplicates URLs.
- Data and index files are git-ignored under `data/`.
- For production, consider running behind a proxy cache, persisting raw HTML, and adding retries/backoff.
