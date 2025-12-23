# AI News Vector Crawler

A small Python tool that collects AI news and arXiv papers, generates text embeddings, stores them in a persistent local ChromaDB database, and lets you run semantic queries from the CLI.

## What it does
- Scrapes multiple AI news sources (RSS/Atom + HTML).
- Pulls recent papers from arXiv (multiple categories + queries).
- Generates lightweight summaries for each article or feed entry.
- Chunks content, embeds it with `sentence-transformers`, and stores it in ChromaDB.
- Provides an interactive semantic search loop.

## Project layout
- `main.py` - Entry point; defines sources and interactive search.
- `scraper.py` - Fetching, parsing, cleanup, and summary generation.
- `vector_store.py` - Embedding, chunking, and ChromaDB storage/query.
- `requirements.txt` - Runtime dependencies.
- `db/` - Persistent ChromaDB storage (created on first run).

## Default sources
Company / news feeds:
- OpenAI news RSS
- Anthropic news + research pages
- Google DeepMind RSS
- Google AI (The Keyword) RSS
- Microsoft AI blog RSS
- Azure blog RSS
- AWS ML blog RSS
- NVIDIA deep learning blog RSS
- Meta (about.fb.com) news RSS
- Meta Engineering RSS
- TechCrunch AI RSS

arXiv queries:
- `cs.AI`, `cs.LG`, `cs.CL`, `cs.CV`, `stat.ML`, `cs.RO`
- Text queries: "explainable ai", "interpretability"

## Setup
Create / activate a virtual environment and install dependencies:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ai_news_crawler\requirements.txt
```

## Run
```powershell
python ai_news_crawler\main.py
```

You will see how many articles were fetched and how many chunks were stored. Then you can ask questions interactively, for example:
- "What are the newest video model releases?"
- "Explainable AI methods in recent arXiv papers"
- "Recent robotics breakthroughs"

Type `exit` to quit.

## How summaries work
The scraper builds a short summary by taking the first few sentences of the article/feed content (simple heuristic, no external API). Summaries are stored in metadata and printed with search results.

## Storage and persistence
- ChromaDB is stored in `ai_news_crawler/db/`.
- The database is persistent across runs.
- If you want to re-ingest everything from scratch, delete `ai_news_crawler/db/` and run again.

## Customizing sources
Edit the `sources` list in `ai_news_crawler/main.py`. You can add RSS/Atom URLs or HTML pages. For arXiv, use the export API, for example:
```
https://export.arxiv.org/api/query?search_query=cat:cs.AI&start=0&max_results=25&sortBy=submittedDate&sortOrder=descending
```

## Troubleshooting
- Some sites block scraping (403/404). Prefer RSS/Atom feeds when available.
- If a source is blocked, replace it with its RSS/Atom feed or remove it.
- Large runs can create many chunks; the vector store batches inserts to avoid size limits.

## Dependencies
- `requests`
- `beautifulsoup4`
- `chromadb`
- `sentence-transformers`
