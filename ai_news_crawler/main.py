from pathlib import Path

from scraper import AINewsScraper
from vector_store import VectorArchive


def run() -> None:
    """Scrape AI news sources, store embeddings, and answer semantic queries."""
    sources = [
        "https://openai.com/news/rss.xml",
        "https://www.anthropic.com/news",
        "https://www.anthropic.com/research",
        "https://deepmind.google/blog/rss.xml",
        "https://blog.google/technology/ai/rss/",
        "https://blogs.microsoft.com/ai/feed/",
        "https://azure.microsoft.com/en-us/blog/feed/",
        "https://aws.amazon.com/blogs/machine-learning/feed/",
        "https://blogs.nvidia.com/blog/category/deep-learning/feed/",
        "https://about.fb.com/news/feed/",
        "https://engineering.fb.com/feed/",
        "https://techcrunch.com/tag/artificial-intelligence/feed/",
        "https://export.arxiv.org/api/query?search_query=cat:cs.AI&start=0&max_results=25&sortBy=submittedDate&sortOrder=descending",
        "https://export.arxiv.org/api/query?search_query=cat:cs.LG&start=0&max_results=25&sortBy=submittedDate&sortOrder=descending",
        "https://export.arxiv.org/api/query?search_query=cat:cs.CL&start=0&max_results=25&sortBy=submittedDate&sortOrder=descending",
        "https://export.arxiv.org/api/query?search_query=cat:cs.CV&start=0&max_results=25&sortBy=submittedDate&sortOrder=descending",
        "https://export.arxiv.org/api/query?search_query=cat:stat.ML&start=0&max_results=25&sortBy=submittedDate&sortOrder=descending",
        "https://export.arxiv.org/api/query?search_query=cat:cs.RO&start=0&max_results=25&sortBy=submittedDate&sortOrder=descending",
        "https://export.arxiv.org/api/query?search_query=all:%22explainable%20ai%22&start=0&max_results=25&sortBy=submittedDate&sortOrder=descending",
        "https://export.arxiv.org/api/query?search_query=all:interpretability&start=0&max_results=25&sortBy=submittedDate&sortOrder=descending",
    ]

    scraper = AINewsScraper(sources)
    articles = scraper.scrape()
    print(f"[main] Fetched {len(articles)} articles")

    db_dir = Path(__file__).resolve().parent / "db"
    archive = VectorArchive(db_dir=db_dir)
    added = archive.add_documents(articles)
    print(f"[main] Added {added} chunks to vector store")

    if added == 0:
        print("[main] No data available for search.")
        return

    while True:
        query = input("Ask about AI news (or 'exit'): ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            break

        results = archive.query(query, n_results=5)
        if not results:
            print("No matches found.")
            continue

        for idx, item in enumerate(results, start=1):
            meta = item.get("metadata", {})
            title = meta.get("title") or "Untitled"
            date = meta.get("date") or "unknown date"
            url = meta.get("url") or ""
            summary = meta.get("summary") or ""
            if summary and len(summary) > 400:
                summary = f"{summary[:400]}..."
            snippet = item.get("text", "")
            max_snippet = 250 if summary else 400
            if len(snippet) > max_snippet:
                snippet = f"{snippet[:max_snippet]}..."

            print(f"\n{idx}. {title} ({date})")
            if url:
                print(f"   {url}")
            if summary:
                print(f"   Summary: {summary}")
            print(f"   {snippet}")
        print("")


if __name__ == "__main__":
    run()
