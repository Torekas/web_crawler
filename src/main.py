import argparse
import logging
from pathlib import Path

from . import crawler
from . import rag


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI content crawler with RAG index")
    sub = parser.add_subparsers(dest="command", required=True)

    crawl_p = sub.add_parser("crawl", help="Crawl AI-related pages and save to JSONL")
    crawl_p.add_argument("--seeds", nargs="+", default=crawler.DEFAULT_SEEDS, help="Seed URLs to start crawling")
    crawl_p.add_argument("--max-pages", type=int, default=80, help="Maximum number of pages to capture")
    crawl_p.add_argument("--depth", type=int, default=2, help="Maximum crawl depth from seeds")
    crawl_p.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    crawl_p.add_argument("--output", type=Path, default=Path("data/pages.jsonl"), help="Where to store captured pages")

    index_p = sub.add_parser("index", help="Build vector index from crawled pages")
    index_p.add_argument("--pages", type=Path, default=Path("data/pages.jsonl"), help="Input JSONL from crawler")
    index_p.add_argument("--index", type=Path, default=Path("data/index.pkl.gz"), help="Where to write the vector index")
    index_p.add_argument("--model", default=rag.DEFAULT_MODEL, help="SentenceTransformer model name")
    index_p.add_argument("--chunk-size", type=int, default=800, help="Words per chunk")
    index_p.add_argument("--overlap", type=int, default=150, help="Word overlap between chunks")
    index_p.add_argument("--min-words", type=int, default=60, help="Minimum words to keep a chunk")

    chat_p = sub.add_parser("chat", help="Interactive retrieval chat")
    chat_p.add_argument("--index", type=Path, default=Path("data/index.pkl.gz"), help="Vector index path")
    chat_p.add_argument("--model", default=None, help="Override embedding model (defaults to index model)")
    chat_p.add_argument("--top-k", type=int, default=4, help="How many chunks to retrieve per query")
    chat_p.add_argument(
        "--llm", choices=["ollama", "openai"], default="ollama", help="LLM backend for answers (default: ollama)"
    )
    chat_p.add_argument("--ollama-model", default="mixtral:8x7b", help="Ollama model name to use")
    chat_p.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model name to use")

    search_p = sub.add_parser("search", help="Single query without interactive chat")
    search_p.add_argument("--index", type=Path, default=Path("data/index.pkl.gz"))
    search_p.add_argument("--query", required=True, help="Query to search for")
    search_p.add_argument("--top-k", type=int, default=5)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "crawl":
        crawler.crawl(
            seeds=args.seeds,
            max_pages=args.max_pages,
            max_depth=args.depth,
            delay_seconds=args.delay,
            output_path=args.output,
        )
    elif args.command == "index":
        rag.build_index(
            pages_path=args.pages,
            index_path=args.index,
            model_name=args.model,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            min_words=args.min_words,
        )
    elif args.command == "chat":
        rag.interactive_chat(
            index_path=args.index,
            model_name=args.model,
            top_k=args.top_k,
            llm=args.llm,
            ollama_model=args.ollama_model,
            openai_model=args.openai_model,
        )
    elif args.command == "search":
        results = rag.search(args.index, args.query, top_k=args.top_k)
        for idx, (score, chunk) in enumerate(results, start=1):
            print(f"[{idx}] score={score:.3f} {chunk.title} - {chunk.url}")
            print(chunk.text[:700] + ("..." if len(chunk.text) > 700 else ""))
            print()


if __name__ == "__main__":
    main()
