@echo off
echo Starting Crawl...
py -m src.main crawl --max-pages 120 --depth 5 --concurrency 6 --delay 0.8 --judge-llm ollama --judge-model mixtral:8x7b --output data/pages.jsonl

IF %ERRORLEVEL% NEQ 0 (
    echo Crawl failed. Exiting.
    exit /b %ERRORLEVEL%
)

echo Starting Indexing...
py -m src.main index --pages data/pages.jsonl --index data/index.pkl.gz --model sentence-transformers/all-MiniLM-L6-v2

echo Pipeline complete.
pause