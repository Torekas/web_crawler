# GPU-Ready AI Knowledge Agent (Crawl + RAG + Chat)

Asynchronous AI crawler tuned for LLM/AI topics with a GPU-first RAG stack and conversational agent that keeps short- and long-term memory.

## GPU Setup
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

## 1) Domain-Aware Crawl (async)
Prioritizes LLM/AI concepts: DSPY, Reflexion, RAG variants (Knowledge-oriented, Tok-RAG, UncertaintyRAG), knowledge injection (SFT, prompt distillation, graphs), judge models (G-Eval, Opik, Prometheus, MT-Bench, Chatbot Arena), hallucination checks (RAGTruth, SelfCheck, NLI/faithfulness), and datasets (GSM8K, ARC, TruthfulQA, HotpotQA, SQuAD). A zero-temperature judge model filters fragments; a verifier loop attempts to clarify unclear content. Reflexion logs self-repair actions (e.g., rotating User-Agent on 403/429).

```bash
py -m src.main crawl --max-pages 120 --depth 5 --concurrency 6 --delay 0.8 --judge-llm ollama --judge-model mixtral:8x7b --output data/pages.jsonl
```
- Seeds default to major AI labs/blogs; stays within those domains and respects robots.txt.
- Use `--judge-llm openai --judge-openai-model gpt-4o-mini` to judge via OpenAI, or `--judge-llm none` to rely on heuristics only.

## 2) Build the Vector Index (GPU embeddings)
```bash
py -m src.main index --pages data/pages.jsonl --index data/index.pkl.gz --model sentence-transformers/all-MiniLM-L6-v2
```
- Text is chunked with overlap and embedded on GPU when available.

## 3) Conversational RAG with Memory
Natural chat that uses retrieval + long-term reflections (from crawler and prior chats) + short-term history. Answers cite sources `[n]` and will propose crawl/verification steps if context is missing.
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

## Agent Behaviors
- Async crawl with politeness delay and concurrency control.
- Domain judge (temperature 0.0) + verification loop for unclear/contradictory fragments.
- Reflexion/self-repair: records failure reasons and retries with alternative strategies.
- DSPY-style prompt separation (`src/prompts.py`) and centralized LLM client (`src/llm.py`).
- Memory:
  - Long-term: `data/memory_longterm.jsonl` stores reflections and chat summaries.
  - Short-term: rolling window of recent turns for conversational grounding.

## Notes
- Data files live under `data/` (git-ignored).
- If the LLM backend is unavailable during chat, the agent falls back to showing top contexts.
