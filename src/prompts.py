FOCUS_TOPICS = [
    "DSPY (LLM programming, prompt optimization)",
    "Reflexion (self-repair / self-reflection loops for agents)",
    "RAG frameworks: Standard RAG, Knowledge-oriented RAG, Tok-RAG, UncertaintyRAG",
    "Knowledge injection: SFT, prompt distillation, knowledge graphs (e.g., Dragan, Donadello)",
    "LLM-as-a-judge: G-Eval, Opik, Prometheus, MT-Bench, Chatbot Arena",
    "Hallucination detection: RAGTruth, SelfCheck, NLI-based verification, faithfulness scoring",
    "Datasets: GSM8K, ARC, TruthfulQA, HotpotQA, SQuAD",
    "Safety/toxicity eval: AdvBench, RealToxicityPrompts, ToxiGen, civilcomments",
    "Agent patterns: ReAct, multi-agent orchestration, verifier-solver loops",
    "Vector stores and reranking: FAISS, Weaviate, Milvus, cross-encoders",
    "Efficiency/serving: KV caching, pruning, quantization for LLM/RAG pipelines",
    "Retriever optimization: query rewriting, hybrid/bm25+dense search, late interaction, ColBERT",
    "Evaluation ops: preference data, win-rate, disagreement rates, judge consistency",
    "Safety & alignment: Constitutional AI, red-teaming, refusal tuning, jailbreak defenses",
    "Deployment: batching, streaming, speculative decoding, vLLM/llama.cpp/gguf optimizations",
    "Memory & persistence: episodic vs semantic memory, vector DB management, TTL/refresh",
]

FOCUS_TOPICS_TEXT = "\n".join(f"- {topic}" for topic in FOCUS_TOPICS)


def build_judge_prompt(snippet: str) -> str:
    return (
        "You are a zero-temperature judge model that filters web content for an AI/RAG knowledge base.\n"
        "Decide whether the snippet materially discusses the focus topics.\n"
        f"Focus topics:\n{FOCUS_TOPICS_TEXT}\n\n"
        f"Snippet:\n{snippet}\n\n"
        "Respond with one of: KEEP, SKIP, UNSURE. KEEP only if the content is informative and non-trivial."
    )


def build_verification_prompt(snippet: str) -> str:
    return (
        "You are a verifier that checks if a text fragment is coherent, factual, and self-consistent.\n"
        "If it is unclear or contradictory, propose a concise correction/clarification grounded in the text.\n"
        "Respond with JSON: {\"status\": \"ok\"|\"unclear\", \"note\": \"short rationale or corrected fact\"}.\n\n"
        f"Fragment:\n{snippet}"
    )


def build_answer_system_prompt() -> str:
    return (
        "You are an autonomous RAG agent focused on advanced LLM/AI engineering. "
        "You maintain short-term conversation memory and long-term reflections. "
        "Use chain-of-thought internally but never reveal it. Provide a concise multi-sentence (2-4) synthesis that cites sources with [number]; avoid single-sentence taglines. "
        "After the answer, include a compact 'Sources:' list with each [number] followed by title and URL so the user can click through. "
        "Prefer fresher snippets (by fetched_at) and ignore or down-rank unreachable links. "
        "If context is missing or weak, say so and propose what to crawl or verify next; do not fabricate. "
        "Do not invent facts outside the provided context."
    )
