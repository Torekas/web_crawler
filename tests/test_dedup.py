from crawl4ai_plus.dedup import Deduper, hamming_distance, simhash


def test_deduper_detects_near_duplicate():
    deduper = Deduper(simhash_threshold=4)
    first = "Artificial intelligence systems improve with retrieval augmented generation."
    second = "Artificial intelligence systems improve with retrieval-augmented generation."
    assert deduper.seen(first) is False
    assert deduper.seen(second) is True


def test_simhash_distance_small_for_similar_text():
    a = simhash("LLM agents use vector database context.")
    b = simhash("LLM agents use vector database context and reranking.")
    assert hamming_distance(a, b) < 10

