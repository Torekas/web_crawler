from crawl4ai_plus.chunker import chunk_text


def test_chunking_overlap_and_min_size():
    text = "A" * 800 + "B" * 800
    chunks = chunk_text(text, chunk_size=600, overlap=100, min_size=200)
    assert len(chunks) == 3
    # ensure overlap keeps continuity
    assert chunks[0]["text"][-5:] == "AAAAA"

