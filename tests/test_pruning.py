from crawl4ai_plus.markdown_cleaner import html_to_markdown


def test_fit_markdown_prunes_noise():
    html = """
    <html><body>
    <div>Subscribe to our newsletter</div>
    <p>Artificial intelligence and machine learning advances.</p>
    </body></html>
    """
    md, text = html_to_markdown(html, keywords=["artificial intelligence"], use_bm25=False)
    assert "Subscribe" not in md
    assert "Artificial intelligence" in md

