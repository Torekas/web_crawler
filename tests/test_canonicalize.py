from crawl4ai_plus.url_utils import canonicalize_url


def test_canonicalize_removes_tracking_and_ports():
    url = "HTTP://Example.com:80/path//page?utm_source=newsletter&a=1#section"
    assert canonicalize_url(url) == "http://example.com/path/page?a=1"

