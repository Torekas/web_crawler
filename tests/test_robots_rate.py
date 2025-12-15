import pytest
from urllib import robotparser

from crawl4ai_plus.http_fetcher import PoliteHttpFetcher


@pytest.mark.asyncio
async def test_robots_and_rate_limit_respected(tmp_path):
    fetcher = PoliteHttpFetcher(user_agent="test-agent", cache_dir=tmp_path, obey_robots=True, per_domain_delay=0.1)
    rp = robotparser.RobotFileParser()
    rp.parse(["User-agent: *", "Disallow: /private"])
    fetcher.robot_parsers["example.com"] = rp
    assert await fetcher._allowed_by_robots("http://example.com/open") is True
    assert await fetcher._allowed_by_robots("http://example.com/private/page") is False
    await fetcher.close()

