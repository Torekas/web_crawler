from crawl4ai_plus.config import load_config


def test_env_override(monkeypatch, tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("crawler:\n  max_pages: 10\n", encoding="utf-8")
    monkeypatch.setenv("CRAWL_CRAWLER__MAX_PAGES", "5")
    config = load_config(cfg_file)
    assert config.crawler.max_pages == 5

