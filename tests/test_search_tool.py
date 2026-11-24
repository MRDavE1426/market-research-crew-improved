"""The scraper tool must work with both providers and degrade gracefully (no network)."""
import pytest
import requests

import config.settings as cfg
from crew.tools import search_tool


class FakeResponse:
    def __init__(self, json_data=None, text="", headers=None):
        self._json, self.text = json_data, text
        self.headers = headers or {"Content-Type": "text/html"}

    def json(self):
        return self._json

    def raise_for_status(self):
        pass


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    for key in ("TAVILY_API_KEY", "SERPER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    cfg.set_output_dir(tmp_path)
    yield
    cfg.set_output_dir(None)


def test_extract_text_strips_scripts_and_whitespace():
    html = "<html><script>var x=1</script><body><h1>Revenue</h1>  <p>$5   bn</p></body></html>"
    assert search_tool.extract_text(html) == "Revenue $5 bn"


def test_tavily_flow_search_and_scrape(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

    def fake_post(url, **kwargs):
        assert url == search_tool.TAVILY_URL
        return FakeResponse({"results": [{"title": "T1", "url": "http://a.test", "content": "snip"}]})

    monkeypatch.setattr(search_tool.requests, "post", fake_post)
    monkeypatch.setattr(
        search_tool.requests, "get", lambda url, **kw: FakeResponse(text="<p>Net income 12bn</p>")
    )
    out = search_tool.web_search_and_scrape.run(query="tech q3")
    assert "http://a.test" in out and "Net income 12bn" in out and "snip" in out


def test_serper_flow(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "serper-test")

    def fake_post(url, **kwargs):
        assert url == search_tool.SERPER_URL
        assert kwargs["headers"]["X-API-KEY"] == "serper-test"
        return FakeResponse({"organic": [{"title": "S1", "link": "http://b.test", "snippet": "sn"}]})

    monkeypatch.setattr(search_tool.requests, "post", fake_post)
    monkeypatch.setattr(search_tool.requests, "get", lambda url, **kw: FakeResponse(text="<p>ok</p>"))
    results = search_tool.search_web("q")
    assert results == [{"title": "S1", "url": "http://b.test", "snippet": "sn"}]


def test_missing_api_keys_returns_clear_error():
    out = search_tool.web_search_and_scrape.run(query="anything")
    assert out.startswith("ERROR:") and "TAVILY_API_KEY" in out


def test_unscrapable_page_falls_back_to_snippet(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr(
        search_tool.requests,
        "post",
        lambda url, **kw: FakeResponse({"results": [{"title": "T", "url": "http://c.test", "content": "only snippet"}]}),
    )

    def boom(url, **kw):
        raise requests.ConnectionError("blocked")

    monkeypatch.setattr(search_tool.requests, "get", boom)
    out = search_tool.web_search_and_scrape.run(query="q")
    assert "could not be scraped" in out and "only snippet" in out


def test_live_research_bundles_six_companies_once_per_run(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setenv("MARKET_RESEARCH_RUN_ID", "test-run")
    calls = []
    monkeypatch.setattr(
        search_tool.requests,
        "post",
        lambda url, **kw: (
            calls.append(kw["json"]["query"])
            or FakeResponse({"results": [{
                "title": "Company Q3 filing", "url": "http://a.test", "content": "Revenue data"
            }]})
        ),
    )
    monkeypatch.setattr(search_tool.requests, "get", lambda url, **kw: FakeResponse(text="<p>Revenue</p>"))
    try:
        first = search_tool.web_search_and_scrape.run(query="Tech Sector Q3 Analysis")
        second = search_tool.web_search_and_scrape.run(query="Tech Sector Q3 Analysis")
    finally:
        search_tool.clear_research_run("test-run")
    assert "Microsoft" in first and "NVIDIA" in first
    assert "Company Q3 filing" in first
    assert "six-company live search has already completed" in second
    assert len(calls) == 6
