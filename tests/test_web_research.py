"""Historical evidence gates, bounded live research, and provider contracts."""

import asyncio
import json
from datetime import date

import httpx
import pytest
from agents import RunContextWrapper
from agents.tool_context import ToolContext as SdkToolContext

from beatspy.tools import market, web
from conftest import make_tctx


def invoke(tool, tctx, **args):
    text = json.dumps(args)
    wrapper = SdkToolContext.from_agent_context(
        RunContextWrapper(context=tctx), "call", tool_name=tool.name, tool_arguments=text
    )
    return json.loads(asyncio.run(tool.on_invoke_tool(wrapper, text)))


def page(published=None, modified=None):
    dates = {"@type": "NewsArticle"}
    if published:
        dates["datePublished"] = published
    if modified:
        dates["dateModified"] = modified
    return {
        "result": {
            "html_content": '<script type="application/ld+json">' + json.dumps(dates) + "</script>",
            "markdown_content": "Dated earnings news",
            "page_metadata": {"status_code": 200},
        }
    }


@pytest.mark.parametrize(
    "published,modified",
    [
        (None, None),
        ("2022-02-02", None),
        ("2022-01-30", "2022-02-02"),
        ("unparseable", None),
    ],
)
def test_unknown_or_future_evidence_is_withheld(published, modified):
    output = web.historical_page(page(published, modified), "2022-02-01")
    assert "error" in output and "content" not in output


def test_past_metadata_is_screened_but_not_claimed_as_authenticated():
    output = web.historical_page(page("2022-01-30", "2022-02-01"), "2022-02-01")
    assert output["content"] == "Dated earnings news"
    assert "not authenticated" in output["warning"]


@pytest.mark.parametrize(
    "html",
    [
        '<span class="category-eyebrow__date">January 29, 2026</span>',
        '<time datetime="2026-01-29T12:00:00Z">January 29, 2026</time>',
    ],
)
def test_visible_publisher_dateline_requires_the_same_cutoff(html):
    data = {"result": {"html_content": html, "markdown_content": "earnings", "page_metadata": {"status_code": 200}}}
    assert web.historical_page(data, "2026-02-01")["published_date"] == "2026-01-29"
    assert "content" not in web.historical_page(data, "2026-01-28")


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "https://127.0.0.1",
        "https://169.254.169.254",
        "https://localhost",
        "https://user:secret@example.com",
        "https://example.com:8000",
    ],
)
def test_private_and_credential_urls_are_rejected(url):
    assert not web.public_web_url(url)


def test_search_scrape_budgets_prevent_external_calls(data_service, scenario, monkeypatch):
    tctx = make_tctx(data_service, scenario, as_of=date(2022, 2, 1))
    tctx.olostep_api_key = "test-key"
    tctx.agent = "research"
    calls = []
    monkeypatch.setattr(
        web,
        "olostep_search",
        lambda *a, **k: (
            calls.append("search") or {"results": [{"url": "https://example.com/article"}], "credits_consumed": 1}
        ),
    )
    monkeypatch.setattr(web, "olostep_scrape", lambda *a, **k: calls.append("scrape") or page("2022-01-31"))
    assert "error" in invoke(market.scrape_webpage, tctx, url="https://example.com/not-discovered")
    for _ in range(2):
        assert invoke(market.search_web, tctx, query="earnings")["results"]
    assert "limit reached" in invoke(market.search_web, tctx, query="earnings")["error"]
    for _ in range(3):
        assert invoke(market.scrape_webpage, tctx, url="https://example.com/article")["content"]
    assert "limit reached" in invoke(market.scrape_webpage, tctx, url="https://example.com/article")["error"]
    assert calls == ["search", "search", "scrape", "scrape", "scrape"]
    assert len([e for e in tctx.events if e["type"] == "web_evidence"]) == 5


def test_current_finnhub_ratings_cannot_leak_into_historical_runs(data_service, scenario, monkeypatch):
    tctx = make_tctx(data_service, scenario, as_of=date(2022, 2, 1))
    tctx.finnhub_api_key = "test-key"
    monkeypatch.setattr(web, "finnhub_recommendations", lambda *a, **k: pytest.fail("no historical API call"))
    assert "withheld" in invoke(market.get_analyst_outlook, tctx, ticker="AAPL")["error"]
    names = [tool.name for tool in market.tools_for_role("research", True, protocol_version=5)]
    assert "get_fundamentals" not in names and "scrape_webpage" in names


def test_scrape_uses_real_api_parameters_and_cache(tmp_path):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body == {
            "url_to_scrape": "https://example.com/article",
            "formats": ["html", "markdown"],
            "wait_before_scraping": 2000,
        }
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json=page("2022-01-31"))

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        token = web.CLIENT.set(client)
        try:
            for _ in range(2):
                assert web.olostep_scrape("test-key", "https://example.com/article", tmp_path)["result"]
        finally:
            web.CLIENT.reset(token)
    assert len(requests) == 1
