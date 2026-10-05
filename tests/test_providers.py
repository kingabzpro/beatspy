"""Live-provider contracts without external requests or credentials."""

import json

import httpx
import pytest

from beatspy.config import Settings, provider_api_key, secret_values
from beatspy.tools import web


def test_olostep_search_contract_and_provider_errors(monkeypatch):
    def respond(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/scrapes"
        assert body["parser"] == {"id": "@olostep/google-search"}
        assert "before%3A2022-02-02" in body["url_to_scrape"]
        return httpx.Response(
            200,
            json={
                "result": {
                    "json_content": json.dumps({"organic": [{"link": "https://example.org", "snippet": "future fact"}]})
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        token = web.CLIENT.set(client)
        try:
            result = web.olostep_search("test-key", "Historical research", "2022-02-01")
            assert result["results"] == [{"url": "https://example.org", "position": None}]
        finally:
            web.CLIENT.reset(token)
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(403))) as client:
        token = web.CLIENT.set(client)
        try:
            with pytest.raises(RuntimeError, match="^provider returned HTTP 403$"):
                web.finnhub_metrics("do-not-expose", "AAPL")
        finally:
            web.CLIENT.reset(token)


def test_existing_provider_keys_activate_and_are_redacted(monkeypatch):
    for provider in ("finnhub", "olostep"):
        monkeypatch.delenv(f"BEATSPY_{provider.upper()}_API_KEY", raising=False)
        monkeypatch.setenv(f"{provider.upper()}_API_KEY", f"private-{provider}-key")
        assert provider_api_key(Settings(), provider) == f"private-{provider}-key"
        assert f"private-{provider}-key" in secret_values()
