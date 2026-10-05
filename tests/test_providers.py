"""Live-provider contracts without external requests or credentials."""

import json

import httpx
import pytest

from beatspy.config import Settings, provider_api_key, secret_values
from beatspy.tools import web


def test_olostep_structured_answer_and_provider_errors(monkeypatch):
    def respond(request):
        assert json.loads(request.content)["json_format"]["answer"] == "string"
        return httpx.Response(
            200,
            json={
                "result": {
                    "json_content": json.dumps({"answer": "Historical evidence", "sources": ["https://example.org"]})
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        token = web.CLIENT.set(client)
        try:
            assert web.olostep_answers("test-key", "Historical research")["answer"] == "Historical evidence"
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
