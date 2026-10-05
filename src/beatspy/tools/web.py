"""Minimal REST clients for optional external providers.

- Olostep: web search (answers) and page scraping.
- Finnhub: company news and basic financials.

Responses are cached on disk inside the scenario data directory so external
data used by a run is frozen alongside prices. Both providers are optional:
bundled scenarios run without them.
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import nullcontext
from contextvars import ContextVar
from pathlib import Path

import httpx

from ..artifacts import atomic_json

log = logging.getLogger(__name__)
CLIENT: ContextVar[httpx.Client | None] = ContextVar("beatspy_http_client", default=None)

OLOSTEP_BASE = "https://api.olostep.com/v1"
FINNHUB_BASE = "https://finnhub.io/api/v1"


def _fetch_json(
    url: str,
    *,
    headers: dict | None = None,
    params: dict | None = None,
    body: dict | None = None,
    cache_dir: Path | None = None,
    timeout: float = 60.0,
) -> dict:
    key = hashlib.sha1(json.dumps([url, params, body], sort_keys=True, default=str).encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"{key}.json" if cache_dir is not None else None
    if cache_file is not None and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    with nullcontext(CLIENT.get()) if CLIENT.get() is not None else httpx.Client(timeout=timeout) as client:
        if body is None:
            response = client.get(url, params=params, headers=headers, timeout=timeout)
        else:
            response = client.post(url, json=body, headers=headers, timeout=timeout)
        if response.is_error:
            # Provider tokens can appear in query strings; never expose request URLs.
            raise RuntimeError(f"provider returned HTTP {response.status_code}")
        data = response.json()

    if cache_file is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(cache_file, data)
    return data


# ---------------------------------------------------------------- Olostep


def olostep_answers(api_key: str, task: str, cache_dir: Path | None = None) -> dict:
    """AI-synthesized web answer with sources. Results may include post-date info; callers must warn the model."""
    data = _fetch_json(
        f"{OLOSTEP_BASE}/answers",
        headers={"Authorization": f"Bearer {api_key}"},
        body={"task": task, "json_format": {"answer": "string", "sources": ["source URL"]}},
        cache_dir=cache_dir,
        timeout=120.0,
    )
    result = data.get("result")
    if isinstance(result, dict) and "json_content" in result:
        content = result["json_content"]
        return json.loads(content) if isinstance(content, str) else content
    return data


# ---------------------------------------------------------------- Finnhub


def finnhub_company_news(
    api_key: str, symbol: str, from_date: str, to_date: str, cache_dir: Path | None = None
) -> list[dict]:
    return _fetch_json(
        f"{FINNHUB_BASE}/company-news",
        params={"symbol": symbol, "from": from_date, "to": to_date, "token": api_key},
        cache_dir=cache_dir,
    )


def finnhub_metrics(api_key: str, symbol: str, cache_dir: Path | None = None) -> dict:
    return _fetch_json(
        f"{FINNHUB_BASE}/stock/metric",
        params={"symbol": symbol, "metric": "all", "token": api_key},
        cache_dir=cache_dir,
    )


FUNDAMENTAL_KEYS = [
    "peTTM",
    "pbTTM",
    "marketCapitalization",
    "dividendYieldIndicatedAnnual",
    "beta",
    "52WeekHigh",
    "52WeekLow",
    "revenueGrowthTTMYoy",
    "epsGrowthTTMYoy",
    "grossMarginTTM",
    "netProfitMarginTTM",
    "roicTTM",
    "currentRatioTTM",
    "totalDebtToEquityAnnual",
]
