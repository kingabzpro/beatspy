"""Function tools exposed to the agents.

Every tool: charges the calling agent's budget first, clamps all data to
ToolContext.as_of (the model can never request future data), and returns JSON
strings. Tool failures return JSON error payloads instead of raising, so a bad
model turn never crashes a benchmark run.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, date, datetime

from agents import RunContextWrapper, function_tool

from . import web
from .context import ToolContext

log = logging.getLogger(__name__)


async def _external(tctx, fn, *args, **kwargs):
    limit = tctx.external_limit or asyncio.Semaphore(6)
    async with limit:
        token = web.CLIENT.set(tctx.http_client)
        work = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            # A worker thread cannot be interrupted: retain its slot and client
            # until the bounded request finishes, even when its agent is cancelled.
            await asyncio.gather(work, return_exceptions=True)
            raise
        finally:
            web.CLIENT.reset(token)


def _out(payload: object) -> str:
    return json.dumps(payload, default=str)


def _guard(ctx: RunContextWrapper[ToolContext], tool: str, **args) -> str | None:
    tctx = ctx.context
    return tctx.spend(tctx.agent, tool, args)


def _as_of(tctx: ToolContext) -> str:
    return tctx.as_of.isoformat()


@function_tool
def get_price_history(ctx: RunContextWrapper[ToolContext], ticker: str, days: int = 90) -> str:
    """Daily open, close, and volume history for a ticker, oldest first, ending at the decision date. days is capped at 250."""
    err = _guard(ctx, "get_price_history", ticker=ticker, days=days)
    if err:
        return err
    tctx = ctx.context
    ticker = ticker.upper()
    if not tctx.data.has_ticker(ticker):
        return _out({"error": f"unknown ticker {ticker}", "available": tctx.scenario.universe})
    days = max(5, min(int(days), 250))
    frame = tctx.data.window(ticker, tctx.as_of, days)
    rows = [
        {
            "date": ts.date().isoformat(),
            "open": round(float(o), 2),
            "close": round(float(c), 2),
            "volume": int(v) if v == v else 0,
        }
        for ts, o, c, v in zip(frame.index, frame["open"], frame["close"], frame["volume"], strict=True)
    ]
    return _out({"ticker": ticker, "as_of": _as_of(tctx), "rows": rows})


@function_tool
def get_technical_indicators(ctx: RunContextWrapper[ToolContext], ticker: str) -> str:
    """Technical snapshot for a ticker as of the decision date: SMAs, RSI, returns, volatility, 52-week position."""
    err = _guard(ctx, "get_technical_indicators", ticker=ticker)
    if err:
        return err
    tctx = ctx.context
    ticker = ticker.upper()
    if not tctx.data.has_ticker(ticker):
        return _out({"error": f"unknown ticker {ticker}", "available": tctx.scenario.universe})
    indicators = tctx.data.indicators(ticker, tctx.as_of)
    if indicators is None:
        return _out({"error": f"not enough history for {ticker}"})
    return _out({"ticker": ticker, "as_of": _as_of(tctx), **indicators})


@function_tool
def get_market_events(ctx: RunContextWrapper[ToolContext], days_back: int = 14) -> str:
    """Curated major market events from the last days_back days (capped at 60), ending at the decision date."""
    err = _guard(ctx, "get_market_events", days_back=days_back)
    if err:
        return err
    tctx = ctx.context
    days_back = max(1, min(int(days_back), 60))
    start = date.fromordinal(tctx.as_of.toordinal() - days_back)
    events = [
        {"date": e.date, "headline": e.headline, "detail": e.detail, "source": e.source}
        for e in tctx.events_feed
        if start.isoformat() <= e.date <= _as_of(tctx)
    ]
    return _out(
        {"as_of": _as_of(tctx), "note": "curated feed of major public events; not exhaustive", "events": events}
    )


@function_tool
async def get_company_news(ctx: RunContextWrapper[ToolContext], ticker: str, days_back: int = 14) -> str:
    """Company news headlines for a ticker over the last days_back days (requires the optional Finnhub integration)."""
    err = _guard(ctx, "get_company_news", ticker=ticker, days_back=days_back)
    if err:
        return err
    tctx = ctx.context
    if not tctx.finnhub_api_key:
        return _out(
            {
                "error": "news feed not configured for this run",
                "hint": "use get_market_events for curated market events instead",
            }
        )
    ticker = ticker.upper()
    days_back = max(1, min(int(days_back), 30))
    start = date.fromordinal(tctx.as_of.toordinal() - days_back)
    try:
        items = await _external(
            tctx,
            web.finnhub_company_news,
            tctx.finnhub_api_key,
            ticker,
            start.isoformat(),
            _as_of(tctx),
            cache_dir=_cache_dir(tctx) / "news",
        )
    except Exception as exc:
        return _out({"error": f"news provider failed: {exc}"})
    news = [
        {
            "date": datetime_utc_iso(item.get("datetime")),
            "headline": item.get("headline", ""),
            "summary": (item.get("summary") or "")[:300],
            "source": item.get("source", ""),
        }
        for item in items
    ]
    news = [n for n in news if n["date"] <= _as_of(tctx)]
    news.sort(key=lambda n: n["date"], reverse=True)  # newest first, provider-order independent
    news = news[:10]
    return _out({"ticker": ticker, "as_of": _as_of(tctx), "news": news})


@function_tool
async def get_fundamentals(ctx: RunContextWrapper[ToolContext], ticker: str) -> str:
    """Basic company fundamentals (valuation, margins, growth) as of the latest filing (requires the optional Finnhub integration)."""
    err = _guard(ctx, "get_fundamentals", ticker=ticker)
    if err:
        return err
    tctx = ctx.context
    if not tctx.finnhub_api_key:
        return _out({"error": "fundamentals provider not configured for this run"})
    ticker = ticker.upper()
    try:
        data = await _external(
            tctx, web.finnhub_metrics, tctx.finnhub_api_key, ticker, cache_dir=_cache_dir(tctx) / "news" / _as_of(tctx)
        )
    except Exception as exc:
        return _out({"error": f"fundamentals provider failed: {exc}"})
    metric = data.get("metric") or {}
    selected = {k: metric.get(k) for k in web.FUNDAMENTAL_KEYS if metric.get(k) is not None}
    # The provider serves TTM metrics as of its current date with no as-of
    # parameter, so past-dated scenarios must treat this as flagged background.
    tctx.record_violation("future_data_risk", f"get_fundamentals for {ticker} returns provider-current TTM data")
    return _out(
        {
            "ticker": ticker,
            "as_of": _as_of(tctx),
            "warning": (
                "these fundamentals reflect the provider's CURRENT date, not the "
                f"decision date {_as_of(tctx)}; using them for a historical decision "
                "is look-ahead bias, so treat them as background context only"
            ),
            "fundamentals": selected,
        }
    )


@function_tool
async def forecast_price(ctx: RunContextWrapper[ToolContext], ticker: str, horizon_days: int = 20) -> str:
    """Statistical/quantitative price forecast for a ticker over horizon_days (capped 5-60), using only data up to the decision date. Call once per ticker."""
    err = _guard(ctx, "forecast_price", ticker=ticker, horizon_days=horizon_days)
    if err:
        return err
    tctx = ctx.context
    if tctx.forecast_fn is None:
        return _out({"error": "no forecast provider configured"})
    ticker = ticker.upper()
    if not tctx.data.has_ticker(ticker):
        return _out({"error": f"unknown ticker {ticker}", "available": tctx.scenario.universe})
    horizon_days = max(5, min(int(horizon_days), 60))
    closes = tctx.data.closes(ticker, tctx.as_of, 260)
    try:
        result = await _external(tctx, tctx.forecast_fn, ticker, closes, horizon_days)
    except Exception as exc:
        return _out({"error": f"forecast provider failed: {exc}"})
    if result is None:
        return _out({"error": f"not enough history to forecast {ticker}"})
    return _out(result.as_dict())


@function_tool
def verify_price(ctx: RunContextWrapper[ToolContext], ticker: str, on_date: str) -> str:
    """Closing price of a ticker on (or the last trading day before) a given date. Use it to fact-check price claims."""
    err = _guard(ctx, "verify_price", ticker=ticker, on_date=on_date)
    if err:
        return err
    tctx = ctx.context
    ticker = ticker.upper()
    try:
        requested = date.fromisoformat(on_date)
    except ValueError:
        return _out({"error": f"invalid date {on_date!r}; use YYYY-MM-DD"})
    effective = min(requested, tctx.as_of)
    found = tctx.data.last_close(ticker, effective)
    if found is None:
        return _out({"error": f"no price data for {ticker}"})
    verified_date, close = found
    out = {
        "ticker": ticker,
        "requested_date": requested.isoformat(),
        "verified_date": verified_date.isoformat(),
        "close": round(close, 2),
    }
    if requested > tctx.as_of:
        out["note"] = f"requested date is after the decision date; clamped to {_as_of(tctx)}"
        tctx.record_violation("future_date_request", f"verify_price for {ticker} on {requested.isoformat()}")
    return _out(out)


@function_tool
async def search_web(ctx: RunContextWrapper[ToolContext], query: str) -> str:
    """Web search via Olostep (only available when enabled for this scenario). Returns an answer with sources."""
    err = _guard(ctx, "search_web", query=query)
    if err:
        return err
    tctx = ctx.context
    if not tctx.olostep_api_key:
        return _out({"error": "web search is disabled for this benchmark run"})
    try:
        # Cache key includes the decision date: the same query at a later
        # decision date must not serve an earlier date's frozen answer.
        cache_dir = _cache_dir(tctx) / "web" / _as_of(tctx)
        data = await _external(tctx, web.olostep_answers, tctx.olostep_api_key, query, cache_dir=cache_dir)
    except Exception as exc:
        return _out({"error": f"web search failed: {exc}"})
    return _out(
        {
            "query": query,
            "answer": data.get("result") or data.get("answer") or "",
            "sources": data.get("sources") or [],
            "warning": f"discard anything published after {_as_of(tctx)}; using later information is look-ahead bias",
        }
    )


def _cache_dir(tctx: ToolContext):
    from ..data.freeze import scenario_data_dir

    return scenario_data_dir(tctx.scenario)


def datetime_utc_iso(epoch_seconds: object) -> str:
    try:
        return datetime.fromtimestamp(int(epoch_seconds), tz=UTC).date().isoformat()
    except (TypeError, ValueError):
        return ""


def tools_for_role(role: str, web_enabled: bool) -> list:
    research = [get_market_events, get_company_news, get_fundamentals]
    if web_enabled:
        research.append(search_web)
    mapping = {
        "research": research,
        "analyst": [get_price_history, get_technical_indicators],
        "forecaster": [forecast_price, get_technical_indicators],
        "critic": [verify_price],
        "portfolio_manager": [get_technical_indicators],
    }
    return mapping[role]
