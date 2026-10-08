"""NYSE sessions, including holidays and early closes (no incomplete daily bars)."""

from calendar import monthrange
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from ..schemas import Scenario


def completed_session(year: int | None = None, now: datetime | None = None) -> date:
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("now must have a timezone")
    local_day = now.astimezone(ZoneInfo("America/New_York")).date()
    upper = min(local_day, date(year, 12, 31)) if year else local_day
    lower = upper - timedelta(days=14)
    calendar = xcals.get_calendar("XNYS", start=lower, end=upper)
    sessions = calendar.sessions
    sessions = sessions[(sessions >= pd.Timestamp(lower)) & (sessions <= pd.Timestamp(upper))]
    for session in reversed(sessions):
        # Give the data provider time to finish a daily bar after the close.
        if calendar.session_close(session) + pd.Timedelta(minutes=15) <= pd.Timestamp(now):
            if year is not None and session.year != year:
                break
            return session.date()
    raise ValueError(f"no completed session available for {year or local_day.year}")


def subtract_months(day: date, months: int) -> date:
    """Step back whole months, clamping the day to the target month's length."""
    month_index = day.month - 1 - months
    year = day.year + month_index // 12
    month = month_index % 12 + 1
    month_length = monthrange(year, month)[1]
    return date(year, month, min(day.day, month_length))


def resolve_scenario(scenario: Scenario, now: datetime | None = None) -> Scenario:
    if scenario.recent_months is not None:
        # A rolling window: end at the latest completed session, start N months
        # back, so the scenario never drifts into an old period.
        end = completed_session(now=now)
        return scenario.model_copy(
            update={
                "end": end.isoformat(),
                "start": subtract_months(end, scenario.recent_months).isoformat(),
                "through_latest": False,
            },
            deep=True,
        )
    if scenario.through_latest:
        return scenario.model_copy(
            update={"end": completed_session(now=now).isoformat(), "through_latest": False}, deep=True
        )
    if scenario.start and scenario.end:
        return scenario.model_copy(deep=True)
    cutoff = completed_session(scenario.year, now)
    start = max(date(scenario.year, 1, 1), cutoff - timedelta(days=scenario.window_days - 1))
    return scenario.model_copy(update={"start": start.isoformat(), "end": cutoff.isoformat()}, deep=True)
