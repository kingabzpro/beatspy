"""Per-decision tool state: budgets, event log, and integrity counters.

One ToolContext exists per decision date and is passed to every function tool
via the Agents SDK RunContextWrapper. The agent that is currently running is
set on `agent` by the executor, so budget accounting always lands on the right
agent regardless of what the model claims.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from ..data.service import DataService
from ..schemas import EventItem, Scenario


@dataclass
class ToolContext:
    as_of: date
    data: DataService
    scenario: Scenario
    events_feed: list[EventItem] = field(default_factory=list)
    forecast_fn: Callable[[list[float], int], object] | None = None
    finnhub_api_key: str | None = None
    olostep_api_key: str | None = None
    tool_budget_per_agent: int = 10
    agent: str = "harness"
    events: list[dict] = field(default_factory=list)
    used: dict[str, int] = field(default_factory=dict)
    violations: list[dict] = field(default_factory=list)
    external_limit: asyncio.Semaphore | None = None
    http_client: object = None
    horizon_days: int = 20
    next_decision_date: str | None = None
    web_used: dict[str, int] = field(default_factory=dict)
    web_urls: set[str] = field(default_factory=set)

    def spend_web(self, tool: str, limit: int) -> str | None:
        used = self.web_used.get(tool, 0)
        if used >= limit:
            return json.dumps({"error": f"{tool} limit reached ({limit} per decision). Stop calling this tool."})
        self.web_used[tool] = used + 1
        return None

    def spend(self, agent: str, tool: str, args: dict | None = None) -> str | None:
        """Charge one tool call to the agent's budget.

        Returns an error payload string when the budget is exhausted; tools
        return that string verbatim so the model learns to stop calling.
        """
        used = self.used.get(agent, 0)
        if used >= self.tool_budget_per_agent:
            self.events.append(
                {"type": "budget_exhausted", "agent": agent, "tool": tool, "date": self.as_of.isoformat()}
            )
            return json.dumps(
                {
                    "error": (
                        f"tool budget exhausted for {agent} "
                        f"({self.tool_budget_per_agent} calls). Stop calling tools and "
                        "answer from the data you already have."
                    )
                }
            )
        self.used[agent] = used + 1
        self.events.append(
            {"type": "tool_call", "agent": agent, "tool": tool, "args": args or {}, "date": self.as_of.isoformat()}
        )
        return None

    def record_violation(self, kind: str, detail: str) -> None:
        entry = {"kind": kind, "detail": detail, "date": self.as_of.isoformat()}
        self.violations.append(entry)
        self.events.append({"type": "violation", **entry})

    def tool_calls_total(self) -> int:
        return sum(self.used.values())
