"""Decision pipeline: five Agents SDK agents orchestrated deterministically.

Per decision date: research, analyst, and forecaster run in parallel; the critic
reviews their artifacts; the portfolio manager produces the final allocation.
The workflow is fixed Python (a benchmark protocol, not a conversation), while
the Agents SDK owns each agent run: model calls, tools, retries, and usage.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Protocol

from agents import Agent, RunConfig, Runner, set_tracing_disabled
from agents.items import ToolCallItem

from ..models.provider import BeatSpyModelProvider, run_config_for
from ..schemas import extract_json, parse_artifact, validate_decision
from ..tools.context import ToolContext
from ..tools.market import tools_for_role
from . import prompts

log = logging.getLogger(__name__)

ROLES = ["research", "analyst", "forecaster", "critic", "portfolio_manager"]
TRIO = ["research", "analyst", "forecaster"]


@dataclass
class AgentOutcome:
    role: str
    final_output: str = ""
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    tool_calls: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    response_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "final_output": self.final_output,
            "error": self.error,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "requests": self.requests,
            "tool_calls": self.tool_calls,
            "duration_s": round(self.duration_s, 3),
            "response_ids": self.response_ids,
        }


class AgentExecutor(Protocol):
    """Boundary for testing: tests inject scripted executors instead of real models."""

    async def run(self, agent: Agent, input_text: str, tctx: ToolContext) -> AgentOutcome: ...


class SdkExecutor:
    """Runs one agent through the OpenAI Agents SDK."""

    def __init__(self, run_config: RunConfig, max_turns: int = 8):
        self.run_config = run_config
        self.max_turns = max_turns

    async def run(self, agent: Agent, input_text: str, tctx: ToolContext) -> AgentOutcome:
        # Per-run view: agent is fixed per coroutine while used/events/violations
        # stay shared, so concurrent agents never charge each other's budgets.
        run_ctx = replace(tctx, agent=agent.name)
        started = time.monotonic()
        try:
            result = await Runner.run(
                agent,
                input_text,
                context=run_ctx,
                max_turns=self.max_turns,
                run_config=self.run_config,
            )
        except Exception as exc:
            log.warning("agent %s failed: %s", agent.name, exc)
            return AgentOutcome(
                role=agent.name, error=f"{type(exc).__name__}: {exc}", duration_s=time.monotonic() - started
            )
        usage = result.context_wrapper.usage
        calls = []
        for item in result.new_items:
            if isinstance(item, ToolCallItem):
                name = getattr(item.raw_item, "name", None)
                if name:
                    calls.append(name)
        return AgentOutcome(
            role=agent.name,
            final_output=str(result.final_output),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            requests=usage.requests,
            tool_calls=calls,
            response_ids=[r.response_id for r in result.raw_responses if r.response_id],
            duration_s=time.monotonic() - started,
        )


@dataclass
class DecisionRecord:
    date: date
    artifacts: dict = field(default_factory=dict)
    outcomes: dict[str, AgentOutcome] = field(default_factory=dict)
    validated: object = None  # ValidatedDecision
    parse_errors: list[str] = field(default_factory=list)
    raw_decision: dict | None = None

    @property
    def predicted_direction(self) -> str:
        value = (self.raw_decision or {}).get("expected_direction", "flat")
        value = str(value).lower().strip()
        return value if value in ("up", "down", "flat") else "flat"

    @property
    def rationale(self) -> str:
        return str((self.raw_decision or {}).get("rationale", ""))[:400]

    def usage_by_role(self) -> dict[str, dict]:
        return {
            role: {
                "input_tokens": o.input_tokens,
                "output_tokens": o.output_tokens,
                "requests": o.requests,
                "tool_calls": len(o.tool_calls),
            }
            for role, o in self.outcomes.items()
        }

    def totals(self) -> dict[str, int]:
        return {
            "input_tokens": sum(o.input_tokens for o in self.outcomes.values()),
            "output_tokens": sum(o.output_tokens for o in self.outcomes.values()),
            "requests": sum(o.requests for o in self.outcomes.values()),
            "tool_calls": sum(len(o.tool_calls) for o in self.outcomes.values()),
        }


class DecisionPipeline:
    def __init__(
        self,
        settings,
        scenario,
        provider: BeatSpyModelProvider,
        executor: AgentExecutor | None = None,
        web_tools: bool = False,
        agent_limit: asyncio.Semaphore | None = None,
    ):
        set_tracing_disabled(True)
        self.settings = settings
        self.scenario = scenario
        self.agent_limit = agent_limit or asyncio.Semaphore(6)
        self.executor = executor or SdkExecutor(
            run_config=run_config_for(provider, settings.model.temperature, settings.model.reasoning_effort),
            max_turns=settings.model.max_turns,
        )
        self.agents: dict[str, Agent] = {}
        for role in ROLES:
            model_name = settings.model.model
            self.agents[role] = Agent(
                name=role,
                instructions=prompts.INSTRUCTIONS[role],
                model=model_name,
                tools=tools_for_role(role, web_enabled=web_tools),
            )

    async def _run(self, role, input_text, tctx):
        async with self.agent_limit:
            return await self.executor.run(self.agents[role], input_text, tctx)

    async def decide(
        self,
        tctx: ToolContext,
        holdings: dict[str, float],
        cash: float,
        recent_decision: str | None,
    ) -> DecisionRecord:
        brief = prompts.market_brief(
            tctx.data.universe_summary(tctx.scenario.universe, tctx.as_of),
            tctx.as_of.isoformat(),
            tctx.scenario.name,
            tctx.scenario.benchmark,
            holdings,
            cash,
            tctx.scenario.max_position_weight,
            recent_decision,
        )
        record = DecisionRecord(date=tctx.as_of)

        tasks = [asyncio.create_task(self._run(role, getattr(prompts, f"{role}_input")(brief), tctx)) for role in TRIO]
        try:
            trio_outcomes = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        for outcome in trio_outcomes:
            record.outcomes[outcome.role] = outcome

        artifacts = self._parse_artifacts(record)
        record.artifacts = artifacts

        critic = await self._run("critic", prompts.critic_input(brief, artifacts), tctx)
        record.outcomes["critic"] = critic
        critique_json, critique_err = parse_artifact("critic", critic.final_output)
        critique_obj = critique_json.model_dump() if critique_json else critic.final_output[:4000]
        if critique_err:
            record.parse_errors.append(f"critic: {critique_err}")

        pm = await self._run("portfolio_manager", prompts.pm_input(brief, artifacts, critique_obj), tctx)
        record.outcomes["portfolio_manager"] = pm

        raw = None
        if pm.error:
            record.parse_errors.append(f"portfolio_manager: agent error: {pm.error}")
        else:
            raw = extract_json(pm.final_output)
            if raw is None:
                record.parse_errors.append("portfolio_manager: no JSON object found in output")
        record.raw_decision = raw if isinstance(raw, dict) else None
        record.validated = validate_decision(raw, self.scenario.universe, self.scenario.max_position_weight)
        if record.validated.invalid:
            record.parse_errors.append("portfolio_manager: decision invalid; holding previous portfolio")
        return record

    def _parse_artifacts(self, record: DecisionRecord) -> dict[str, object]:
        artifacts: dict[str, object] = {}
        for role in TRIO:
            outcome = record.outcomes.get(role)
            if outcome is None or outcome.error:
                if outcome and outcome.error:
                    record.parse_errors.append(f"{role}: agent error: {outcome.error}")
                artifacts[role] = "{}"
                continue
            parsed, err = parse_artifact(role, outcome.final_output)
            if parsed is not None:
                artifacts[role] = parsed.model_dump()
            else:
                record.parse_errors.append(f"{role}: {err}")
                artifacts[role] = outcome.final_output[:4000]
        return artifacts
