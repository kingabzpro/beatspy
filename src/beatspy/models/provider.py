"""OpenAI-compatible model provider for the OpenAI Agents SDK, plus capability probes.

Any endpoint speaking the Chat Completions API works (Ollama, vLLM, LM Studio,
OpenAI, OpenRouter, ...). Users configure exactly three things: base_url, model
name, and API key.
"""

from __future__ import annotations

import logging

from agents import (
    Agent,
    ModelProvider,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
)
from agents.items import ToolCallItem
from openai import AsyncOpenAI
from openai.types.shared import Reasoning

log = logging.getLogger(__name__)


class BeatSpyModelProvider(ModelProvider):
    def __init__(self, base_url: str, api_key: str | None, request_timeout: float = 180.0, max_retries: int = 3):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
            max_retries=max_retries,
            timeout=request_timeout,
        )
        self._models: dict[str, OpenAIChatCompletionsModel] = {}

    def get_model(self, model_name: str | None) -> OpenAIChatCompletionsModel:
        name = model_name or "default"
        if name not in self._models:
            self._models[name] = OpenAIChatCompletionsModel(model=name, openai_client=self.client)
        return self._models[name]


def run_config_for(
    provider: BeatSpyModelProvider,
    temperature: float,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
) -> RunConfig:
    # New OpenAI models require max_completion_tokens; compatible servers use max_tokens.
    openai_limit = bool(provider and provider.client.base_url.host == "api.openai.com" and max_output_tokens)
    mimo_no_thinking = bool(
        provider and provider.client.base_url.host.endswith("xiaomimimo.com") and reasoning_effort == "none"
    )
    extra_body = {"max_completion_tokens": max_output_tokens} if openai_limit else None
    if mimo_no_thinking:
        extra_body = {"thinking": {"type": "disabled"}}
    return RunConfig(
        model_provider=provider,
        model_settings=ModelSettings(
            temperature=temperature,
            reasoning=Reasoning(effort=reasoning_effort) if reasoning_effort and not mimo_no_thinking else None,
            max_tokens=None if openai_limit else max_output_tokens,
            extra_body=extra_body,
        ),
        tracing_disabled=True,
    )


@function_tool
def _ping_tool() -> str:
    """Return the word pong."""
    return "pong"


async def probe_chat(provider: BeatSpyModelProvider, model_name: str, reasoning_effort=None) -> tuple[bool, str]:
    """Minimal chat completion. Returns (ok, detail)."""
    agent = Agent(name="probe", instructions="Reply with the single word: pong.", model=model_name)
    try:
        result = await Runner.run(
            agent, "ping", max_turns=1, run_config=run_config_for(provider, 0.0, reasoning_effort)
        )
        text = str(result.final_output).strip()
        return True, text[:80] if text else "empty response"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


async def probe_tools(provider: BeatSpyModelProvider, model_name: str, reasoning_effort=None) -> tuple[bool, str]:
    """Check that the model can call tools (BeatSPY's hard requirement)."""
    agent = Agent(
        name="probe-tools",
        instructions="You must call the _ping_tool function once, then reply with the word done.",
        model=model_name,
        tools=[_ping_tool],
    )
    try:
        result = await Runner.run(
            agent, "Call _ping_tool now.", max_turns=3, run_config=run_config_for(provider, 0.0, reasoning_effort)
        )
        saw_call = any(isinstance(item, ToolCallItem) for item in result.new_items)
        if saw_call:
            return True, "tool call observed"
        return False, "model replied without ever calling a tool; BeatSPY requires tool calling"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


async def probe_capabilities(
    provider: BeatSpyModelProvider, model_name: str, reasoning_effort=None, *, require_tools=False
) -> dict[str, tuple[bool, str]]:
    """Run all probes in one event loop (the HTTP client is loop-bound)."""
    probes = {"chat": await probe_chat(provider, model_name, reasoning_effort)}
    if require_tools:
        probes["tools"] = await probe_tools(provider, model_name, reasoning_effort)
    await provider.client.close()
    return probes
