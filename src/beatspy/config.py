"""Configuration, credentials, and environment handling.

Precedence: CLI flags > BEATSPY_* environment variables > ~/.beatspy/config.toml > defaults.
API keys live in ~/.beatspy/secrets.env (never inside a repository) or in the environment.
"""

from __future__ import annotations

import contextlib
import json
import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MODEL_PRESETS: dict[str, str] = {
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "lmstudio": "http://localhost:1234/v1",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "custom": "",
}

DEFAULT_MODEL_API_KEY_ENV = "BEATSPY_API_KEY"


class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_turns: int = 8
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] | None = None
    cost_per_m_input: float | None = None
    cost_per_m_output: float | None = None


class ToolsConfig(BaseModel):
    forecast_provider: str = "baseline"


class BenchConfig(BaseModel):
    initial_capital: float = 100_000.0
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    max_position_weight: float = 0.35
    tool_budget_per_agent: int = 10
    risk_free_annual: float = 0.0


class Settings(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    bench: BenchConfig = Field(default_factory=BenchConfig)


def beatspy_home() -> Path:
    return Path(os.environ.get("BEATSPY_HOME", str(Path.home() / ".beatspy")))


def config_path() -> Path:
    return beatspy_home() / "config.toml"


def secrets_path() -> Path:
    return beatspy_home() / "secrets.env"


def data_dir() -> Path:
    return Path.cwd() / ".beatspy-data"


def results_dir() -> Path:
    return Path.cwd() / "results"


def load_secrets() -> None:
    """Load secrets.env into the environment (never overriding existing vars)."""
    for key, value in read_secrets().items():
        os.environ.setdefault(key, value)


def read_secrets() -> dict[str, str]:
    path = secrets_path()
    if not path.exists():
        return {}
    secrets: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        secrets[key.strip()] = value.strip().strip('"').strip("'")
    return secrets


def secret_values() -> set[str]:
    values = set(read_secrets().values()) | {
        value for key, value in os.environ.items() if key.startswith("BEATSPY_") and "KEY" in key
    }
    if value := os.environ.get(os.environ.get("BEATSPY_API_KEY_ENV", DEFAULT_MODEL_API_KEY_ENV)):
        values.add(value)
    return values


def load_settings(path: Path | None = None) -> Settings:
    load_secrets()
    path = path or config_path()
    data: dict = {}
    if path.exists():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    settings = Settings.model_validate(data)

    m = settings.model
    if (v := os.environ.get("BEATSPY_BASE_URL")) is not None:
        m.base_url = v
    if (v := os.environ.get("BEATSPY_MODEL")) is not None:
        m.model = v
    if (v := os.environ.get("BEATSPY_TEMPERATURE")) is not None:
        m.temperature = float(v)
    if (v := os.environ.get("BEATSPY_MAX_TURNS")) is not None:
        m.max_turns = int(v)
    if (v := os.environ.get("BEATSPY_REASONING_EFFORT")) is not None:
        m.reasoning_effort = v
        settings = Settings.model_validate(settings.model_dump())
    if (v := os.environ.get("BEATSPY_FORECAST_PROVIDER")) is not None:
        settings.tools.forecast_provider = v
    return settings


def model_api_key(settings: Settings) -> str | None:
    env_name = os.environ.get("BEATSPY_API_KEY_ENV", DEFAULT_MODEL_API_KEY_ENV)
    value = os.environ.get(env_name)
    return value or None


def provider_api_key(settings: Settings, which: str) -> str | None:
    env_names = {
        "olostep": "BEATSPY_OLOSTEP_API_KEY",
        "finnhub": "BEATSPY_FINNHUB_API_KEY",
        "timegpt": "BEATSPY_TIMEGPT_API_KEY",
    }
    value = os.environ.get(env_names[which])
    return value or None


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value))


def render_config(settings: Settings) -> str:
    lines: list[str] = []
    m = settings.model
    lines += [
        "[model]",
        f"base_url = {_toml_value(m.base_url)}",
        f"model = {_toml_value(m.model)}",
        f"temperature = {_toml_value(m.temperature)}",
        f"max_turns = {_toml_value(m.max_turns)}",
    ]
    if m.cost_per_m_input is not None:
        lines.append(f"cost_per_m_input = {_toml_value(m.cost_per_m_input)}")
    if m.reasoning_effort is not None:
        lines.append(f"reasoning_effort = {_toml_value(m.reasoning_effort)}")
    if m.cost_per_m_output is not None:
        lines.append(f"cost_per_m_output = {_toml_value(m.cost_per_m_output)}")
    lines += [
        "",
        "[tools]",
        f"forecast_provider = {_toml_value(settings.tools.forecast_provider)}",
        "",
        "[bench]",
    ]
    for name, value in settings.bench.model_dump().items():
        lines.append(f"{name} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def save_settings(settings: Settings, secrets: dict[str, str]) -> None:
    home = beatspy_home()
    home.mkdir(parents=True, exist_ok=True)
    config_path().write_text(render_config(settings), encoding="utf-8")
    sp = secrets_path()
    body = "".join(f"{key}={value}\n" for key, value in sorted(secrets.items()) if value)
    sp.write_text("# BeatSPY secrets. Do not commit or share this file.\n" + body, encoding="utf-8")
    with contextlib.suppress(OSError):  # best effort on platforms without POSIX permissions
        sp.chmod(0o600)
