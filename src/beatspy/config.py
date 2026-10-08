"""Configuration, credentials, and environment handling.

Precedence: CLI flags > BEATSPY_* environment variables > ~/.beatspy/config.toml > defaults.
API keys live in ~/.beatspy/secrets.env (never inside a repository) or in the environment.

This benchmark talks to *decision models*, not chat models. A decision model
reads a state plus typed questions and returns probabilities, so there is no
temperature, no turn budget, and no tool budget to configure — only which
endpoint to call, how hard to retry it, and how its probabilities become a
portfolio.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Decision-model providers. All three speak the same state-plus-typed-questions
# contract; they differ only in transport, auth, and question limits.
PROVIDERS = ("openai_decisions", "typesafe", "cloudflare")

# The decision models this benchmark covers, with pricing verified against the
# sources below on 2026-10-08. Decision models bill input only: there is no
# output charge and no cache charge anywhere in this table, and none of the three
# providers offers prompt caching.
#
# - OpenAI Decisions API (public beta) — gpt-6-luna, $0.10 per 1M input:
#   https://community.openai.com/t/decisions-api-is-now-available-in-public-beta/1403877
# - TypeSafe Jev — $42 per 1B input (~$0.042 per 1M), output free:
#   https://docs.typesafe.ai/models
# - Cloudflare Clef / Clef-flash — $0.24 / $0.09 per 1M input:
#   https://developers.cloudflare.com/workers-ai/models/clef/index.md
DECISION_MODELS: dict[str, dict] = {
    "gpt-6-luna": {
        "provider": "openai_decisions",
        "model": "gpt-6-luna",
        "label": "OpenAI Decisions (gpt-6-luna)",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "cost_per_m_input": 0.10,
        "cost_per_m_output": 0.0,
        "context_window": 128_000,
        "max_questions": 64,
        "state_note": "no prompt caching; input-only billing",
    },
    "jev-latest": {
        "provider": "typesafe",
        "model": "jev-latest",
        "label": "TypeSafe Jev",
        "base_url": "https://api.typesafe.ai/v1/systemone",
        "api_key_env": "TYPESAFE_API_KEY",
        "cost_per_m_input": 0.042,
        "cost_per_m_output": 0.0,
        "context_window": 32_000,
        "max_questions": 64,
        "state_note": "output tokens are free; no prompt caching",
    },
    "clef": {
        "provider": "cloudflare",
        "model": "clef",
        "label": "Cloudflare Clef (27B)",
        "base_url": "https://api.cloudflare.com/client/v4/accounts",
        "api_key_env": "CLOUDFLARE_AUTH_TOKEN",
        "cost_per_m_input": 0.24,
        "cost_per_m_output": 0.0,
        "context_window": 65_536,
        "max_questions": 64,
        "state_note": "Workers AI truncates long text state to roughly the first 2K tokens",
    },
    "clef-flash": {
        "provider": "cloudflare",
        "model": "clef-flash",
        "label": "Cloudflare Clef-flash (9B)",
        "base_url": "https://api.cloudflare.com/client/v4/accounts",
        "api_key_env": "CLOUDFLARE_AUTH_TOKEN",
        "cost_per_m_input": 0.09,
        "cost_per_m_output": 0.0,
        "context_window": 65_536,
        "max_questions": 64,
        "state_note": "Workers AI truncates long text state to roughly the first 2K tokens",
    },
}

# Clef validates question ids as letters, digits, '_', '.', '-', at most 100 chars.
SAFE_QUESTION_ID = r"[A-Za-z0-9_.-]{1,100}"

DEFAULT_DECISION_MODEL = "clef-flash"

# Cloudflare needs an account id alongside the token.
ACCOUNT_ID_ENV = "CLOUDFLARE_ACCOUNT_ID"


class DecisionConfig(BaseModel):
    """How one DCN run talks to its provider and turns answers into a portfolio."""

    model_config = ConfigDict(protected_namespaces=())

    decision_model: str = DEFAULT_DECISION_MODEL
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: float = 120.0
    max_retries: int = 4
    concurrency: int = 6
    # The sizing policy is fixed here so it cannot be tuned after seeing a
    # result. Only noul probabilities enter the portfolio.
    #   graded (default) — fully invested across the whole universe. `base_weight`
    #     of the book is a uniform spread over every asset and the rest is
    #     allocated in proportion to the probability above the prior, so a weak
    #     forecast becomes a mild tilt instead of a binary switch. Without the
    #     uniform base, one confident asset would take the entire book.
    #   top_n — the legacy cut: hold only `top_n` names above `min_probability`,
    #     equal-weighted, cash if fewer than `min_names` qualify. Kept so runs
    #     recorded under it stay reproducible.
    selection: Literal["graded", "top_n"] = "graded"
    # Share of the graded book allocated by conviction; the remainder is even.
    base_weight: float = 0.5
    min_probability: float = 0.5
    top_n: int = 3
    min_names: int = 2
    # Workers AI silently truncates long state to ~2K tokens, so the state is
    # budgeted conservatively for every provider and the budget is recorded.
    max_state_tokens: int = 1800

    @model_validator(mode="after")
    def check_model(self):
        if self.decision_model not in DECISION_MODELS:
            raise ValueError(
                f"unknown decision model {self.decision_model!r}; "
                f"choose from {', '.join(sorted(DECISION_MODELS))}"
            )
        if self.concurrency < 1 or self.max_retries < 0:
            raise ValueError("concurrency must be >= 1 and max_retries >= 0")
        return self

    @property
    def spec(self) -> dict:
        return DECISION_MODELS[self.decision_model]

    @property
    def provider(self) -> str:
        return self.spec["provider"]

    @property
    def label(self) -> str:
        return self.spec["label"]

    @property
    def endpoint(self) -> str:
        return self.base_url or self.spec["base_url"]

    @property
    def max_questions(self) -> int:
        return int(self.spec.get("max_questions", 64))

    @property
    def cost_per_m_input(self) -> float | None:
        return self.spec.get("cost_per_m_input")

    @property
    def cost_per_m_output(self) -> float | None:
        return self.spec.get("cost_per_m_output")


class BenchConfig(BaseModel):
    initial_capital: float = 100_000.0
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    max_position_weight: float = 0.35
    risk_free_annual: float = 0.0


class Settings(BaseModel):
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
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


def env_file_path() -> Path:
    return Path.cwd() / ".env"


def parse_env_file(text: str) -> dict[str, str]:
    """Parse a dotenv-style file: KEY=value, `#` comments, optional quotes.

    A `#` that follows whitespace starts a comment, which is what people expect
    from dotenv files. Getting this wrong is expensive rather than cosmetic: a
    value with a trailing `# note` appended becomes an invalid credential, and
    the provider answers 401 as if the key were simply wrong.
    """
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        # Strip an unquoted trailing comment, but never inside a quoted value.
        if not value.lstrip().startswith(("'", '"')):
            value = re.split(r"\s+#", value, maxsplit=1)[0]
        value = value.strip().strip('"').strip("'")
        key = key.strip()
        if key:
            values[key] = value
    return values


def read_env_file() -> dict[str, str]:
    """Read the project `.env`. It is git-ignored and never required."""
    path = env_file_path()
    if not path.exists():
        return {}
    try:
        return parse_env_file(path.read_text(encoding="utf-8"))
    except OSError:
        return {}


def load_secrets() -> None:
    """Load credentials into the environment, never overriding what is already set.

Precedence: a real exported variable, then `~/.beatspy/secrets.env` (the
deliberate store, so it wins over a project file), then the project `.env`.
A `.env` in the working directory is what most people expect to work, so it is
honoured rather than silently ignored.
    """
    merged = {**read_env_file(), **read_secrets()}
    for key, value in merged.items():
        if value:
            os.environ.setdefault(key, value)


def read_secrets() -> dict[str, str]:
    path = secrets_path()
    if not path.exists():
        return {}
    return parse_env_file(path.read_text(encoding="utf-8"))


def secret_values() -> set[str]:
    values = set(read_secrets().values())
    values |= {
        value
        for key, value in os.environ.items()
        if key in {"OPENAI_API_KEY", "TYPESAFE_API_KEY", "CLOUDFLARE_AUTH_TOKEN", "CLOUDFLARE_ACCOUNT_ID"}
        or (key.startswith("BEATSPY_") and "KEY" in key)
    }
    return values


def _api_key_env_names(spec: dict) -> list[str]:
    primary = spec["api_key_env"]
    return [f"BEATSPY_{primary}", primary]


def decision_api_key(decision_model: str) -> str | None:
    """Resolve the API key for one decision model from the environment."""
    spec = DECISION_MODELS[decision_model]
    for env_name in _api_key_env_names(spec):
        if value := os.environ.get(env_name):
            return value
    return None


def cloudflare_account_id() -> str | None:
    return os.environ.get(ACCOUNT_ID_ENV) or os.environ.get(f"BEATSPY_{ACCOUNT_ID_ENV}")


def load_settings(path: Path | None = None) -> Settings:
    load_secrets()
    path = path or config_path()
    data: dict = {}
    if path.exists():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    settings = Settings.model_validate(data)

    if (v := os.environ.get("BEATSPY_DECISION_MODEL")) is not None:
        settings.decision.decision_model = v
        settings = Settings.model_validate(settings.model_dump())
    if (v := os.environ.get("BEATSPY_DECISION_BASE_URL")) is not None:
        settings.decision.base_url = v
    if (v := os.environ.get("BEATSPY_DECISION_CONCURRENCY")) is not None:
        settings.decision.concurrency = int(v)
    if (v := os.environ.get("BEATSPY_DECISION_TIMEOUT_S")) is not None:
        settings.decision.timeout_s = float(v)
    return settings


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value))


def render_config(settings: Settings) -> str:
    d = settings.decision
    lines = [
        "[decision]",
        f"decision_model = {_toml_value(d.decision_model)}",
        f"timeout_s = {_toml_value(d.timeout_s)}",
        f"max_retries = {_toml_value(d.max_retries)}",
        f"concurrency = {_toml_value(d.concurrency)}",
        f"min_probability = {_toml_value(d.min_probability)}",
        f"top_n = {_toml_value(d.top_n)}",
        f"min_names = {_toml_value(d.min_names)}",
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