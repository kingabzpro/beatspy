"""Scenario loading: bundled TOMLs plus user scenarios in ~/.beatspy/scenarios.

A scenario file defines dates, universe, costs, and risk limits. Bundled
scenarios never require external API subscriptions, and they are frozen price
windows rather than live feeds.

The curated event feeds the old agent benchmark used are gone: a decision model
receives its evidence as point-in-time state, so there is nothing for a hand-kept
news list to contribute.
"""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path

from ..schemas import Scenario

DEFAULT_SCENARIO = "2026-ytd"


def user_scenario_dir() -> Path:
    return Path.home() / ".beatspy" / "scenarios"


def make_scenario(**overrides) -> Scenario:
    """Build a Scenario from defaults plus overrides (used by tests and tools)."""
    payload = {
        "name": "custom",
        "title": "Custom scenario",
        "start": "2022-01-03",
        "end": "2022-03-31",
        "frequency": "monthly",
        "benchmark": "SPY",
        "tradable": ["AAPL", "MSFT", "TLT"],
        "initial_capital": 100_000.0,
    }
    payload.update(overrides)
    return Scenario(**payload)


def available_names() -> list[str]:
    names: set[str] = set()
    builtin = resources.files("beatspy.scenarios") / "builtin"
    for entry in builtin.iterdir():
        if entry.name.endswith(".toml"):
            names.add(entry.name.removesuffix(".toml"))
    user = user_scenario_dir()
    if user.exists():
        names |= {path.name.removesuffix(".toml") for path in user.glob("*.toml")}
    return sorted(names)


def _read_toml(path: Path | resources.Traversable) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_scenario(name: str) -> Scenario:
    user_file = user_scenario_dir() / f"{name}.toml"
    if user_file.exists():
        data = _read_toml(user_file)
    else:
        builtin = resources.files("beatspy.scenarios") / "builtin" / f"{name}.toml"
        if not builtin.is_file():
            raise FileNotFoundError(f"unknown scenario {name!r}; available: {', '.join(available_names())}")
        data = _read_toml(builtin)
    data["name"] = name
    return Scenario.model_validate(data)