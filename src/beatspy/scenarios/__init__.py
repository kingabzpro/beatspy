"""Scenario loading: bundled TOMLs plus user scenarios in ~/.beatspy/scenarios.

A scenario file defines dates, universe, costs, and risk limits; an optional
`<name>.events.toml` beside it provides the curated event feed. Bundled
scenarios never require external API subscriptions.
"""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path

from ..schemas import EventItem, Scenario


def user_scenario_dir() -> Path:
    return Path.home() / ".beatspy" / "scenarios"


def available_names() -> list[str]:
    names: set[str] = set()
    builtin = resources.files("beatspy.scenarios") / "builtin"
    for entry in builtin.iterdir():
        if entry.name.endswith(".toml") and not entry.name.endswith(".events.toml"):
            names.add(entry.name.removesuffix(".toml"))
    user = user_scenario_dir()
    if user.exists():
        names |= {p.name.removesuffix(".toml") for p in user.glob("*.toml") if not p.name.endswith(".events.toml")}
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


def load_events(name: str) -> list[EventItem]:
    user_file = user_scenario_dir() / f"{name}.events.toml"
    if user_file.exists():
        data = _read_toml(user_file)
    else:
        builtin = resources.files("beatspy.scenarios") / "builtin" / f"{name}.events.toml"
        if not builtin.is_file():
            return []
        data = _read_toml(builtin)
    return [EventItem.model_validate(item) for item in data.get("events", [])]
