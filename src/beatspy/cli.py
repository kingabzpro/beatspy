"""BeatSPY command line interface.

Commands: setup, doctor, scenarios, run, compare, report, demo, validate, submit.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import logging
import sys
import webbrowser
from pathlib import Path

import pandas as pd

from . import __version__
from .config import (
    MODEL_PRESETS,
    ModelConfig,
    Settings,
    ToolsConfig,
    config_path,
    data_dir,
    load_settings,
    model_api_key,
    provider_api_key,
    read_secrets,
    results_dir,
    save_settings,
    secrets_path,
)
from .models.provider import BeatSpyModelProvider, probe_capabilities
from .scenarios import available_names, load_events, load_scenario

log = logging.getLogger("beatspy")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser = argparse.ArgumentParser(
        prog="beatspy",
        description="Give any AI model a team of trading agents. Can it beat SPY?",
        epilog="Run locally. Share validated results through a pull request.",
    )
    parser.add_argument("--version", action="version", version=f"beatspy {__version__}")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("setup", parents=[common], help="interactive configuration wizard")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("doctor", parents=[common], help="check config, model, and data health")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("scenarios", parents=[common], help="list available benchmark scenarios")
    p.set_defaults(func=cmd_scenarios)

    p = sub.add_parser("run", parents=[common], help="run a benchmark scenario")
    p.add_argument("--scenario", action="append", help="repeat for multiple scenarios (default: 2026-recent)")
    p.add_argument("--model", action="append", help="repeat for multiple models")
    p.add_argument("--base-url", help="override configured model base URL")
    p.add_argument("--freq", choices=["weekly", "monthly"], help="decision frequency override")
    p.add_argument("--start", help="override start date YYYY-MM-DD")
    p.add_argument("--end", help="override end date YYYY-MM-DD")
    p.add_argument("--label", help="free-text label stored in the run record")
    p.add_argument("--refresh-data", action="store_true", help="re-download the frozen price snapshot")
    p.add_argument("--jobs", type=positive_int, default=2, help="parallel independent runs (default: 2)")
    p.add_argument(
        "--concurrency", type=positive_int, default=6, help="active agent and external request limits (default: 6)"
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", parents=[common], help="leaderboard across runs")
    p.add_argument("--scenario", help="filter by scenario name")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--include-synthetic", action="store_true", help="include synthetic demo runs")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("report", parents=[common], help="render the HTML dashboard")
    p.add_argument("--run", help="run id to feature (default: latest)")
    p.add_argument("--open", action="store_true", help="open the report in a browser")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("demo", parents=[common], help="generate labeled synthetic runs and a report")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_demo)

    for command in ("validate", "submit"):
        p = sub.add_parser(
            command, parents=[common], help="replay a run" if command == "validate" else "prepare result files for a PR"
        )
        p.add_argument("--run", required=True, type=Path, help="path to a completed run directory")
        if command == "submit":
            p.add_argument("--data-root", type=Path, default=Path("dashboard/data"))
        p.set_defaults(func=cmd_validate if command == "validate" else cmd_submit)

    return parser


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _secret_input(prompt: str) -> str:
    """getpass reads the console on Windows, which hangs for piped/CI stdin."""
    if sys.stdin.isatty():
        return getpass.getpass(prompt)
    return input(prompt)


# --------------------------------------------------------------------------- #


def cmd_setup(args: argparse.Namespace) -> int:
    print("BeatSPY setup")
    print("API keys are stored in your home directory, never inside a repository.\n")

    preset_names = list(MODEL_PRESETS)
    for i, name in enumerate(preset_names, 1):
        url = MODEL_PRESETS[name]
        print(f"  {i}. {name}" + (f"  ({url})" if url else "  (your own OpenAI-compatible endpoint)"))
    choice = input(f"\nPreset [{preset_names[0]}]: ").strip().lower() or preset_names[0]
    if choice.isdigit() and 1 <= int(choice) <= len(preset_names):
        choice = preset_names[int(choice) - 1]
    if choice not in MODEL_PRESETS:
        print(f"Unknown preset {choice!r}; using custom.")
        choice = "custom"

    default_url = MODEL_PRESETS[choice] or "https://api.openai.com/v1"
    base_url = input(f"Model base URL [{default_url}]: ").strip() or default_url
    model = input("Model name (e.g. llama3.1:8b, gpt-4o-mini): ").strip()
    if not model:
        print("A model name is required.")
        return 1
    api_key = _secret_input("Model API key (leave empty for local servers): ").strip()

    print("\nOptional integrations; press Enter to skip any of them.")
    olostep = _secret_input("Olostep API key (web search): ").strip()
    finnhub = _secret_input("Finnhub API key (news, fundamentals): ").strip()
    timegpt = _secret_input("TimeGPT API key (forecasting): ").strip()
    forecast = input("Forecast provider [baseline/naive/chronos/timegpt, default baseline]: ").strip() or "baseline"

    settings = Settings(
        model=ModelConfig(base_url=base_url, model=model),
        tools=ToolsConfig(forecast_provider=forecast),
    )
    # Keep previously stored keys that the user left blank this time.
    secrets = {
        **read_secrets(),
        **{
            k: v
            for k, v in {
                "BEATSPY_API_KEY": api_key,
                "BEATSPY_OLOSTEP_API_KEY": olostep,
                "BEATSPY_FINNHUB_API_KEY": finnhub,
                "BEATSPY_TIMEGPT_API_KEY": timegpt,
            }.items()
            if v
        },
    }
    save_settings(settings, secrets)
    print(f"\nWrote {config_path()}")
    print(f"Wrote {secrets_path()} (chmod 600 where supported)")

    print("\nProbing model capabilities...")
    # probes fail fast: no retries, short timeout, so a dead endpoint answers quickly
    provider = BeatSpyModelProvider(base_url, api_key or None, request_timeout=15.0, max_retries=0)
    probes = asyncio.run(probe_capabilities(provider, model))
    for name, (ok, detail) in probes.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print("\nNext steps:")
    print("  beatspy doctor                      # full health check")
    print("  beatspy run --scenario 2026-recent    # first benchmark run")
    print("  beatspy report                      # HTML dashboard")
    return 0 if all(ok for ok, _ in probes.values()) else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[tuple[str, str, str]] = []

    try:
        settings = load_settings()
    except Exception as exc:
        checks.append(("config", "FAIL", f"config could not be loaded: {exc}"))
        _print_checks(checks)
        return 1

    if config_path().exists():
        checks.append(("config", "PASS", f"{settings.model.model} @ {settings.model.base_url}"))
    else:
        checks.append(("config", "WARN", f"no config file at {config_path()}; using defaults (run: beatspy setup)"))

    provider = BeatSpyModelProvider(
        settings.model.base_url, model_api_key(settings), request_timeout=15.0, max_retries=0
    )
    probes = asyncio.run(probe_capabilities(provider, settings.model.model))
    for name, (ok, detail) in probes.items():
        checks.append((f"model {name}", "PASS" if ok else "FAIL", detail))

    names = available_names()
    checks.append(("scenarios", "PASS" if names else "FAIL", ", ".join(names)))

    snapshots = sorted(p.name for p in data_dir().iterdir()) if data_dir().exists() else []
    checks.append(
        (
            "frozen data",
            "PASS" if snapshots else "WARN",
            ", ".join(snapshots) if snapshots else "no snapshots yet; the first run downloads prices via yfinance",
        )
    )

    forecast = settings.tools.forecast_provider
    if forecast == "chronos":
        try:
            import chronos  # noqa: F401

            checks.append(("forecast provider", "PASS", "chronos installed"))
        except ImportError:
            checks.append(("forecast provider", "FAIL", "chronos not installed; run: uv sync --extra chronos"))
    elif forecast == "timegpt":
        checks.append(("forecast provider", "PASS" if provider_api_key(settings, "timegpt") else "FAIL", "timegpt"))
    else:
        checks.append(("forecast provider", "PASS", f"{forecast} (built-in, no dependencies)"))

    _print_checks(checks)
    return 0 if all(status != "FAIL" for _, status, _ in checks) else 1


def _print_checks(checks: list[tuple[str, str, str]]) -> None:
    for name, status, detail in checks:
        marker = {"PASS": "\033[92mPASS\033[0m", "FAIL": "\033[91mFAIL\033[0m", "WARN": "\033[93mWARN\033[0m"}.get(
            status, status
        )
        print(f"  [{marker}] {name}: {detail}")


def cmd_scenarios(args: argparse.Namespace) -> int:
    for name in available_names():
        try:
            scenario = load_scenario(name)
        except Exception as exc:
            print(f"  {name}: unreadable ({exc})")
            continue
        from .data.calendar import resolve_scenario

        scenario = resolve_scenario(scenario)
        events = len(load_events(name))
        print(f"  {name}")
        print(f"    {scenario.title}")
        print(
            f"    {scenario.start} to {scenario.end}, decisions {scenario.frequency}, "
            f"universe {', '.join(scenario.tradable)} (+{scenario.benchmark} benchmark), {events} curated events"
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .bench.batch import run_batch

    settings = load_settings()
    if args.base_url:
        settings.model.base_url = args.base_url
    outcomes = asyncio.run(
        run_batch(
            [load_scenario(name) for name in (args.scenario or ["2026-recent"])],
            settings,
            args.model,
            jobs=args.jobs,
            concurrency=args.concurrency,
            freq=args.freq,
            start=args.start,
            end=args.end,
            label=args.label,
            refresh_data=args.refresh_data,
        )
    )
    failed = False
    for outcome in outcomes:
        if isinstance(outcome, Exception):
            print(f"FAILED: {outcome}", file=sys.stderr)
            failed = True
            continue
        metrics = json.loads((outcome / "metrics.json").read_text(encoding="utf-8"))
        print(f"Run complete: {outcome}")
        print(
            f"  Return {metrics['total_return']:.2%} | SPY {metrics['spy_total_return']:.2%} | "
            f"Excess {metrics['excess_return_vs_spy']:.2%}"
        )
        print(
            f"  Requests {metrics['requests']} | Tokens {metrics['input_tokens']:,} in / "
            f"{metrics['output_tokens']:,} out"
        )
    print("Hypothetical backtest results. Preview with: beatspy report")
    return int(failed)


def cmd_compare(args: argparse.Namespace) -> int:
    from .reporting.dashboard import collect_runs

    runs = collect_runs(results_dir())
    if not args.include_synthetic:
        runs = [r for r in runs if not r["synthetic"]]
    if args.scenario:
        runs = [r for r in runs if r["scenario"] == args.scenario]
    if not runs:
        print("No runs found in ./results. Try: beatspy run --scenario 2026-recent  (or beatspy demo)")
        return 1

    if args.json:
        print(json.dumps([{k: v for k, v in r.items() if k not in ("meta",)} for r in runs], indent=2, default=str))
        return 0

    runs = sorted(runs, key=lambda r: (r["group"], -r["metrics"].get("excess_return_vs_spy", 0.0)))
    frame = pd.DataFrame(
        [
            {
                "group": r["group"],
                "model": r["model"],
                "scenario": r["scenario"],
                "period": f"{r['start']}..{r['end']}",
                "total_return": f"{r['metrics'].get('total_return', 0) * 100:.1f}%",
                "spy": f"{r['metrics'].get('spy_total_return', 0) * 100:.1f}%",
                "excess": f"{r['metrics'].get('excess_return_vs_spy', 0) * 100:.1f}%",
                "sharpe": r["metrics"].get("sharpe"),
                "max_dd": f"{r['metrics'].get('max_drawdown', 0) * 100:.1f}%",
                "dir_acc": f"{r['metrics'].get('directional_accuracy', 0) * 100:.0f}%",
                "decisions": r["metrics"].get("decisions"),
                "invalid": r["metrics"].get("invalid_outputs"),
                "tokens_in": r["metrics"].get("input_tokens"),
                "tokens_out": r["metrics"].get("output_tokens"),
                "synthetic": r["synthetic"],
            }
            for r in runs
        ]
    )
    print(frame.to_string(index=False))
    print(
        "\nGrouped by identical windows, snapshots, and rules; sorted by excess within each group. "
        "Hypothetical backtest results; not live performance."
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .reporting.dashboard import render_dashboard

    path = render_dashboard(results_dir(), run_id=args.run)
    print(f"Dashboard written to {path}")
    print(f"Preview: python -m http.server 8000 --bind 127.0.0.1 --directory {path.parent}")
    if args.open:
        from functools import partial
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

        with ThreadingHTTPServer(
            ("127.0.0.1", 8000), partial(SimpleHTTPRequestHandler, directory=str(path.parent))
        ) as server:
            webbrowser.open("http://127.0.0.1:8000/" + (f"#run={args.run}" if args.run else ""))
            print("Serving local dashboard; press Ctrl+C to stop.")
            server.serve_forever()
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .bench.demo import generate_demo_runs
    from .reporting.dashboard import render_dashboard

    dirs = generate_demo_runs(seed=args.seed)
    path = render_dashboard(results_dir())
    print(f"Generated {len(dirs)} SYNTHETIC demo runs under {results_dir()}")
    print(f"Dashboard written to {path}")
    print("Demo runs use fabricated data and are excluded from `beatspy compare` by default.")
    return 0


def cmd_validate(args):
    from .bench.validation import validate_run

    meta = validate_run(args.run)
    print(f"PASS: {meta['run_id']} — artifact hashes, decisions, trades, equity, and metrics replay correctly.")
    print("Model identity remains unverified.")
    return 0


def cmd_submit(args):
    from .reporting.catalog import submit_run

    path = submit_run(args.run, args.data_root)
    print(f"Prepared community submission: {path}")
    print(
        "Review the exported files, commit dashboard/data, and open a PR: https://github.com/kingabzpro/beatspy/compare"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # httpx request lines include full URLs with query strings (API tokens for
    # Finnhub) and are pure noise at INFO; keep them for --verbose only.
    for noisy in ("httpx", "httpx2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        if args.command is None:
            parser.print_help()
            return 0
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        log.debug("command failed", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
