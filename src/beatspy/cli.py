"""BeatSPY command line interface.

Commands: setup, doctor, models, scenarios, run, calibrate, verify, compare,
report, demo.

This is a DCN benchmark: a *decision model* reads a frozen point-in-time state
plus typed questions and answers with probabilities. Each command below is about
choosing one of those models, running it over frozen prices, and scoring its
probabilities as well as the portfolio they produce.

Only two commands can reach the network, and each does so only when the user
asked for it: `run` without `--offline-fixture`, and `doctor --probe`. `setup`
stores credentials but never calls a provider.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import logging
import sys
import webbrowser
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from . import __version__
from .config import (
    ACCOUNT_ID_ENV,
    DECISION_MODELS,
    DEFAULT_DECISION_MODEL,
    DecisionConfig,
    Settings,
    cloudflare_account_id,
    config_path,
    data_dir,
    decision_api_key,
    load_settings,
    read_secrets,
    results_dir,
    save_settings,
    secrets_path,
)
from .scenarios import available_names, load_scenario

log = logging.getLogger("beatspy")

DEFAULT_SCENARIO = "2026-ytd"
DEMO_DECISION_MODELS = ("gpt-6-luna", "jev-latest", "clef-flash")
DEMO_TICKERS = ["AAPL", "MSFT", "TLT", "GLD"]


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser = argparse.ArgumentParser(
        prog="beatspy",
        description="Benchmark decision models on frozen market data. Can any of them beat SPY?",
        epilog="Everything runs locally. Probabilities and prices decide the score; no LLM judge.",
    )
    parser.add_argument("--version", action="version", version=f"beatspy {__version__}")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("setup", parents=[common], help="choose which decision model to benchmark")
    p.add_argument("--model", help="configure without prompts (a decision model id)")
    p.add_argument("--base-url", help="override the provider endpoint (optional)")
    p.add_argument("--api-key", help="store this provider key in secrets.env instead of prompting")
    p.add_argument("--account-id", help="Cloudflare account id (Clef models only)")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("doctor", parents=[common], help="check config, credentials, and scenario health")
    p.add_argument("--probe", action="store_true", help="send one live decision request per model (network)")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("models", parents=[common], help="list decision models, pricing, and key status")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("scenarios", parents=[common], help="list available benchmark scenarios")
    p.set_defaults(func=cmd_scenarios)

    p = sub.add_parser("run", parents=[common], help="run one decision model over one scenario")
    p.add_argument(
        "--model",
        nargs="+",
        help=(
            f"decision model id(s) to run (default: the configured one, {DEFAULT_DECISION_MODEL}). "
            "Several ids run concurrently, since they share nothing but the frozen snapshot."
        ),
    )
    p.add_argument("--scenario", help=f"scenario name (default: {DEFAULT_SCENARIO})")
    p.add_argument("--freq", choices=["weekly", "monthly"], help="decision frequency override")
    p.add_argument("--start", help="override start date YYYY-MM-DD")
    p.add_argument("--end", help="override end date YYYY-MM-DD")
    p.add_argument("--max-decisions", type=positive_int, help="limit decisions, evenly spread across the period")
    p.add_argument(
        "--form",
        choices=["per_asset", "noul", "choice", "rank", "twin", "all"],
        default="per_asset",
        help=(
            "question form (default: per_asset — one question for every investable asset, "
            "which is what makes a single asset's answer mean something on its own). "
            "twin/all ask extra forms for the same ticker to measure the form itself."
        ),
    )
    p.add_argument("--label", help="free-text label stored in the run record")
    p.add_argument("--refresh-data", action="store_true", help="re-download the frozen price snapshot")
    p.add_argument(
        "--offline-fixture",
        action="store_true",
        help="run against scripted synthetic data and a scripted client: no credentials, no network (test aid)",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("calibrate", parents=[common], help="print the calibration table of a finished run")
    p.add_argument("run", nargs="?", help="run directory or run id (default: latest run)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("verify", parents=[common], help="replay a finished run offline and check its artifacts")
    p.add_argument("run", nargs="?", help="run directory or run id (default: latest run)")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("compare", aliases=["results"], parents=[common], help="latest run per decision model")
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


def _require_decision_model(name: str) -> dict:
    if name not in DECISION_MODELS:
        raise ValueError(f"unknown decision model {name!r}; choose from {', '.join(DECISION_MODELS)}")
    return DECISION_MODELS[name]


def _number(value, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _percent(value, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def _pricing(spec: dict) -> str:
    out = spec.get("cost_per_m_output") or 0.0
    return f"${spec['cost_per_m_input']:.3f}/M input, ${out:.3f}/M output"


def _key_status(name: str, spec: dict) -> tuple[bool, str]:
    """Whether a model can authenticate, plus a one-line explanation."""
    if not decision_api_key(name):
        return False, f"missing {spec['api_key_env']}"
    if spec["provider"] == "cloudflare" and not cloudflare_account_id():
        return False, f"set {ACCOUNT_ID_ENV} for the account id"
    return True, "key found"


# --------------------------------------------------------------------------- #
# setup / doctor / models / scenarios
# --------------------------------------------------------------------------- #


def cmd_setup(args: argparse.Namespace) -> int:
    settings = load_settings()
    secrets = read_secrets()

    model = args.model
    if not model:
        print("BeatSPY setup — choose the decision model to benchmark.")
        print("Decision models return typed probabilities; this benchmark scores them, not their prose.\n")
        names = list(DECISION_MODELS)
        for index, name in enumerate(names, 1):
            spec = DECISION_MODELS[name]
            print(f"  {index}. {name:<12} {spec['label']:<32} {_pricing(spec)}")
        entered = input(f"\nDecision model [{settings.decision.decision_model}]: ").strip()
        entered = entered or settings.decision.decision_model
        if entered.isdigit() and 1 <= int(entered) <= len(names):
            entered = names[int(entered) - 1]
        model = entered

    spec = _require_decision_model(model)
    settings.decision.decision_model = model
    if args.base_url:
        settings.decision.base_url = args.base_url

    if args.api_key:
        secrets[spec["api_key_env"]] = args.api_key
    if args.account_id:
        secrets[ACCOUNT_ID_ENV] = args.account_id
    if not args.model:  # interactive: offer to store the key
        key = _secret_input(f"{spec['api_key_env']} (leave empty to use the environment): ").strip()
        if key:
            secrets[spec["api_key_env"]] = key
        if spec["provider"] == "cloudflare":
            account = input(f"Cloudflare account id (leave empty to use {ACCOUNT_ID_ENV}): ").strip()
            if account:
                secrets[ACCOUNT_ID_ENV] = account

    settings = Settings.model_validate(settings.model_dump())
    save_settings(settings, secrets)

    ok, detail = _key_status(model, spec)
    print(f"\nConfigured decision model: {settings.decision.label} ({model})")
    print(f"  endpoint  {settings.decision.endpoint}")
    print(f"  pricing   {_pricing(spec)}")
    print(f"  key       {spec['api_key_env']}: {detail}")
    print(f"  config    {config_path()}")
    print(f"  secrets   {secrets_path()}")
    print("\nNext steps:")
    print("  beatspy doctor              # health check (add --probe for one live call per model)")
    print("  beatspy run                 # run the configured model on the default scenario")
    if not ok:
        print(f"  (set {spec['api_key_env']} in the environment or rerun setup to store it)")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[tuple[str, str, str]] = []

    try:
        settings = load_settings()
    except Exception as exc:
        checks.append(("config", "FAIL", f"config could not be loaded: {exc}"))
        _print_checks(checks)
        return 1

    if config_path().exists():
        checks.append(("config", "PASS", f"{settings.decision.decision_model} @ {settings.decision.endpoint}"))
    else:
        checks.append(("config", "WARN", f"no config file at {config_path()}; using defaults (run: beatspy setup)"))

    for name, spec in DECISION_MODELS.items():
        ok, detail = _key_status(name, spec)
        checks.append(
            (
                f"model {name}",
                "PASS" if ok else "WARN",
                f"{spec['label']} | {spec['base_url']} | {_pricing(spec)} | {detail}",
            )
        )

    names = available_names()
    checks.append(("scenarios", "PASS" if names else "FAIL", ", ".join(names) if names else "none found"))

    snapshots = sorted(path.name for path in data_dir().iterdir()) if data_dir().exists() else []
    checks.append(
        (
            "frozen data",
            "PASS" if snapshots else "WARN",
            ", ".join(snapshots) if snapshots else "no snapshots yet; a live run downloads prices via yfinance",
        )
    )

    _print_checks(checks)

    failed = any(status == "FAIL" for _, status, _ in checks)
    if not args.probe:
        print("\nNo network calls were made. Add --probe for one live decision request per model.")
        return int(failed)

    print("\nProbing every decision model with a single noul question (live network calls)...")
    for name in DECISION_MODELS:
        ok, detail = _probe_decision_model(name)
        print(f"  [{'PASS' if ok else 'FAIL'}] probe {name}: {detail}")
        failed = failed or not ok
    return int(failed)


def _print_checks(checks: list[tuple[str, str, str]]) -> None:
    for name, status, detail in checks:
        marker = {"PASS": "\033[92mPASS\033[0m", "FAIL": "\033[91mFAIL\033[0m", "WARN": "\033[93mWARN\033[0m"}.get(
            status, status
        )
        print(f"  [{marker}] {name}: {detail}")


def _probe_state() -> dict:
    """A minimal, static state shaped like a real decision's state."""
    return {
        "as_of": date.today().isoformat(),
        "scenario": "probe",
        "horizon_trading_days": 5,
        "next_decision_date": None,
        "benchmark": {"ticker": "SPY", "return_30d_pct": 1.0, "return_90d_pct": 4.0},
        "costs": {"fee_bps": 5.0, "slippage_bps": 5.0, "max_position_weight_pct": 35.0},
        "portfolio": {"cash_weight_pct": 100.0, "holdings": []},
        "market": [
            {
                "ticker": "AAPL",
                "last_close": 210.0,
                "last_5d_pct": 0.6,
                "return_30d_pct": 2.4,
                "return_90d_pct": 5.1,
                "return_90d_vs_benchmark_pct": 1.1,
            }
        ],
        "decide_on": ["AAPL"],
        "constraints": {"long_only": True, "notes": "connectivity probe; not a benchmark decision"},
        "recent_decisions": [],
    }


def _probe_decision_model(name: str) -> tuple[bool, str]:
    """One live call for one model. Returns (ok, human detail or exact error)."""
    from .dcn.clients import make_client
    from .dcn.questions import build_questions

    settings = Settings(decision=DecisionConfig(decision_model=name))
    questions = build_questions("noul", ["AAPL"], "SPY", 5).questions

    async def call():
        client = make_client(settings)
        try:
            return await client.decide(_probe_state(), questions)
        finally:
            await client.close()

    try:
        result = asyncio.run(call())
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    answer = result.answers.get("noul_AAPL")
    if answer is None:
        missing = ", ".join(result.missing_ids) or "none reported"
        return False, f"no answer for noul_AAPL (missing ids: {missing})"
    return True, (
        f"noul_AAPL={answer.noul:.4f} | {result.input_tokens:,} input tokens | "
        f"{result.latency_s:.2f}s | {result.attempts} attempt(s)"
    )


def cmd_models(args: argparse.Namespace) -> int:
    settings = load_settings()
    active = settings.decision.decision_model
    print(f"{'':<2}{'model':<12} {'provider':<17} {'$ / M input':>12} {'context':>9}  key")
    for name, spec in DECISION_MODELS.items():
        ok, detail = _key_status(name, spec)
        marker = "*" if name == active else " "
        context = f"{spec['context_window'] // 1000}k"
        print(
            f"{marker:<2}{name:<12} {spec['provider']:<17} {spec['cost_per_m_input']:>12.3f} {context:>9}  "
            f"{'yes' if ok else 'no'} ({detail})"
        )
    print(f"\n* configured decision model ({DEFAULT_DECISION_MODEL} is the default).")
    print("Decision models bill input tokens only; output is free and no provider offers prompt caching.")
    print("Run `beatspy doctor --probe` for one live call per model.")
    return 0


def cmd_scenarios(args: argparse.Namespace) -> int:
    from .data.calendar import resolve_scenario

    for name in available_names():
        try:
            scenario = resolve_scenario(load_scenario(name))
        except Exception as exc:
            print(f"  {name}: unreadable ({exc})")
            continue
        print(f"  {name}")
        print(f"    {scenario.title}")
        print(
            f"    {scenario.start} to {scenario.end}, decisions {scenario.frequency}, "
            f"universe {', '.join(scenario.tradable)} (+{scenario.benchmark} benchmark), "
            f"up to {scenario.max_candidates} candidates per decision"
        )
    return 0


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def _fixture_helpers():
    """Load tests/conftest.py by path. `--offline-fixture` is a documented test aid."""
    import importlib.util

    candidates = [
        Path(__file__).resolve().parents[2] / "tests" / "conftest.py",
        Path.cwd() / "tests" / "conftest.py",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise RuntimeError(
            "--offline-fixture needs tests/conftest.py beside the checkout; "
            "install the test dependencies and run from the repository root"
        )
    spec = importlib.util.spec_from_file_location("beatspy_offline_fixture", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise RuntimeError(
            f"--offline-fixture needs the test dependencies (pytest, exchange_calendars): {exc}"
        ) from exc
    return module


def _offline_fixture(base_scenario, args):
    """Scripted client + synthetic frozen snapshot, so a run needs no credentials.

    `base_scenario` supplies the scenario name and frequency when the requested
    scenario exists; otherwise the fixture names itself. Prices are always
    generated, so this mode never touches the network.
    """
    from .data.freeze import snapshot_id
    from .data.service import DataService
    from .schemas import Scenario

    helpers = _fixture_helpers()
    start = args.start or "2022-01-03"
    end = args.end or "2022-03-31"
    fixture = helpers.make_scenario(
        name=base_scenario.name if base_scenario else (args.scenario or "offline-fixture"),
        title=f"{base_scenario.title if base_scenario else 'Offline fixture'} (offline fixture)",
        description="Synthetic offline fixture: generated prices, scripted decision model, no network.",
        start=start,
        end=end,
        frequency=args.freq or (base_scenario.frequency if base_scenario else "monthly"),
    )
    if args.max_decisions is not None:
        fixture = Scenario.model_validate({**fixture.model_dump(), "max_decisions": args.max_decisions})

    prices = helpers.make_prices(
        fixture.universe,
        start=(date.fromisoformat(start) - timedelta(days=180)).isoformat(),
        end=(date.fromisoformat(end) + timedelta(days=100)).isoformat(),
    )
    directory = data_dir() / "offline-fixture"
    directory.mkdir(parents=True, exist_ok=True)
    prices.to_csv(directory / "prices.csv", index=False, date_format="%Y-%m-%d")

    from .artifacts import sha256

    manifest = {
        "schema_version": 2,
        "scenario": fixture.name,
        "tickers": fixture.universe,
        "start": prices.date.min().date().isoformat(),
        "end": prices.date.max().date().isoformat(),
        "requested_start": fixture.start,
        "requested_end": fixture.end,
        "lookback_days": fixture.lookback_days,
        "rows": len(prices),
        "source": "synthetic offline fixture (tests/conftest.py)",
        "prices_sha256": sha256(directory / "prices.csv"),
        "fetched_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    manifest["snapshot_id"] = snapshot_id(manifest)
    (directory / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    prepared = (DataService(prices, benchmark=fixture.benchmark), {**manifest, "path": str(directory.resolve())})
    return fixture, prepared, helpers.OfflineDcnClient()


def cmd_run(args: argparse.Namespace) -> int:
    from .dcn.runner import run_dcn
    from .schemas import Scenario

    settings = load_settings()
    # `--model` may name several models. They share nothing but the frozen
    # snapshot, so they run concurrently rather than one after another.
    models = list(dict.fromkeys(args.model or [settings.decision.decision_model]))
    for name in models:
        _require_decision_model(name)

    scenario = None
    prepared_data = None
    client = None
    if args.offline_fixture:
        base = None
        if args.scenario:
            try:
                base = load_scenario(args.scenario)
            except FileNotFoundError:
                base = None
        else:
            base = load_scenario(DEFAULT_SCENARIO)
        scenario, prepared_data, client = _offline_fixture(base, args)
        print("Offline fixture: synthetic prices written under .beatspy-data, scripted client, no network.")
        print("This is a test aid. Its numbers are not a benchmark result.\n")
    else:
        scenario = load_scenario(args.scenario or DEFAULT_SCENARIO)
        if args.max_decisions is not None:
            scenario = Scenario.model_validate({**scenario.model_dump(), "max_decisions": args.max_decisions})

    for name in models:
        spec = DECISION_MODELS[name]
        print(f"Decision model : {spec['label']} ({name})")
        print(f"Endpoint       : {settings.decision.base_url or spec['base_url']}")
    print(f"Scenario       : {scenario.name} | form {args.form} | freq {args.freq or scenario.frequency}")
    if len(models) > 1:
        print(f"Models         : {len(models)} in parallel\n")

    async def run_all():
        jobs = {}
        for name in models:
            model_settings = Settings.model_validate(
                {**settings.model_dump(), "decision": {**settings.model_dump()["decision"], "decision_model": name}}
            )

            # Each model walks its own decision schedule, and later decisions
            # depend on earlier weights, so the loop inside run_dcn stays
            # sequential; the models are what run concurrently.
            def progress(index: int, total: int, day: date, model=name) -> None:
                if len(models) == 1:
                    print(f"  decision {index}/{total} {day.isoformat()}", flush=True)
                else:
                    print(f"  [{model}] decision {index}/{total} {day.isoformat()}", flush=True)

            jobs[name] = asyncio.create_task(
                run_dcn(
                    scenario,
                    model_settings,
                    freq=args.freq,
                    start=args.start,
                    end=args.end,
                    label=args.label,
                    refresh_data=args.refresh_data,
                    progress=progress,
                    prepared_data=prepared_data,
                    client=client,
                    question_form=args.form,
                )
            )
        return await asyncio.gather(*jobs.values(), return_exceptions=True)

    outcomes = asyncio.run(run_all())

    failed = False
    for name, outcome in zip(models, outcomes, strict=True):
        if isinstance(outcome, Exception):
            print(f"\nFAILED {name}: {type(outcome).__name__}: {outcome}", file=sys.stderr)
            failed = True
            continue
        out_dir = outcome
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        calibration = metrics.get("calibration") or {}
        noul = (calibration.get("signals") or {}).get("noul") or {}
        print(f"\nRun complete: {out_dir}")
        print(
            f"  Return {_percent(metrics.get('total_return'))} | SPY {_percent(metrics.get('spy_total_return'))} | "
        f"Excess {_percent(metrics.get('excess_return_vs_spy'))}"
    )
    print(
        f"  Sharpe {_number(metrics.get('sharpe'))} | Max DD {_percent(metrics.get('max_drawdown'))} | "
        f"Decisions {metrics.get('decisions')} | Trades {metrics.get('trades')}"
    )
    print(
        f"  Calibration (noul): n {noul.get('n', 0)} | Brier {_number(noul.get('brier'))} | "
        f"skill {_number(noul.get('brier_skill_score'))} | coverage {_percent(metrics.get('answer_coverage'))}"
    )
    print(
        f"  Requests {metrics.get('requests')} | {metrics.get('input_tokens', 0):,} input tokens | "
        f"est. cost ${_number(metrics.get('estimated_cost_usd'), 6)}"
    )
    print("\nNext: beatspy calibrate   # probability quality   |   beatspy report   # dashboard")
    return int(failed)


# --------------------------------------------------------------------------- #
# calibrate / verify / compare
# --------------------------------------------------------------------------- #


def _latest_run() -> Path:
    root = results_dir()
    rows = []
    if root.exists():
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or not (directory / "run.json").is_file():
                continue
            try:
                meta = json.loads((directory / "run.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            rows.append((meta.get("created_utc", ""), directory))
    if not rows:
        raise ValueError(f"no runs found under {root}; start with: beatspy run")
    return max(rows, key=lambda row: row[0])[1]


def _resolve_run(value: str | None) -> Path:
    if value:
        candidate = Path(value).expanduser()
        if candidate.is_dir():
            return candidate
        root = results_dir()
        if (root / value).is_dir():
            return root / value
        matches = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(value)) if root.exists() else []
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"no run matches {value!r} under {root}")
        raise ValueError(f"ambiguous run id {value!r}: {', '.join(p.name for p in matches)}")
    return _latest_run()


def _calibration_of(directory: Path, metrics: dict, lines: list[dict]) -> dict:
    from .dcn.runner import calibration_from_decisions

    calibration = metrics.get("calibration")
    if calibration:
        return calibration
    return calibration_from_decisions(lines).model_dump()


CALIBRATION_COLUMNS = ("signal", "n", "base rate", "brier", "skill", "log loss", "auc", "acc", "ece", "mce")
CALIBRATION_WIDTHS = (9, 6, 11, 9, 9, 10, 7, 7, 9, 9)
CALIBRATION_DASHES = ("-",) * (len(CALIBRATION_COLUMNS) - 2)


def _calibration_row(cells: list[str], *, first_left: bool = False) -> str:
    padded = [
        cell.ljust(width) if (first_left and index == 0) else cell.rjust(width)
        for index, (cell, width) in enumerate(zip(cells, CALIBRATION_WIDTHS, strict=True))
    ]
    return "  " + "".join(padded)


def _calibration_table(calibration: dict) -> str:
    lines = [_calibration_row(list(CALIBRATION_COLUMNS), first_left=True)]
    for signal in ("noul", "choice", "rank"):
        stats = (calibration.get("signals") or {}).get(signal)
        if not stats:
            lines.append(_calibration_row([signal, "0", *CALIBRATION_DASHES], first_left=True) + "  (no observations)")
            continue
        lines.append(
            _calibration_row(
                [
                    signal,
                    str(stats.get("n", 0)),
                    _number(stats.get("base_rate")),
                    _number(stats.get("brier")),
                    _number(stats.get("brier_skill_score")),
                    _number(stats.get("log_loss")),
                    _number(stats.get("auc"), 3),
                    _number(stats.get("accuracy"), 3),
                    _number(stats.get("expected_calibration_error")),
                    _number(stats.get("max_calibration_error")),
                ],
                first_left=True,
            )
        )
    return "\n".join(lines)


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .dcn.runner import read_run

    directory = _resolve_run(args.run)
    meta, metrics, lines = read_run(directory)
    calibration = _calibration_of(directory, metrics, lines)
    decision = meta.get("decision") or {}

    if args.json:
        print(
            json.dumps(
                {
                    "run_id": meta.get("run_id", directory.name),
                    "run_dir": str(directory),
                    "decision_model": decision.get("decision_model"),
                    "provider": decision.get("provider"),
                    "question_form": decision.get("question_form"),
                    "decisions": metrics.get("decisions"),
                    "calibration": calibration,
                },
                indent=2,
                default=str,
            )
        )
        return 0

    print(f"Run            : {meta.get('run_id', directory.name)}")
    print(f"Run dir        : {directory}")
    print(
        f"Decision model : {decision.get('decision_model', '?')} ({decision.get('label', '?')}) | "
        f"provider {decision.get('provider', '?')} | form {decision.get('question_form', '?')}"
    )
    print(
        f"Decisions      : {metrics.get('decisions')} | scored observations {calibration.get('observations', 0)} | "
        f"answer coverage {_percent(calibration.get('coverage'))}"
    )
    print()
    print(_calibration_table(calibration))
    gap = calibration.get("miscalibration_gap")
    print(f"\n  noul-vs-choice miscalibration gap: {_number(gap)}")
    print("  (mean |P_noul - P_choice| on the same ticker and horizon; the paired form gap this benchmark tests)")
    if calibration.get("note"):
        print(f"  note: {calibration['note']}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    directory = _resolve_run(args.run)
    try:
        from .dcn.verify import validate_run
    except ImportError as exc:
        print(f"note: full artifact replay is unavailable ({exc}); running the offline calibration replay check.")
        return _local_verify(directory)

    try:
        meta = validate_run(directory)
    except Exception as exc:
        print(f"FAIL: {directory.name}: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: {meta.get('run_id', directory.name)} — artifacts, decisions, and metrics replay exactly.")
    return 0


def _local_verify(directory: Path) -> int:
    """Fallback replay: rebuild the calibration block from decisions.jsonl and compare."""
    from .artifacts import RUN_FILES, sha256
    from .dcn.runner import calibration_from_decisions, read_run

    try:
        meta, metrics, lines = read_run(directory)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {directory.name}: unreadable run ({exc})", file=sys.stderr)
        return 1

    problems: list[str] = []
    if meta.get("benchmark") != "dcn":
        problems.append(f"run.json benchmark is {meta.get('benchmark')!r}, expected 'dcn'")

    recorded = (meta.get("artifact_hashes") or {})
    for name in RUN_FILES:
        path = directory / name
        if not path.is_file():
            problems.append(f"missing artifact: {name}")
        elif name in recorded and sha256(path) != recorded[name]:
            problems.append(f"artifact hash mismatch: {name}")

    if len(lines) != metrics.get("decisions"):
        problems.append(f"decisions.jsonl has {len(lines)} lines but metrics.json reports {metrics.get('decisions')}")
    dates = [line.get("date") for line in lines]
    if len(set(dates)) != len(dates):
        problems.append("decisions.jsonl contains duplicate decision dates")

    try:
        rebuilt = calibration_from_decisions(lines).model_dump()
    except Exception as exc:
        print(f"FAIL: {directory.name}: calibration could not be rebuilt ({exc})", file=sys.stderr)
        return 1
    stored = metrics.get("calibration")
    if not stored:
        problems.append("metrics.json has no calibration block")
    else:
        for key in ("horizons", "observations"):
            if rebuilt.get(key) != stored.get(key):
                problems.append(f"calibration {key}: recorded {stored.get(key)}, replayed {rebuilt.get(key)}")
        if abs((rebuilt.get("coverage") or 0.0) - (stored.get("coverage") or 0.0)) > 1e-6:
            problems.append("calibration coverage does not replay")
        for signal, stats in rebuilt["signals"].items():
            theirs = (stored.get("signals") or {}).get(signal)
            if theirs is None:
                problems.append(f"calibration signal {signal} missing from metrics.json")
                continue
            for field in ("n", "brier", "brier_skill_score", "log_loss", "auc", "expected_calibration_error"):
                mine, yours = stats.get(field), theirs.get(field)
                if mine is None or yours is None:
                    if mine != yours:
                        problems.append(f"calibration {signal}.{field}: recorded {yours}, replayed {mine}")
                elif abs(float(mine) - float(yours)) > 1e-6:
                    problems.append(f"calibration {signal}.{field}: recorded {yours}, replayed {mine}")

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    print(f"PASS: {meta.get('run_id', directory.name)} — artifacts, decisions, and calibration replay exactly.")
    return 0


def _read_run_row(directory: Path) -> dict | None:
    """One comparably-shaped row per run dir, or None when it is not a run dir."""
    run_json = directory / "run.json"
    metrics_json = directory / "metrics.json"
    if not run_json.is_file() or not metrics_json.is_file():
        return None
    try:
        meta = json.loads(run_json.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"warning: skipping unreadable run {directory.name}: {exc}", file=sys.stderr)
        return None

    decision = meta.get("decision") or {}
    scenario = meta.get("scenario") or {}
    requested = meta.get("requested") or {}
    calibration = metrics.get("calibration") or {}
    noul = (calibration.get("signals") or {}).get("noul") or {}
    pricing = decision.get("pricing") or {}
    return {
        "run_id": meta.get("run_id", directory.name),
        "dir": str(directory),
        "decision_model": decision.get("decision_model", "?"),
        "provider": decision.get("provider", "?"),
        "question_form": decision.get("question_form", "?"),
        "scenario": scenario.get("name", "?"),
        "start": requested.get("start") or scenario.get("start", ""),
        "end": requested.get("end") or scenario.get("end", ""),
        "created_utc": meta.get("created_utc", ""),
        "synthetic": bool(meta.get("synthetic")),
        "cost_per_m_input": pricing.get("cost_per_m_input"),
        "total_return": metrics.get("total_return"),
        "spy_total_return": metrics.get("spy_total_return"),
        "excess_return_vs_spy": metrics.get("excess_return_vs_spy"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_drawdown"),
        "brier": noul.get("brier"),
        "brier_skill_score": noul.get("brier_skill_score"),
        "auc": noul.get("auc"),
        "ece": noul.get("expected_calibration_error"),
        "calibration_gap": calibration.get("miscalibration_gap"),
        "answer_coverage": metrics.get("answer_coverage"),
        "mean_latency_s": metrics.get("mean_latency_s"),
        "estimated_cost_usd": metrics.get("estimated_cost_usd"),
        "decisions": metrics.get("decisions"),
        "trades": metrics.get("trades"),
        "invalid_outputs": metrics.get("invalid_outputs"),
        "input_tokens": metrics.get("input_tokens"),
        "requests": metrics.get("requests"),
        "metrics": metrics,
    }


def cmd_compare(args: argparse.Namespace) -> int:
    root = results_dir()
    rows = []
    if root.exists():
        for directory in sorted(root.iterdir()):
            if not directory.is_dir():
                continue
            row = _read_run_row(directory)
            if row is None:
                continue
            rows.append(row)

    if not args.include_synthetic:
        rows = [row for row in rows if not row["synthetic"]]
    if args.scenario:
        rows = [row for row in rows if row["scenario"] == args.scenario]
    if not rows:
        print(f"No runs found in {root} (synthetic demo runs need --include-synthetic). Try: beatspy run")
        return 1

    latest: dict[str, dict] = {}
    for row in rows:
        current = latest.get(row["decision_model"])
        if current is None or row["created_utc"] > current["created_utc"]:
            latest[row["decision_model"]] = row
    runs = sorted(latest.values(), key=lambda row: -(row["excess_return_vs_spy"] or 0.0))

    if args.json:
        print(json.dumps([{k: v for k, v in row.items() if k != "metrics"} for row in runs], indent=2, default=str))
        return 0

    frame = pd.DataFrame(
        [
            {
                "model": row["decision_model"],
                "provider": row["provider"],
                "form": row["question_form"],
                "scenario": row["scenario"],
                "period": f"{row['start']}..{row['end']}",
                "total_return": _percent(row["total_return"]),
                "spy": _percent(row["spy_total_return"]),
                "excess": _percent(row["excess_return_vs_spy"]),
                "sharpe": _number(row["sharpe"]),
                "max_dd": _percent(row["max_drawdown"]),
                "brier": _number(row["brier"]),
                "skill": _number(row["brier_skill_score"], 3),
                "auc": _number(row["auc"], 3),
                "ece": _number(row["ece"], 3),
                "coverage": _percent(row["answer_coverage"]),
                "cost_usd": _number(row["estimated_cost_usd"], 4),
                "latency_s": _number(row["mean_latency_s"], 2),
                "decisions": row["decisions"],
                "synthetic": row["synthetic"],
            }
            for row in runs
        ]
    )
    print(frame.to_string(index=False))
    print(
        f"\nLatest run per decision model from {root}, sorted by excess return. "
        "Probabilities are scored with Brier, skill, AUC, and ECE on the noul form; "
        "full settings are available with --json."
    )
    return 0


# --------------------------------------------------------------------------- #
# report / demo
# --------------------------------------------------------------------------- #


def cmd_report(args: argparse.Namespace) -> int:
    from .reporting.dashboard import render_dashboard

    path = render_dashboard(results_dir(), run_id=args.run)
    print(f"Dashboard written to {path}")
    print(f"Preview over HTTP: python -m http.server 8000 --bind 127.0.0.1 --directory {path.parent}")
    if args.open:
        webbrowser.open(path.resolve().as_uri())
    return 0


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:48] or "run"


def _demo_decision_days(days: list[date]) -> list[date]:
    """First session plus every month end, so the demo has distinct decision windows."""
    month_ends: dict[tuple[int, int], date] = {}
    for day in days:
        month_ends[(day.year, day.month)] = day
    first = (days[0].year, days[0].month)
    return [days[0], *[day for key, day in month_ends.items() if key != first]]


def _write_demo_runs(seed: int = 42, out_root: Path | None = None) -> list[Path]:
    """Write synthetic demo runs in the real DCN artifact shape. No network, no model.

    The dashboard and `compare` then have something to render on a machine with
    no credentials. Every run is marked synthetic=true, and the probabilities and
    outcomes are fabricated by construction — which is exactly why demo runs are
    excluded from `compare` unless `--include-synthetic` is passed.
    """
    import hashlib

    import numpy as np

    from .artifacts import RUN_FILES, atomic_json, sha256
    from .dcn import DCN_PROTOCOL
    from .dcn.calibration import Observation, build_report
    from .dcn.runner import DISCLAIMER, write_run_artifacts
    from .dcn.scoring import estimate_cost
    from .engine.metrics import (
        annualized_volatility,
        cagr,
        max_drawdown,
        outperformance_hit_rate,
        sharpe_ratio,
        total_return,
        window_returns,
    )
    from .schemas import RunMetrics, Scenario

    out_root = out_root or results_dir()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    created = datetime.now(UTC).isoformat(timespec="seconds")
    days = [timestamp.date() for timestamp in pd.bdate_range("2022-01-03", "2022-06-30")]
    decision_days = _demo_decision_days(days)
    index_days = pd.Index(days, name="date")
    run_dirs: list[Path] = []

    for index, name in enumerate(DEMO_DECISION_MODELS):
        spec = DECISION_MODELS[name]
        settings = Settings(decision=DecisionConfig(decision_model=name))
        rng = np.random.default_rng(seed + index)

        spy_log = rng.normal(-0.0013, 0.012, len(days))
        edge = 0.0009 - 0.0006 * index
        port_log = spy_log + rng.normal(edge, 0.004 + 0.002 * index, len(days))
        portfolio = pd.Series(100_000.0 * np.exp(np.cumsum(port_log)), index=index_days)
        spy_line = pd.Series(100_000.0 * np.exp(np.cumsum(spy_log)), index=index_days)
        baselines = {
            key: pd.Series(100_000.0 * np.exp(np.cumsum(rng.normal(drift, vol, len(days)))), index=index_days)
            for key, drift, vol in (
                ("equal_weight", -0.0008, 0.013),
                ("sixty_forty", -0.0006, 0.008),
                ("momentum_12_1", 0.0002, 0.011),
            )
        }
        equity_df = pd.DataFrame({"portfolio": portfolio, "spy_buy_hold": spy_line, **baselines})

        decision_lines: list[dict] = []
        event_lines: list[dict] = []
        observations: list[Observation] = []
        previous_book: dict[str, float] = {"CASH": 1.0}
        turnovers: list[float] = []

        for day in decision_days:
            probabilities: dict[str, float] = {}
            choice_probabilities: dict[str, float] = {}
            rank_probabilities: dict[str, float] = {}
            answers: dict[str, dict] = {}
            observed: list[dict] = []
            for ticker in DEMO_TICKERS:
                probability = round(float(rng.uniform(0.35, 0.85)), 4)
                # The choice form is deliberately sharper than the noul form, mirroring
                # the overconfidence this benchmark exists to measure.
                sharp = round(min(0.99, max(0.01, (probability - 0.5) * 1.6 + 0.5)), 4)
                rank = round(min(1.0, max(0.0, probability + float(rng.normal(0.0, 0.05)))), 4)
                probabilities[ticker] = probability
                choice_probabilities[ticker] = sharp
                rank_probabilities[ticker] = rank
                answers[f"noul_{ticker}"] = {"type": "noul", "noul": probability}
                outcome = int(rng.random() < probability)
                excess_return = round(float(rng.normal(0.0, 0.05)), 6)
                observed.append(
                    {
                        "ticker": ticker,
                        "noul": probability,
                        "choice": sharp,
                        "rank": rank,
                        "outcome": outcome,
                        "excess_return": excess_return,
                    }
                )
                observations.append(
                    Observation(
                        ticker=ticker,
                        noul=probability,
                        choice=sharp,
                        rank=rank,
                        outcome=outcome,
                        excess_return=excess_return,
                    )
                )

            bar = settings.decision.min_probability
            eligible = sorted(
                ((ticker, value) for ticker, value in probabilities.items() if value > bar),
                key=lambda item: (-item[1], item[0]),
            )[: settings.decision.top_n]
            weights = (
                {ticker: round(1.0 / len(eligible), 4) for ticker, _ in eligible}
                if len(eligible) >= settings.decision.min_names
                else {}
            )
            cash = round(1.0 - sum(weights.values()), 4)
            book = {**weights, "CASH": cash}
            moved = sum(
                abs(book.get(ticker, 0.0) - previous_book.get(ticker, 0.0)) for ticker in {*book, *previous_book}
            )
            turnovers.append(0.5 * moved)
            previous_book = book

            state_sha = hashlib.sha256(f"demo:{name}:{day.isoformat()}".encode()).hexdigest()
            question_ids = [f"noul_{ticker}" for ticker in DEMO_TICKERS]
            decision_lines.append(
                {
                    "date": day.isoformat(),
                    "question_form": "twin",
                    "tickers": list(DEMO_TICKERS),
                    "state_sha256": state_sha,
                    "state_chars": 1200,
                    "questions": {
                        f"noul_{ticker}": {
                            "type": "noul",
                            "instructions": {"question": "synthetic demo decision; no model was called"},
                        }
                        for ticker in DEMO_TICKERS
                    },
                    "answers": answers,
                    "missing_ids": [],
                    "backfilled": [],
                    "probabilities": probabilities,
                    "choice_probabilities": choice_probabilities,
                    "rank_probabilities": rank_probabilities,
                    "validated": {"weights": weights, "cash": cash, "violations": [], "invalid": False},
                    "coverage": 1.0,
                    "observed": observed,
                    "call": {
                        "provider": spec["provider"],
                        "model": spec["model"],
                        "endpoint": spec["base_url"],
                        "state_sha256": state_sha,
                        "state_chars": 1200,
                        "question_ids": question_ids,
                        "answers": answers,
                        "missing_ids": [],
                        "input_tokens": int(rng.integers(900, 1600)),
                        "output_tokens": 0,
                        "requests": 1,
                        "latency_s": 0.0,
                        "attempts": 1,
                        "error": None,
                    },
                    "error": None,
                }
            )
            event_lines.append(
                {
                    "type": "decision",
                    "date": day.isoformat(),
                    "weights": weights,
                    "cash": cash,
                    "coverage": 1.0,
                    "violations": [],
                    "error": None,
                }
            )

        input_tokens = sum(line["call"]["input_tokens"] for line in decision_lines)
        calibration = build_report(
            observations,
            coverage=1.0,
            horizons=len(decision_lines),
            note="synthetic demo: fabricated probabilities and outcomes, no model was called",
        )
        metrics = RunMetrics(
            total_return=round(total_return(portfolio), 6),
            cagr=round(cagr(portfolio), 6),
            annual_volatility=round(annualized_volatility(portfolio.pct_change().dropna()), 6),
            sharpe=round(sharpe_ratio(portfolio.pct_change().dropna(), settings.bench.risk_free_annual), 4),
            max_drawdown=round(max_drawdown(portfolio), 6),
            spy_total_return=round(total_return(spy_line), 6),
            excess_return_vs_spy=round(total_return(portfolio) - total_return(spy_line), 6),
            outperformance_hit_rate=round(
                outperformance_hit_rate(
                    window_returns(portfolio, decision_days), window_returns(spy_line, decision_days)
                ),
                4,
            ),
            decisions=len(decision_lines),
            avg_turnover=round(sum(turnovers) / len(turnovers), 6) if turnovers else 0.0,
            trades=0,
            invalid_outputs=0,
            risk_violations=0,
            input_tokens=input_tokens,
            output_tokens=0,
            requests=len(decision_lines),
            calibration=calibration,
            answer_coverage=1.0,
            mean_latency_s=0.0,
            estimated_cost_usd=estimate_cost(settings, input_tokens, 0),
            baselines={key: round(total_return(values), 6) for key, values in baselines.items()},
        )

        scenario_block = Scenario(
            name="2022-bear",
            title="2022 bear market (synthetic demo window)",
            description="Synthetic demo window written by `beatspy demo`; not the bundled 2022-bear scenario.",
            start=days[0].isoformat(),
            end=days[-1].isoformat(),
            frequency="monthly",
            benchmark="SPY",
            tradable=list(DEMO_TICKERS),
        )
        run_id = f"{stamp}_2022-bear_{_slug(name)}_demo{index}"
        run_meta = {
            "schema_version": 3,
            "benchmark": "dcn",
            "dcn_protocol": DCN_PROTOCOL,
            "source_revision": "demo",
            "run_id": run_id,
            "label": "demo",
            "beatspy_version": __version__,
            "created_utc": created,
            "synthetic": True,
            "disclaimer": DISCLAIMER,
            "decision": {
                "decision_model": settings.decision.decision_model,
                "provider": settings.decision.provider,
                "label": settings.decision.label,
                "endpoint": settings.decision.endpoint,
                "question_form": "twin",
                "pricing": {
                    "cost_per_m_input": settings.decision.cost_per_m_input,
                    "cost_per_m_output": settings.decision.cost_per_m_output,
                },
                "policy": {
                    "min_probability": settings.decision.min_probability,
                    "top_n": settings.decision.top_n,
                    "min_names": settings.decision.min_names,
                },
            },
            "scenario": scenario_block.model_dump(),
            "requested": {"frequency": "monthly", "start": days[0].isoformat(), "end": days[-1].isoformat()},
            "data": {
                "snapshot_id": "synthetic",
                "tickers": scenario_block.universe,
                "start": days[0].isoformat(),
                "end": days[-1].isoformat(),
                "rows": 0,
                "fetched_at_utc": created,
                "source": "synthetic demo series (no market data was downloaded)",
                "prices_sha256": "synthetic",
            },
            "bench": settings.bench.model_dump(),
            "note": (
                "Synthetic demo run generated by `beatspy demo`. No decision model was called and "
                "no market data was downloaded."
            ),
        }

        out_dir = out_root / run_id
        write_run_artifacts(out_dir, run_meta, metrics.model_dump(), equity_df, [], decision_lines, event_lines)
        run_meta["artifact_hashes"] = {file_name: sha256(out_dir / file_name) for file_name in RUN_FILES}
        atomic_json(out_dir / "run.json", run_meta)
        run_dirs.append(out_dir)
        log.info("wrote synthetic demo run %s", out_dir)
    return run_dirs


def cmd_demo(args: argparse.Namespace) -> int:
    dirs = _write_demo_runs(seed=args.seed)
    print(f"Generated {len(dirs)} SYNTHETIC demo runs under {results_dir()}")
    for directory in dirs:
        print(f"  {directory.name}")
    try:
        from .reporting.dashboard import render_dashboard

        path = render_dashboard(results_dir())
        print(f"Dashboard written to {path}")
    except Exception as exc:  # a broken dashboard must not discard generated runs
        print(f"Dashboard not rendered: {exc}", file=sys.stderr)
    print("Demo runs are fabricated, so compare hides them unless you pass --include-synthetic.")
    return 0


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # httpx request lines include full URLs with query strings and are pure noise at INFO.
    for noisy in ("httpx", "httpx2"):
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