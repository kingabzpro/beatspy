"""BeatSPY as a Kaggle Benchmark task: any Model Proxy LLM scored on the three published windows.

Local:  uv run python examples/kbench_beatspy.py  (needs .env from `kaggle b init`)
Push:   kaggle b t push beatspy -f examples/kbench_beatspy.py --wait
Run:    kaggle b t run beatspy -m <model-slug> --wait
"""

# %%
import asyncio
import json
import subprocess
import sys
from pathlib import Path

# %%
# The SDK's dotenv lookup misses .env when this file runs as a script; load it from the cwd first
if (Path.cwd() / ".env").is_file():
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env", override=True)

import kaggle_benchmarks as kbench  # noqa: E402

REPOSITORY_URL = "https://github.com/kingabzpro/beatspy.git"
# ponytail: pins to main; set to the release commit here before a push that must be reproducible
BEATSPY_REF = "main"

# %%
def _repository() -> Path:
    """The beatspy checkout to run against: local editable install, else a fresh clone."""
    try:
        import beatspy as pkg

        local = Path(pkg.__file__).resolve().parents[2]
        if (local / "examples" / "published_windows.py").is_file():
            return local
    except ModuleNotFoundError:
        pass
    working = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
    clone = working / "beatspy-repo"
    if not (clone / "examples" / "published_windows.py").is_file():
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", "--branch", BEATSPY_REF, REPOSITORY_URL, str(clone)],
            check=True,
        )
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", str(clone)], check=True)
    return clone


# %%
REPOSITORY = _repository()
sys.path.insert(0, str(REPOSITORY / "examples"))

from published_windows import benchmark  # noqa: E402

from beatspy.config import DecisionConfig, Settings  # noqa: E402
from beatspy.dcn import DCN_PROTOCOL  # noqa: E402
from beatspy.dcn.kbench import KbenchClient, register  # noqa: E402


# %%
@kbench.task(name="beatspy")
def beatspy_decisions(llm) -> dict:
    slug = register(getattr(llm, "model", None) or getattr(llm, "name", None) or "kaggle-llm")
    settings = Settings(decision=DecisionConfig(decision_model=slug))
    output = Path("/kaggle/working/beatspy-results")
    if not output.parent.is_dir():
        output = REPOSITORY / "results" / "kbench-kaggle" / "runs"
    runtime = {"model": slug, "transport": "kaggle-model-proxy", "dcn_protocol": DCN_PROTOCOL}

    def client(data):
        return KbenchClient(llm, model=slug)

    asyncio.run(benchmark(output, settings, client, runtime))

    windows = json.loads((output / "summary.json").read_text())["windows"]
    for window in windows:
        kbench.assertions.assert_true(
            window["answer_coverage"] == 1.0,
            expectation=f"{window['window']} answers every question",
        )
    return {
        window["window"]: {
            "brier": window["brier"],
            "excess_return_vs_spy": window["excess_return_vs_spy"],
        }
        for window in windows
    }


# %%
beatspy_decisions.run(kbench.llm)
