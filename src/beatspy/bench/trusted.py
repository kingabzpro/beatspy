"""GitHub-only controlled rerun; request data never selects code or arbitrary endpoints."""

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ..config import ModelConfig, Settings, ToolsConfig, provider_api_key
from ..reporting.catalog import build_catalog, submit_run
from ..scenarios import load_scenario
from .runner import run_benchmark, source_revision
from .verification import code_digest, sign_result, verify_result


def require_trusted_checkout():
    ref = os.environ.get("GITHUB_WORKFLOW_REF", "")
    if not ref.endswith("/.github/workflows/trusted-benchmark.yml@refs/heads/main"):
        raise ValueError("signing requires the trusted main-branch workflow")
    if os.environ.get("GITHUB_SHA") != source_revision():
        raise ValueError("trusted checkout revision mismatch")
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", "src", "pyproject.toml", "uv.lock"],
        text=True,
    )
    if changed.strip():
        raise ValueError("trusted benchmark code has been modified")


def request_model(event: dict) -> str:
    if "issue" in event:
        match = re.search(r"```json\s*(.*?)\s*```", event["issue"].get("body", ""), re.DOTALL)
        if not match:
            raise ValueError("request must contain a JSON benchmark request")
        request = json.loads(match[1])
        if request.get("benchmark") != "2026-ytd":
            raise ValueError("only 2026-ytd requests are accepted")
        return request["model"]
    return event["inputs"]["model"]


def main():
    require_trusted_checkout()
    if sys.argv[1] == "run":
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
        model = request_model(event)
        preset = json.loads(Path(".github/benchmark-models.json").read_text(encoding="utf-8"))[model]
        if not os.environ.get(preset["api_key_env"]):
            raise ValueError(f"missing configured model credential: {preset['api_key_env']}")
        os.environ["BEATSPY_API_KEY_ENV"] = preset["api_key_env"]
        settings = Settings(
            model=ModelConfig(
                model=model,
                base_url=preset["base_url"],
                max_turns=16,
                reasoning_effort=preset.get("reasoning_effort"),
                cost_per_m_input=preset.get("cost_per_m_input"),
                cost_per_m_output=preset.get("cost_per_m_output"),
            )
        )
        settings.tools = ToolsConfig(
            forecast_provider="timegpt" if provider_api_key(settings, "timegpt") else "baseline"
        )
        before = code_digest()
        run = asyncio.run(run_benchmark(load_scenario("2026-ytd"), settings))
        require_trusted_checkout()
        if code_digest() != before:
            raise ValueError("benchmark code changed during execution")
        target = submit_run(run, Path("dashboard/data"))
        Path("results/trusted-result.txt").write_text(str(target), encoding="utf-8")
    elif sys.argv[1] == "sign":
        target = Path("results/trusted-result.txt").read_text(encoding="utf-8")
        directory = Path(target)
        if directory.resolve().parent != Path("dashboard/data/runs").resolve():
            raise ValueError("invalid trusted result location")
        private_pem = os.environ["BEATSPY_SIGNING_KEY"].encode()
        public_pem = Path(".github/verification-key.pem").read_bytes()
        key = serialization.load_pem_private_key(private_pem, password=None)
        public = serialization.load_pem_public_key(public_pem)
        if not isinstance(key, Ed25519PrivateKey) or key.public_key().public_bytes_raw() != public.public_bytes_raw():
            raise ValueError("signing key does not match the repository's pinned public key")
        execution = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        sign_result(directory, private_pem, execution)
        verify_result(directory, public_pem)
        build_catalog(Path("dashboard/data"))
    else:
        raise ValueError("expected run or sign")


if __name__ == "__main__":
    main()
