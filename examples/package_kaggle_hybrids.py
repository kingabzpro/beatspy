"""Package the hybrid benchmark and offline Kaggle notebook from local input files."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from tpi_variants import WINDOWS

REPO = Path(__file__).resolve().parents[1]
INPUT_NAME = "beatspy-hybrid-inputs"
NOTEBOOK_NAME = "beatspy-offline-hybrids"
EXAMPLES = ("kaggle_hybrids.py", "published_windows.py", "tpi_variants.py")

# Kaggle unpacks every uploaded zip archive into a folder of the same name, so the
# notebook reads those folders and checks each file against its own hash.
BOOTSTRAP = '''import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

manifests = list(Path('/kaggle/input').rglob('hybrids-inputs.json'))
if len(manifests) != 1:
    raise RuntimeError('Attach the BeatSPY Hybrid Inputs dataset to this notebook.')
inputs = manifests[0].parent
manifest = json.loads(manifests[0].read_text())
for name, expected in manifest.items():
    with (inputs / name).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise RuntimeError(f'Input hash mismatch for {name}')

work = Path(tempfile.mkdtemp(prefix='beatspy-hybrids-'))
source = work / 'source'
shutil.copytree(inputs / 'hybrids-source', source, copy_function=shutil.copyfile)
checkpoint = work / 'chronos-2'
checkpoint.mkdir()
for name in ('config.json', 'model.safetensors'):
    (checkpoint / name).symlink_to(inputs / name)

env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
       'PIP_NO_INDEX': '1', 'PYTHONPATH': str(source / 'src')}
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                '--find-links', str(inputs / 'hybrids-wheels'), 'chronos-forecasting==2.3.2',
                'exchange-calendars==4.13.2', 'scikit-learn==1.9.1', 'lightgbm==4.6.0',
                'accelerate', 'safetensors', 'toolz', 'pyluach', 'korean-lunar-calendar', 'tzdata'],
               env=env, check=True)
subprocess.run([sys.executable, '-c', 'import sys, torch, chronos, exchange_calendars, lightgbm, sklearn, scipy, '
                'pydantic, httpx, transformers, importlib.metadata; '
                'print(sys.version.split()[0], torch.__version__, transformers.__version__, '
                '{name: importlib.metadata.version(name) for name in ("scipy", "scikit-learn", "lightgbm", '
                '"chronos-forecasting", "numpy", "pandas", "pydantic", "httpx")})'],
               cwd=source, env=env, check=True)
'''

EXECUTE = '''results = work / 'results'
source_sha256 = hashlib.sha256(json.dumps({name: digest for name, digest in manifest.items()
                                           if name.startswith('hybrids-source/')}, sort_keys=True).encode()).hexdigest()
try:
    subprocess.run([sys.executable, 'examples/kaggle_hybrids.py', '--checkpoint', str(checkpoint),
                    '--source-sha256', source_sha256, '--output', str(results)],
                   cwd=source, env=env, check=True)
finally:
    if results.is_dir():
        shutil.make_archive('/kaggle/working/hybrid-results', 'zip', results)
'''


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def input_manifest(output: Path) -> dict:
    """Hash every input as Kaggle stores it: an uploaded archive becomes a folder of the same name."""
    manifest = {}
    for name in ("config.json", "model.safetensors", "README.md"):
        with (output / name).open("rb") as source:
            manifest[name] = hashlib.file_digest(source, "sha256").hexdigest()
    for stem in ("hybrids-source", "hybrids-wheels"):
        with ZipFile(output / f"{stem}.zip") as archive:
            manifest.update({f"{stem}/{info.filename}": hashlib.sha256(archive.read(info)).hexdigest()
                             for info in archive.infolist()})
    return manifest


def notebook(output: Path, owner: str, *, device: str = "cuda") -> None:
    output.mkdir(parents=True, exist_ok=True)
    notebook_name = f"{NOTEBOOK_NAME}.ipynb"
    write_json(output / "kernel-metadata.json", {
        "id": f"{owner}/{NOTEBOOK_NAME}", "title": "BeatSPY Offline Hybrids",
        "code_file": notebook_name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": device == "cuda", "enable_internet": False,
        "dataset_sources": [f"{owner}/{INPUT_NAME}"], "model_sources": [],
        "competition_sources": [], "kernel_sources": [],
    })
    execute = EXECUTE
    if device == "cpu":
        # Reuse the uploaded model and data, while including the current CPU
        # runner in both the notebook and the source fingerprint of the run.
        runner = (REPO / "examples/kaggle_hybrids.py").read_bytes()
        encoded = base64.b64encode(runner).decode("ascii")
        execute = (
            "import base64\n"
            f"runner = base64.b64decode({encoded!r})\n"
            "(source / 'examples/kaggle_hybrids.py').write_bytes(runner)\n"
            "manifest['hybrids-source/examples/kaggle_hybrids.py'] = hashlib.sha256(runner).hexdigest()\n"
            + EXECUTE.replace(
                "'--source-sha256', source_sha256,", "'--device', 'cpu', '--source-sha256', source_sha256,",
            )
        )
    write_json(output / notebook_name, {
        "nbformat": 4, "nbformat_minor": 4,
        "metadata": {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}},
        "cells": [
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}
            for source in (BOOTSTRAP, execute)
        ],
    })


def bundle(output: Path, owner: str, checkpoint: Path, wheels: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    paths = [path for path in (REPO / "src/beatspy").rglob("*")
             if path.is_file() and path.suffix in (".py", ".toml", ".json") and "__pycache__" not in path.parts]
    paths += [REPO / "examples" / name for name in EXAMPLES]
    paths += [REPO / "dashboard/data/index.json"]
    for window in WINDOWS.values():
        paths.extend(REPO / window["snapshot"] / name for name in ("MANIFEST.json", "prices.csv"))
    with ZipFile(output / "hybrids-source.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(REPO))
    extras = list(wheels.glob("*.whl"))
    if not extras:
        raise ValueError("Provide the offline Kaggle/Linux wheels")
    with ZipFile(output / "hybrids-wheels.zip", "w") as archive:
        for wheel in sorted(extras):
            archive.write(wheel, wheel.name)
    for name in ("config.json", "model.safetensors", "README.md"):
        shutil.copyfile(checkpoint / name, output / name)
    write_json(output / "hybrids-inputs.json", input_manifest(output))
    write_json(output / "dataset-metadata.json", {
        "title": "BeatSPY Hybrid Inputs", "id": f"{owner}/{INPUT_NAME}", "licenses": [{"name": "Apache 2.0"}],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--wheels", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    notebook(args.output / "upload", args.owner, device=args.device)
    if args.checkpoint and args.wheels:
        bundle(args.output / "dataset", args.owner, args.checkpoint, args.wheels)
