import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
from zipfile import ZipFile

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load_packager(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    spec = importlib.util.spec_from_file_location("package_kaggle_hybrids", EXAMPLES / "package_kaggle_hybrids.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_input_manifest_names_every_file_where_kaggle_unpacks_the_archives(tmp_path, monkeypatch):
    with ZipFile(tmp_path / "hybrids-source.zip", "w") as archive:
        archive.writestr("examples/kaggle_hybrids.py", "print('hi')\n")
        archive.writestr("src/pkg/__init__.py", "")
    with ZipFile(tmp_path / "hybrids-wheels.zip", "w") as archive:
        archive.writestr("pkg-1.0-py3-none-any.whl", "wheel bytes")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "README.md").write_text("read me")

    manifest = load_packager(monkeypatch).input_manifest(tmp_path)

    # Kaggle replaces each uploaded .zip with a folder of the same name.
    assert set(manifest) == {
        "model.safetensors", "config.json", "README.md",
        "hybrids-source/examples/kaggle_hybrids.py", "hybrids-source/src/pkg/__init__.py",
        "hybrids-wheels/pkg-1.0-py3-none-any.whl",
    }
    assert manifest["model.safetensors"] == hashlib.sha256(b"weights").hexdigest()
    assert manifest["hybrids-source/examples/kaggle_hybrids.py"] == hashlib.sha256(b"print('hi')\n").hexdigest()
    assert manifest["hybrids-wheels/pkg-1.0-py3-none-any.whl"] == hashlib.sha256(b"wheel bytes").hexdigest()


def test_notebook_reads_the_inputs_kaggle_actually_stores(tmp_path, monkeypatch):
    packager = load_packager(monkeypatch)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    inputs = tmp_path / "input"
    files = {"hybrids-source/examples/kaggle_hybrids.py": b"print('hi')\n", "hybrids-wheels/pkg.whl": b"wheel",
             "config.json": b"{}", "README.md": b"read me", "model.safetensors": b"weights"}
    for name, content in files.items():
        (inputs / name).parent.mkdir(parents=True, exist_ok=True)
        (inputs / name).write_bytes(content)
    (inputs / "hybrids-inputs.json").write_text(json.dumps(
        {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    ))
    setup = packager.BOOTSTRAP.split("\ncheckpoint = work / 'chronos-2'")[0].replace("'/kaggle/input'", repr(str(inputs)))

    namespace = {}
    exec(setup, namespace)

    assert (namespace["source"] / "examples/kaggle_hybrids.py").read_bytes() == b"print('hi')\n"
    (inputs / "hybrids-source/examples/kaggle_hybrids.py").write_bytes(b"tampered\n")
    with pytest.raises(RuntimeError, match="Input hash mismatch for hybrids-source/examples/kaggle_hybrids.py"):
        exec(setup, {})


def test_notebook_import_check_runs_in_a_fresh_interpreter(tmp_path, monkeypatch, capsys):
    # The check runs through `python -c`, so it can only use names it imports itself.
    snippet = ast.literal_eval("(" + load_packager(monkeypatch).BOOTSTRAP.split("'-c', ")[1].split("],\n")[0] + ")")
    for name in ("torch", "chronos", "exchange_calendars", "lightgbm", "sklearn", "scipy", "pydantic", "httpx", "transformers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setattr(sys.modules["torch"], "__version__", "0-torch", raising=False)
    monkeypatch.setattr(sys.modules["transformers"], "__version__", "0-transformers", raising=False)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: f"0-{name}")

    exec(snippet, {})

    assert capsys.readouterr().out.split()[:3] == [sys.version.split()[0], "0-torch", "0-transformers"]


def test_cpu_notebook_reuses_inputs_with_gpu_and_internet_disabled(tmp_path, monkeypatch):
    packager = load_packager(monkeypatch)
    packager.notebook(tmp_path, "local-owner", device="cpu")
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["enable_gpu"] is False and meta["enable_internet"] is False
    assert meta["dataset_sources"] == ["local-owner/beatspy-hybrid-inputs"]
    cells = json.loads((tmp_path / meta["code_file"]).read_text())["cells"]
    execution = cells[1]["source"]
    assert "'--device', 'cpu'" in execution
    assert "manifest['hybrids-source/examples/kaggle_hybrids.py']" in execution
    for cell in cells:
        compile(cell["source"], "<kaggle-notebook>", "exec")
