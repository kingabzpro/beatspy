"""Config, secrets, and precedence tests."""

from __future__ import annotations

import pytest

from beatspy.config import (
    Settings,
    load_secrets,
    load_settings,
    render_config,
    save_settings,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("BEATSPY_HOME", str(tmp_path))
    return tmp_path


def test_roundtrip(home):
    settings = Settings()
    settings.model.base_url = "http://localhost:11434/v1"
    settings.model.model = "llama3.1:8b"
    save_settings(settings, {"BEATSPY_API_KEY": "sk-test"})
    loaded = load_settings()
    assert loaded.model.base_url == "http://localhost:11434/v1"
    assert loaded.model.model == "llama3.1:8b"


def test_secrets_load_and_priority(home, monkeypatch):
    save_settings(Settings(), {"BEATSPY_API_KEY": "secret-1"})
    monkeypatch.delenv("BEATSPY_API_KEY", raising=False)
    load_secrets()
    import os

    assert os.environ["BEATSPY_API_KEY"] == "secret-1"
    # existing environment variables win over secrets.env
    monkeypatch.setenv("BEATSPY_API_KEY", "env-wins")
    monkeypatch.delenv("BEATSPY_API_KEY_CHECK", raising=False)
    load_secrets()
    assert os.environ["BEATSPY_API_KEY"] == "env-wins"


def test_env_overrides(home, monkeypatch):
    monkeypatch.setenv("BEATSPY_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("BEATSPY_BASE_URL", "http://localhost:8000/v1")
    loaded = load_settings()
    assert loaded.model.model == "qwen2.5:7b"
    assert loaded.model.base_url == "http://localhost:8000/v1"


def test_render_config_shapes(home):
    settings = Settings()
    text = render_config(settings)
    assert "[model]" in text and "[bench]" in text and "[tools]" in text
    assert "[agents]" not in text


def test_defaults_when_missing(home):
    loaded = load_settings()
    assert loaded == Settings()
