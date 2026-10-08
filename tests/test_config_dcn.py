"""Credentials: where they come from, and the fact that `.env` is honoured.

A `.env` in the working directory is what people expect to work, so it is read —
but only after a real exported variable, and nothing here may raise when it is
missing or malformed.
"""

from __future__ import annotations

import pytest

from beatspy.config import (
    DECISION_MODELS,
    decision_api_key,
    load_secrets,
    parse_env_file,
    read_env_file,
)


def test_parse_env_file_handles_comments_quotes_and_inline_notes():
    parsed = parse_env_file(
        "\n".join(
            [
                "# a comment",
                "",
                "PLAIN=value",
                'QUOTED="quoted value"',
                "SINGLE='single'",
                "TRAILING=value   # from .env (kept)",
                "EMPTY=",
                "NOT_A_PAIR",
                "URL=https://example.com/v1",
            ]
        )
    )
    assert parsed["PLAIN"] == "value"
    assert parsed["QUOTED"] == "quoted value"
    assert parsed["SINGLE"] == "single"
    assert parsed["TRAILING"] == "value", "an inline note must not become part of the value"
    assert parsed["EMPTY"] == ""
    assert parsed["URL"] == "https://example.com/v1"
    assert "NOT_A_PAIR" not in parsed


def test_missing_env_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert read_env_file() == {}


def test_env_file_is_loaded_into_the_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("BEATSPY_TEST_TOKEN=from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("BEATSPY_TEST_TOKEN", raising=False)
    load_secrets()
    assert __import__("os").environ["BEATSPY_TEST_TOKEN"] == "from-dotenv"


def test_an_exported_variable_beats_the_env_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("BEATSPY_TEST_TOKEN=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("BEATSPY_TEST_TOKEN", "from-environment")
    load_secrets()
    assert __import__("os").environ["BEATSPY_TEST_TOKEN"] == "from-environment"


def test_an_empty_value_in_the_env_file_does_not_erase_a_real_one(tmp_path, monkeypatch):
    """A placeholder line like `OPENAI_API_KEY=` must not blank out a live key."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("BEATSPY_TEST_TOKEN=\n", encoding="utf-8")
    monkeypatch.setenv("BEATSPY_TEST_TOKEN", "real")
    load_secrets()
    assert __import__("os").environ["BEATSPY_TEST_TOKEN"] == "real"


def test_env_file_does_not_override_the_secrets_file(tmp_path, monkeypatch):
    """~/.beatspy/secrets.env wins over the project .env when both define a key.

    `BEATSPY_HOME` is the secrets directory itself, so the file lives directly
    inside it as `secrets.env`.
    """
    home = tmp_path / "beatspy-home"
    home.mkdir()
    (home / "secrets.env").write_text("BEATSPY_TEST_TOKEN=from-secrets\n", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    (work / ".env").write_text("BEATSPY_TEST_TOKEN=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("BEATSPY_HOME", str(home))
    monkeypatch.chdir(work)
    monkeypatch.delenv("BEATSPY_TEST_TOKEN", raising=False)
    load_secrets()
    assert __import__("os").environ["BEATSPY_TEST_TOKEN"] == "from-secrets"


def test_each_decision_model_reads_its_own_key_variable(monkeypatch):
    for env in ("OPENAI_API_KEY", "BEATSPY_OPENAI_API_KEY", "TYPESAFE_API_KEY",
                "BEATSPY_TYPESAFE_API_KEY", "CLOUDFLARE_AUTH_TOKEN", "BEATSPY_CLOUDFLARE_AUTH_TOKEN"):
        monkeypatch.delenv(env, raising=False)
    for name in DECISION_MODELS:
        assert decision_api_key(name) is None

    monkeypatch.setenv("TYPESAFE_API_KEY", "jev-key")
    assert decision_api_key("jev-latest") == "jev-key"
    assert decision_api_key("gpt-6-luna") is None  # a provider key is not shared

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert decision_api_key("gpt-6-luna") == "openai-key"

    # Clef and Clef-flash share the Cloudflare token.
    monkeypatch.setenv("CLOUDFLARE_AUTH_TOKEN", "cf-token")
    assert decision_api_key("clef") == "cf-token"
    assert decision_api_key("clef-flash") == "cf-token"


def test_cloudflare_needs_an_account_id_as_well_as_a_token(monkeypatch):
    from beatspy.config import cloudflare_account_id

    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    assert cloudflare_account_id() is None
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    assert cloudflare_account_id() == "acct"
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID")
    monkeypatch.setenv("BEATSPY_CLOUDFLARE_ACCOUNT_ID", "prefixed")
    assert cloudflare_account_id() == "prefixed"


def test_no_credential_means_none_not_an_exception(monkeypatch):
    """Running offline is a supported state, not an error."""
    for name in DECISION_MODELS:
        for env in (DECISION_MODELS[name]["api_key_env"], f"BEATSPY_{DECISION_MODELS[name]['api_key_env']}"):
            monkeypatch.delenv(env, raising=False)
    for name in DECISION_MODELS:
        assert decision_api_key(name) is None


@pytest.mark.parametrize("name", sorted(DECISION_MODELS))
def test_registry_entries_are_complete(name):
    spec = DECISION_MODELS[name]
    for key in ("provider", "model", "label", "base_url", "api_key_env", "cost_per_m_input",
                "cost_per_m_output", "context_window", "max_questions"):
        assert key in spec, f"{name} is missing {key}"
    assert spec["cost_per_m_output"] == 0.0, "decision models bill input only"