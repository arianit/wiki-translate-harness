from pathlib import Path

import pytest

from wiki_translate_harness.config import build_config

_CONTACT = {"wikimedia_contact": "test@example.com"}


def test_missing_api_key_raises(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No OpenRouter API key"):
        build_config(None, _CONTACT)


def test_env_var_supplies_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cfg = build_config(None, _CONTACT)
    assert cfg.openrouter_api_key == "sk-test"


def test_yaml_config_loaded(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "provider: openrouter\nmodel: qwen/qwen3-235b-a22b\nworkers: 8\nopenrouter_api_key: sk-yaml\n"
    )
    cfg = build_config(config_file, _CONTACT)
    assert cfg.model == "qwen/qwen3-235b-a22b"
    assert cfg.workers == 8
    assert cfg.openrouter_api_key == "sk-yaml"


def test_cli_overrides_win_over_yaml(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config_file = tmp_path / "config.yaml"
    config_file.write_text("model: deepseek/deepseek-v3\nworkers: 4\nopenrouter_api_key: sk-yaml\n")
    cfg = build_config(config_file, {**_CONTACT, "model": "google/gemini-2.5-flash", "workers": 16})
    assert cfg.model == "google/gemini-2.5-flash"
    assert cfg.workers == 16


def test_env_var_overrides_yaml_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("openrouter_api_key: sk-yaml\n")
    cfg = build_config(config_file, _CONTACT)
    assert cfg.openrouter_api_key == "sk-env"


def test_validate_alias_from_yaml(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("validate: false\n")
    cfg = build_config(config_file, _CONTACT)
    assert cfg.validate_output is False


def test_missing_wikimedia_contact_raises(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    monkeypatch.delenv("WIKIMEDIA_CONTACT", raising=False)
    with pytest.raises(ValueError, match="User-Agent policy"):
        build_config(None, {})


def test_wikimedia_contact_env_var(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    monkeypatch.setenv("WIKIMEDIA_CONTACT", "env-contact@example.com")
    cfg = build_config(None, {})
    assert "env-contact@example.com" in cfg.user_agent


def test_user_agent_computed_from_contact(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    cfg = build_config(None, {"wikimedia_contact": "someone@example.com", "wikimedia_tool_name": "mybot"})
    assert cfg.user_agent.startswith("mybot/")
    assert "someone@example.com" in cfg.user_agent


def test_explicit_user_agent_bypasses_contact_requirement(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    monkeypatch.delenv("WIKIMEDIA_CONTACT", raising=False)
    cfg = build_config(None, {"user_agent": "custom-ua/1.0 (custom@example.com)"})
    assert cfg.user_agent == "custom-ua/1.0 (custom@example.com)"
