"""Load config.yaml and merge in CLI / environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from wiki_translation_harness import __version__
from wiki_translation_harness.models import Config

_ENV_API_KEY = "OPENROUTER_API_KEY"
_ENV_LOCAL_API_KEY = "LOCAL_API_KEY"
_ENV_EXPERIENTIAL_API_KEY = "EXPLABS_API_KEY"
_ENV_WIKIMEDIA_CONTACT = "WIKIMEDIA_CONTACT"


def default_model_for_provider(provider: str) -> str:
    """Same provider-appropriate default build_config() picks when nothing
    else sets `model` — factored out so a mid-run fallback-provider switch
    (pipeline.py) can pick a sane model for the new provider too."""
    if provider == "claude_code":
        return "claude-sonnet-5"
    if provider == "experiential":
        # Matches Experiential Labs' own curated-catalog example model id
        # (platform.experientiallabs.ai/docs) rather than assuming its
        # catalog carries OpenRouter's deepseek/deepseek-v3.2 slug.
        return "qwen3.8-27b"
    return "deepseek/deepseek-v3.2"


def load_yaml_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must contain a YAML mapping at the top level")
    return data


def build_config(
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Merge config.yaml < environment < explicit CLI overrides (highest priority)."""

    data = load_yaml_config(config_path)

    api_key = os.environ.get(_ENV_API_KEY)
    if api_key:
        data["openrouter_api_key"] = api_key

    local_api_key = os.environ.get(_ENV_LOCAL_API_KEY)
    if local_api_key:
        data["local_api_key"] = local_api_key

    experiential_api_key = os.environ.get(_ENV_EXPERIENTIAL_API_KEY)
    if experiential_api_key:
        data["experiential_api_key"] = experiential_api_key

    contact_env = os.environ.get(_ENV_WIKIMEDIA_CONTACT)
    if contact_env:
        data["wikimedia_contact"] = contact_env

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                data[key] = value

    # If the CLI explicitly switched --provider without also passing
    # --model, a model id left over in config.yaml for the *previous*
    # provider must not silently survive the switch. Confirmed necessary in
    # practice: `--provider claude_code` alone, against a config.yaml with
    # `provider: openrouter` and an OpenRouter model id, sent that model id
    # straight to the Claude Code CLI, which rejected it as
    # claude-code:unrecognized_model.
    overrides = overrides or {}
    if overrides.get("provider") is not None and overrides.get("model") is None:
        data.pop("model", None)

    # Pick a provider-appropriate default model when nothing set one (in
    # config.yaml, or via --model, or above) -- same reasoning as above,
    # just covering the case where config.yaml never had a model at all.
    if "model" not in data:
        data["model"] = default_model_for_provider(data.get("provider", "claude_code"))

    config = Config.model_validate(data)

    if config.provider == "openrouter" and not config.openrouter_api_key:
        raise ValueError(
            f"No OpenRouter API key configured. Set the {_ENV_API_KEY} environment "
            "variable or 'openrouter_api_key' in config.yaml."
        )

    if config.provider == "experiential" and not config.experiential_api_key:
        raise ValueError(
            f"No Experiential Labs API key configured. Set the {_ENV_EXPERIENTIAL_API_KEY} "
            "environment variable or 'experiential_api_key' in config.yaml."
        )

    if not config.user_agent:
        if not config.wikimedia_contact:
            raise ValueError(
                "Wikimedia's User-Agent policy requires automated requests to self-identify "
                "with contact info, so the operator can be reached "
                "(https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy). Set "
                f"'wikimedia_contact' (an email or URL) in config.yaml, the {_ENV_WIKIMEDIA_CONTACT} "
                "environment variable, or 'user_agent' directly to override this check."
            )
        config.user_agent = (
            f"{config.wikimedia_tool_name}/{__version__} ({config.wikimedia_contact})"
        )

    return config


def resolve_llm_endpoint(config: Config) -> tuple[str, str, str]:
    """Returns (base_url, api_key, model) for whichever provider is configured."""
    if config.provider == "local":
        return (
            config.local_base_url,
            config.local_api_key or "local",
            config.local_model or config.model,
        )
    if config.provider == "experiential":
        return (
            config.experiential_base_url,
            config.experiential_api_key,
            config.experiential_model or config.model,
        )
    return config.openrouter_base_url, config.openrouter_api_key, config.model
