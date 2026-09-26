import pytest

from wiki_translation_harness.claude_code_client import ClaudeCodeClient
from wiki_translation_harness.engines import build_client_pool, build_llm_client
from wiki_translation_harness.models import Config
from wiki_translation_harness.opencode_go_client import OpenCodeGoClient
from wiki_translation_harness.openrouter import OpenRouterClient


def _config(**overrides) -> Config:
    base = dict(model="test-model", openrouter_api_key="sk-test", user_agent="test-agent/1.0")
    base.update(overrides)
    return Config.model_validate(base)


def test_claude_code_provider_dispatches_to_claude_code_client():
    client, effective_model = build_llm_client(_config(provider="claude_code", model="claude-sonnet-5"))
    assert isinstance(client, ClaudeCodeClient)
    assert effective_model == "claude-sonnet-5"


def test_claude_code_provider_defaults_effort_to_medium():
    client, _ = build_llm_client(_config(provider="claude_code", model="claude-sonnet-5"))
    assert client.effort == "medium"


def test_local_provider_falls_back_to_local_model():
    client, effective_model = build_llm_client(
        _config(provider="local", model="deepseek/deepseek-v3", local_model="llama-3.1-8b-instruct")
    )
    assert isinstance(client, OpenRouterClient)
    assert effective_model == "llama-3.1-8b-instruct"


def test_opencode_go_agent_config_threaded_through():
    client, _ = build_llm_client(
        _config(provider="opencode_go", model="auto", opencode_go_agent="wiki-translation-harness")
    )
    assert isinstance(client, OpenCodeGoClient)
    assert client.agent == "wiki-translation-harness"


def test_client_pool_builds_second_client_for_different_complex_provider():
    pool, cfg = build_client_pool(
        _config(
            provider="opencode_go",
            model="auto",
            complex_model="claude-sonnet-5",
            complex_provider="claude_code",
        )
    )
    assert set(pool.clients.keys()) == {"opencode_go", "claude_code"}
    assert isinstance(pool.get("opencode_go"), OpenCodeGoClient)
    assert isinstance(pool.get("claude_code"), ClaudeCodeClient)


def test_client_pool_review_provider_reuses_complex_provider_client():
    # review_model/review_provider unset -> inherit complex_model/complex_provider
    # (config.resolve_review_model/resolve_review_provider) -- must not build
    # a third client for what resolves to the same provider as complex.
    pool, cfg = build_client_pool(
        _config(
            provider="opencode_go",
            model="auto",
            complex_model="claude-sonnet-5",
            complex_provider="claude_code",
        )
    )
    assert set(pool.clients.keys()) == {"opencode_go", "claude_code"}


def test_client_pool_updates_config_model_for_local_provider_substitution():
    pool, cfg = build_client_pool(
        _config(provider="local", model="deepseek/deepseek-v3", local_model="llama-3.1-8b-instruct")
    )
    assert cfg.model == "llama-3.1-8b-instruct"


@pytest.mark.asyncio
async def test_client_pool_aclose_closes_every_distinct_client():
    pool, _ = build_client_pool(
        _config(
            provider="opencode_go",
            model="auto",
            complex_model="claude-sonnet-5",
            complex_provider="claude_code",
        )
    )
    await pool.aclose()  # both fake-free real clients: no-op aclose(), just confirms no crash
