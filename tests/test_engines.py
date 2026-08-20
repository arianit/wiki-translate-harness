from wiki_translate_harness.claude_code_client import ClaudeCodeClient
from wiki_translate_harness.engines import build_llm_client
from wiki_translate_harness.models import Config
from wiki_translate_harness.openrouter import OpenRouterClient


def _config(**overrides) -> Config:
    base = dict(model="test-model", openrouter_api_key="sk-test", user_agent="test-agent/1.0")
    base.update(overrides)
    return Config.model_validate(base)


def test_claude_code_provider_dispatches_to_claude_code_client():
    client, effective_model = build_llm_client(_config(provider="claude_code", model="claude-sonnet-5"))
    assert isinstance(client, ClaudeCodeClient)
    assert effective_model == "claude-sonnet-5"


def test_openrouter_provider_dispatches_to_openrouter_client():
    client, effective_model = build_llm_client(_config(provider="openrouter", model="deepseek/deepseek-v3"))
    assert isinstance(client, OpenRouterClient)
    assert effective_model == "deepseek/deepseek-v3"


def test_local_provider_falls_back_to_local_model():
    client, effective_model = build_llm_client(
        _config(provider="local", model="deepseek/deepseek-v3", local_model="llama-3.1-8b-instruct")
    )
    assert isinstance(client, OpenRouterClient)
    assert effective_model == "llama-3.1-8b-instruct"
