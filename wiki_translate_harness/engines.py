"""The single place a new translation engine gets registered.

Every engine just needs to satisfy LLMEngineClient's contract — proven
structural/duck-typed, not nominal, by tests/test_translator.py's
FakeOpenRouterClient, which implements chat_completion() alone with no
inheritance from OpenRouterClient — and be wired into build_llm_client()
below. translator.py's translate_chunk() and repair.py's repair_chunk()
never need to change when a new engine is added; only this file and a new
client module do.
"""
from __future__ import annotations

from typing import Protocol

from wiki_translate_harness.config import resolve_llm_endpoint
from wiki_translate_harness.models import Config, ModelPricing
from wiki_translate_harness.openrouter import OpenRouterClient, RetryCallback


class LLMEngineClient(Protocol):
    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        on_retry: RetryCallback | None = None,
    ) -> tuple[str, int, int]: ...

    async def get_pricing_for(self, model: str) -> ModelPricing | None: ...

    async def fetch_pricing(self) -> dict[str, ModelPricing]: ...

    async def aclose(self) -> None: ...


def build_llm_client(config: Config) -> tuple[LLMEngineClient, str]:
    """Returns (client, effective_model). effective_model may differ from
    config.model (e.g. provider: local's local_model fallback, resolved by
    resolve_llm_endpoint) — callers should
    config.model_copy(update={"model": effective_model}) if it differs, the
    same way both call sites did before this function replaced their inline
    resolve_llm_endpoint(...) -> OpenRouterClient(...) construction."""
    if config.provider == "claude_code":
        # Local import: keeps claude_code_client.py's subprocess/tempfile
        # imports out of the path for anyone only ever using openrouter/local.
        from wiki_translate_harness.claude_code_client import ClaudeCodeClient

        client = ClaudeCodeClient(
            model=config.model,
            cli_path=config.claude_code_cli_path,
            permission_mode=config.claude_code_permission_mode,
            timeout_s=config.request_timeout_s,
            max_retries=config.max_retries,
        )
        return client, config.model

    base_url, api_key, model = resolve_llm_endpoint(config)
    client = OpenRouterClient(
        api_key=api_key,
        base_url=base_url,
        user_agent=config.user_agent,
        timeout=config.request_timeout_s,
        max_retries=config.max_retries,
        provider=config.provider,
    )
    return client, model
