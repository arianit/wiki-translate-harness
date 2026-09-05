"""Syntax repair: invokes the same Pi skill, asked to fix structure only.

Never re-translates or embeds translation guidance of its own — the skill
text is still the sole source of style/convention context, and the fixed
repair framing (skill_loader._REPAIR_FRAME) restricts the model to
correcting the specific validation errors passed in.
"""

from __future__ import annotations

from wiki_translation_harness.engines import LLMEngineClient
from wiki_translation_harness.models import ModelPricing, TranslationResult
from wiki_translation_harness.openrouter import RetryCallback, run_completion
from wiki_translation_harness.skill_loader import SkillContent, build_repair_messages


async def repair_chunk(
    client: LLMEngineClient,
    skill: SkillContent,
    model: str,
    temperature: float,
    source_lang: str,
    target_lang: str,
    article_title: str,
    section_title: str,
    invalid_text: str,
    errors: list[str],
    pricing: ModelPricing | None,
    on_retry: RetryCallback | None = None,
) -> TranslationResult:
    messages = build_repair_messages(
        skill, source_lang, target_lang, article_title, section_title, invalid_text, errors
    )
    return await run_completion(client, model, messages, temperature, pricing, on_retry=on_retry)
