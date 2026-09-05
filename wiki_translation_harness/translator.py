"""Per-chunk orchestration: cache check -> invoke skill -> validate -> repair -> cache store.

This module contains no translation judgment — it only sequences calls into
cache.py, skill_loader.py, openrouter.py, validator.py, and repair.py.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from wiki_translation_harness.cache import TranslationCache, compute_key
from wiki_translation_harness.engines import LLMEngineClient
from wiki_translation_harness.models import Chunk, ChunkStatus, Config, ModelPricing, RunStats, TranslationResult, ValidationResult
from wiki_translation_harness.openrouter import RetryCallback, run_completion
from wiki_translation_harness.repair import repair_chunk
from wiki_translation_harness.skill_loader import SkillContent, build_translation_messages
from wiki_translation_harness.validator import format_errors, validate_wikitext
from wiki_translation_harness.verification import VerifiedFacts, build_verified_facts_block


@dataclass
class ChunkOutcome:
    chunk: Chunk
    validation: ValidationResult
    from_cache: bool
    repair_attempts: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


def _accumulate(stats: RunStats, result: TranslationResult) -> None:
    stats.tokens_in += result.prompt_tokens
    stats.tokens_out += result.completion_tokens
    stats.estimated_cost_usd += result.cost_usd
    stats.translation_time_s += result.latency_s


async def translate_chunk(
    chunk: Chunk,
    config: Config,
    client: LLMEngineClient,
    skill: SkillContent,
    cache: TranslationCache | None,
    pricing: ModelPricing | None,
    stats: RunStats,
    on_retry: RetryCallback | None = None,
    verified_facts: VerifiedFacts | None = None,
) -> ChunkOutcome:
    facts_block = build_verified_facts_block(chunk.text, verified_facts) if verified_facts else ""
    facts_hash = hashlib.sha256(facts_block.encode("utf-8")).hexdigest()[:16] if facts_block else ""

    key = compute_key(
        config.model, chunk.source_lang, config.target_lang, chunk.text, skill.content_hash, facts_hash
    )

    cached_text = cache.get(key) if cache is not None else None
    if cached_text is not None:
        stats.cache_hits += 1
        chunk.translated_text = cached_text
        chunk.status = ChunkStatus.CACHED
        validation = validate_wikitext(cached_text) if config.validate_output else ValidationResult(valid=True)
        return ChunkOutcome(chunk=chunk, validation=validation, from_cache=True, repair_attempts=0, prompt_tokens=0, completion_tokens=0, latency_s=0.0)

    stats.cache_misses += 1

    messages = build_translation_messages(
        skill,
        chunk.source_lang,
        config.target_lang,
        chunk.article_title,
        chunk.section_title,
        chunk.text,
        verified_facts_block=facts_block,
    )
    result = await run_completion(client, config.model, messages, config.temperature, pricing, on_retry=on_retry)
    _accumulate(stats, result)
    total_prompt_tokens = result.prompt_tokens
    total_completion_tokens = result.completion_tokens
    total_latency = result.latency_s

    translated_text = result.text
    validation = ValidationResult(valid=True)
    repair_attempts = 0

    if config.validate_output:
        validation = validate_wikitext(translated_text)
        while not validation.valid and config.repair and repair_attempts < config.max_repair_attempts:
            repair_attempts += 1
            stats.repair_attempts += 1
            errors = format_errors(validation)
            repair_result = await repair_chunk(
                client,
                skill,
                config.model,
                config.temperature,
                chunk.source_lang,
                config.target_lang,
                chunk.article_title,
                chunk.section_title,
                translated_text,
                errors,
                pricing,
                on_retry=on_retry,
            )
            _accumulate(stats, repair_result)
            total_prompt_tokens += repair_result.prompt_tokens
            total_completion_tokens += repair_result.completion_tokens
            total_latency += repair_result.latency_s
            translated_text = repair_result.text
            validation = validate_wikitext(translated_text)

        if repair_attempts > 0:
            if validation.valid:
                stats.repairs_succeeded += 1
            else:
                stats.repairs_failed += 1
        if not validation.valid:
            stats.validation_failures += 1

    chunk.translated_text = translated_text
    if not config.validate_output:
        chunk.status = ChunkStatus.TRANSLATED
    elif validation.valid:
        chunk.status = ChunkStatus.REPAIRED if repair_attempts > 0 else ChunkStatus.TRANSLATED
    else:
        chunk.status = ChunkStatus.FAILED

    if cache is not None and chunk.status != ChunkStatus.FAILED:
        cache.set(key, config.model, chunk.source_lang, config.target_lang, chunk.text, translated_text)

    stats.sections_translated += 1

    return ChunkOutcome(chunk=chunk, validation=validation, from_cache=False, repair_attempts=repair_attempts, prompt_tokens=total_prompt_tokens, completion_tokens=total_completion_tokens, latency_s=total_latency)
