from pathlib import Path

import pytest

from wiki_translation_harness.cache import TranslationCache
from wiki_translation_harness.models import Chunk, ChunkStatus, Config, RunStats
from wiki_translation_harness.skill_loader import SkillContent
from wiki_translation_harness.translator import translate_chunk
from wiki_translation_harness.verification import VerifiedFacts


class FakeOpenRouterClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat_completion(self, model, messages, temperature=0.0, on_retry=None, usage_out=None):
        self.calls.append(messages)
        text = self.responses.pop(0)
        return text, 100, 50


def _skill() -> SkillContent:
    return SkillContent(skill_md="Translate faithfully.", reference_texts={})


def _config(**overrides) -> Config:
    base = dict(
        model="test-model",
        source_lang="en",
        target_lang="sq",
        openrouter_api_key="sk-test",
        cache=True,
        validate=True,
        repair=True,
        max_repair_attempts=2,
    )
    base.update(overrides)
    return Config.model_validate(base)


def _chunk(text="'''Paris''' is a city.", source_lang="en") -> Chunk:
    return Chunk(
        article_title="Paris",
        section_titles=["Lead"],
        order=0,
        text=text,
        token_estimate=10,
        source_lang=source_lang,
    )


@pytest.mark.asyncio
async def test_translate_valid_output_no_repair(tmp_path: Path):
    client = FakeOpenRouterClient(["'''Parisi''' është qytet."])
    cache = TranslationCache(tmp_path / "c.sqlite3")
    stats = RunStats()
    outcome = await translate_chunk(_chunk(), _config(), client, _skill(), cache, None, stats)

    assert outcome.validation.valid
    assert outcome.repair_attempts == 0
    assert outcome.chunk.status == ChunkStatus.TRANSLATED
    assert stats.sections_translated == 1
    assert stats.cache_misses == 1
    assert stats.tokens_in == 100
    assert stats.tokens_out == 50
    assert stats.model_usage["test-model"].tokens_in == 100
    assert stats.model_usage["test-model"].tokens_out == 50
    assert stats.model_usage["test-model"].calls == 1
    cache.close()


@pytest.mark.asyncio
async def test_translate_cache_hit_skips_api_call(tmp_path: Path):
    cache = TranslationCache(tmp_path / "c.sqlite3")
    client = FakeOpenRouterClient(["'''Parisi''' është qytet."])
    stats = RunStats()
    chunk = _chunk()
    await translate_chunk(chunk, _config(), client, _skill(), cache, None, stats)

    # second call with identical input+model+langs must hit cache, not call the API again
    client2 = FakeOpenRouterClient([])  # would raise IndexError if called
    stats2 = RunStats()
    outcome2 = await translate_chunk(_chunk(), _config(), client2, _skill(), cache, None, stats2)

    assert outcome2.from_cache
    assert stats2.cache_hits == 1
    assert stats2.cache_misses == 0
    assert outcome2.chunk.translated_text == "'''Parisi''' është qytet."
    cache.close()


@pytest.mark.asyncio
async def test_invalid_output_triggers_repair_and_succeeds(tmp_path: Path):
    client = FakeOpenRouterClient(
        [
            "[[broken link translation",  # initial translation: invalid
            "'''Parisi''' është qytet me [[Lidhje|lidhje]].",  # repair: valid
        ]
    )
    cache = TranslationCache(tmp_path / "c.sqlite3")
    stats = RunStats()
    outcome = await translate_chunk(_chunk(), _config(), client, _skill(), cache, None, stats)

    assert outcome.validation.valid
    assert outcome.repair_attempts == 1
    assert outcome.chunk.status == ChunkStatus.REPAIRED
    assert stats.repair_attempts == 1
    assert stats.repairs_succeeded == 1
    # repair call's messages should carry the validation errors
    repair_call_content = client.calls[-1][-1]["content"]
    assert "link" in repair_call_content.lower()
    cache.close()


@pytest.mark.asyncio
async def test_repair_exhausted_marks_failed_and_does_not_cache(tmp_path: Path):
    client = FakeOpenRouterClient(
        [
            "[[broken 1",
            "[[broken 2",
            "[[broken 3",
        ]
    )
    cache = TranslationCache(tmp_path / "c.sqlite3")
    stats = RunStats()
    outcome = await translate_chunk(
        _chunk(), _config(max_repair_attempts=2), client, _skill(), cache, None, stats
    )

    assert not outcome.validation.valid
    assert outcome.chunk.status == ChunkStatus.FAILED
    assert stats.repairs_failed == 1
    assert stats.validation_failures == 1

    # a failed translation must not be cached, so a rerun retries it
    from wiki_translation_harness.cache import compute_key

    key = compute_key("test-model", "en", "sq", _chunk().text, _skill().content_hash)
    assert cache.get(key) is None
    cache.close()


@pytest.mark.asyncio
async def test_later_chunk_is_told_about_earlier_chunks_established_rendering(tmp_path: Path):
    cache = TranslationCache(tmp_path / "c.sqlite3")
    stats = RunStats()
    facts = VerifiedFacts(links={"Sustainable architecture": None})

    first_client = FakeOpenRouterClient(
        ["Shih {{ill|arkitektura e qëndrueshme|en|Sustainable architecture}}."]
    )
    await translate_chunk(
        _chunk(text="See [[Sustainable architecture]]."),
        _config(),
        first_client,
        _skill(),
        cache,
        None,
        stats,
        verified_facts=facts,
    )

    second_client = FakeOpenRouterClient(["ok"])
    await translate_chunk(
        _chunk(text="Also [[Sustainable architecture]] again."),
        _config(),
        second_client,
        _skill(),
        cache,
        None,
        stats,
        verified_facts=facts,
    )
    second_user_message = second_client.calls[-1][-1]["content"]
    assert "arkitektura e qëndrueshme" in second_user_message
    assert "reuse that exact rendering" in second_user_message
    cache.close()


@pytest.mark.asyncio
async def test_per_chunk_source_lang_isolates_cache(tmp_path: Path):
    cache = TranslationCache(tmp_path / "c.sqlite3")
    client_en = FakeOpenRouterClient(["English-sourced translation"])
    stats1 = RunStats()
    await translate_chunk(_chunk(source_lang="en"), _config(), client_en, _skill(), cache, None, stats1)

    # same text, same model, different source_lang -> must not hit the "en" cache entry
    client_sq = FakeOpenRouterClient(["Albanian-sourced translation"])
    stats2 = RunStats()
    outcome = await translate_chunk(
        _chunk(source_lang="sq"), _config(), client_sq, _skill(), cache, None, stats2
    )
    assert not outcome.from_cache
    assert outcome.chunk.translated_text == "Albanian-sourced translation"
    cache.close()


def _complex_chunk(text="{{Infobox settlement}}") -> Chunk:
    return Chunk(
        article_title="Paris",
        section_titles=["Infobox"],
        order=0,
        text=text,
        token_estimate=10,
        source_lang="en",
        is_complex=True,
    )


@pytest.mark.asyncio
async def test_complex_chunk_routes_to_distinct_complex_client_when_given():
    draft_client = FakeOpenRouterClient([])  # would raise IndexError if called
    complex_client = FakeOpenRouterClient(["'''Parisi''' është qytet."])
    stats = RunStats()
    outcome = await translate_chunk(
        _complex_chunk(),
        _config(complex_model="strong-model"),
        draft_client,
        _skill(),
        None,
        None,
        stats,
        complex_client=complex_client,
    )
    assert outcome.validation.valid
    assert draft_client.calls == []
    assert len(complex_client.calls) == 1
    # model_usage is keyed by model id, not client -- confirms spend is
    # attributed to "strong-model" (config.complex_model), not "test-model"
    # (config.model), even though both share this test's draft_client type.
    assert "strong-model" in stats.model_usage
    assert "test-model" not in stats.model_usage


@pytest.mark.asyncio
async def test_complex_chunk_repair_also_uses_complex_client():
    draft_client = FakeOpenRouterClient([])  # would raise IndexError if called
    complex_client = FakeOpenRouterClient(
        # First response uses {{harvc}}, which validator.py flags unconditionally
        # (broken on sqwiki) -> triggers a repair round; second is clean.
        ["Bibliografia.\n{{harvc|last=Smith|c=Ch1}}\n", "Bibliografia.\n{{Cite book|last=Smith}}\n"]
    )
    stats = RunStats()
    outcome = await translate_chunk(
        _complex_chunk(),
        _config(complex_model="strong-model", max_repair_attempts=1),
        draft_client,
        _skill(),
        None,
        None,
        stats,
        complex_client=complex_client,
    )
    assert draft_client.calls == []
    assert len(complex_client.calls) == 2
    assert outcome.repair_attempts == 1
    assert outcome.validation.valid
