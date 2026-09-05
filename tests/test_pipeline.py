"""run_assembly_repair is pipeline.py's whole-article validate/repair loop
— the counterpart to translator.translate_chunk's per-chunk loop, but
operating on the assembled article and with its own round budget
(max_assembly_repair_rounds). Tested directly rather than through the full
run_pipeline(), which would require mocking fetch/plan/cache/OpenRouter
pricing lookups unrelated to this loop's own logic.

Static (validator.py) defects are used to drive most of these tests rather
than live-API ones, since the static checks need no network and are
already covered by test_validator.py — this file's job is proving the
orchestration (round counting, chunk-targeted repair, the cap, and
independence from the per-chunk loop), not re-testing either validator.
"""

from pathlib import Path

import pytest

from wiki_translate_harness.cache import TranslationCache, compute_key
from wiki_translate_harness.models import Chunk, Config, RunStats
from wiki_translate_harness.pipeline import run_assembly_repair
from wiki_translate_harness.skill_loader import SkillContent


class FakeOpenRouterClient:
    """Matches test_translator.py's fake — repair_chunk (via run_completion)
    calls client.chat_completion(model, messages, temperature, on_retry=...)."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat_completion(self, model, messages, temperature=0.0, on_retry=None):
        self.calls.append(messages)
        text = self.responses.pop(0)
        return text, 100, 50


class FakeMediaWikiClient:
    """Only parse_wikitext is exercised by run_assembly_repair (via
    live_validator.validate_wikitext_live)."""

    def __init__(self, responses: list[dict] | None = None, raise_if_called: bool = False):
        self.responses = list(responses or [])
        self.raise_if_called = raise_if_called
        self.calls = 0

    async def parse_wikitext(self, text: str, title: str = "API") -> dict:
        self.calls += 1
        if self.raise_if_called:
            raise AssertionError("parse_wikitext should not have been called")
        if self.responses:
            return self.responses.pop(0)
        return {"text": "<p>clean</p>", "templates": []}


def _skill() -> SkillContent:
    return SkillContent(skill_md="Translate faithfully.", reference_texts={})


def _config(**overrides) -> Config:
    base = dict(
        model="test-model",
        source_lang="en",
        target_lang="sq",
        live_validate=False,
        max_assembly_repair_rounds=3,
    )
    base.update(overrides)
    return Config.model_validate(base)


def _chunk(text: str, order: int = 0) -> Chunk:
    return Chunk(
        article_title="Test Article",
        section_titles=[f"S{order}"],
        order=order,
        text=text,
        token_estimate=10,
        translated_text=text,
    )


class _FakeSource:
    title = "Test Article"


@pytest.mark.asyncio
async def test_no_issues_needs_no_repair():
    chunk = _chunk("Prozë krejt e pastër shqipe.")
    client = FakeOpenRouterClient([])  # would raise IndexError if a repair call happened
    stats = RunStats()

    assembled, issues, rounds, _ = await run_assembly_repair(
        [chunk], _FakeSource(), _config(), client, _skill(), None,
        FakeMediaWikiClient(raise_if_called=True), None, stats,
    )

    assert issues == []
    assert rounds == 0
    assert client.calls == []


@pytest.mark.asyncio
async def test_resolves_within_cap():
    # {{harvc}} is a static defect (validator.py) — no network needed to detect it.
    chunk = _chunk("Bibliografia.\n{{harvc|last=Smith|c=Ch1}}\n")
    client = FakeOpenRouterClient(["Bibliografia.\n{{Cite book|last=Smith}}\n"])
    stats = RunStats()

    assembled, issues, rounds, _ = await run_assembly_repair(
        [chunk], _FakeSource(), _config(max_assembly_repair_rounds=3), client, _skill(), None,
        FakeMediaWikiClient(raise_if_called=True), None, stats,
    )

    assert issues == []
    assert rounds == 1
    assert "harvc" not in assembled
    assert chunk.translated_text.strip() == "Bibliografia.\n{{Cite book|last=Smith}}"
    assert stats.repair_attempts == 1


@pytest.mark.asyncio
async def test_exhausts_cap_returns_remaining_issues():
    chunk = _chunk("{{harvc|last=Smith}}")
    # Every repair attempt still contains {{harvc}} — never actually fixed.
    client = FakeOpenRouterClient(["{{harvc|last=Smith}} v2", "{{harvc|last=Smith}} v3"])
    stats = RunStats()

    assembled, issues, rounds, _ = await run_assembly_repair(
        [chunk], _FakeSource(), _config(max_assembly_repair_rounds=2), client, _skill(), None,
        FakeMediaWikiClient(raise_if_called=True), None, stats,
    )

    assert rounds == 2
    assert any(i.kind == "harvc_used" for i in issues)
    assert len(client.calls) == 2
    assert stats.repair_attempts == 2


@pytest.mark.asyncio
async def test_only_the_affected_chunk_gets_repaired():
    broken = _chunk("{{harvc|last=Smith}}", order=0)
    clean = _chunk("Prozë krejt e pastër.", order=1)
    client = FakeOpenRouterClient(["{{Cite book|last=Smith}}"])
    stats = RunStats()

    await run_assembly_repair(
        [broken, clean], _FakeSource(), _config(max_assembly_repair_rounds=3), client, _skill(), None,
        FakeMediaWikiClient(raise_if_called=True), None, stats,
    )

    assert len(client.calls) == 1  # only the broken chunk's repair call
    assert clean.translated_text == "Prozë krejt e pastër."  # untouched


@pytest.mark.asyncio
async def test_loop_cap_independent_of_per_chunk_max_repair_attempts():
    # run_assembly_repair never reads config.max_repair_attempts (that's
    # translate_chunk's own, separate loop) — max_assembly_repair_rounds
    # alone governs this loop, even set to 0 for the per-chunk knob.
    chunk = _chunk("{{harvc|last=Smith}}")
    client = FakeOpenRouterClient(["{{harvc|last=Smith}} still broken", "{{harvc|last=Smith}} still broken 2"])
    stats = RunStats()

    _, issues, rounds, _ = await run_assembly_repair(
        [chunk], _FakeSource(), _config(max_repair_attempts=0, max_assembly_repair_rounds=2),
        client, _skill(), None, FakeMediaWikiClient(raise_if_called=True), None, stats,
    )

    assert rounds == 2  # governed by max_assembly_repair_rounds, not the 0
    assert len(client.calls) == 2
    assert issues  # still broken, cap reached


@pytest.mark.asyncio
async def test_live_validate_disabled_never_calls_mediawiki_client():
    chunk = _chunk("Prozë krejt e pastër shqipe.")
    mw_client = FakeMediaWikiClient(raise_if_called=True)
    stats = RunStats()

    _, issues, _, _ = await run_assembly_repair(
        [chunk], _FakeSource(), _config(live_validate=False), FakeOpenRouterClient([]),
        _skill(), None, mw_client, None, stats,
    )
    assert issues == []
    assert mw_client.calls == 0


@pytest.mark.asyncio
async def test_live_validate_enabled_issue_repairs_via_live_check():
    chunk = _chunk("{{NonExistentTemplateXYZ}}")
    # First parse: reports the template as missing (drives one repair round).
    # Second parse (post-repair): clean.
    mw_client = FakeMediaWikiClient(
        responses=[
            {"text": "<p>irrelevant</p>", "templates": [{"ns": 10, "title": "Stampa:NonExistentTemplateXYZ", "exists": False}]},
            {"text": "<p>clean</p>", "templates": [{"ns": 10, "title": "Stampa:Sfn", "exists": True}]},
        ]
    )
    client = FakeOpenRouterClient(["Fixed: {{Sfn|Smith|2020}}"])
    stats = RunStats()

    assembled, issues, rounds, _ = await run_assembly_repair(
        [chunk], _FakeSource(), _config(live_validate=True, max_assembly_repair_rounds=2),
        client, _skill(), None, mw_client, None, stats,
    )

    assert issues == []
    assert rounds == 1
    assert mw_client.calls == 2


@pytest.mark.asyncio
async def test_unlocalized_issue_reaches_every_chunk():
    # An orphaned-named-ref finding's line_number/snippet come back None
    # (see test_live_validator.py) — it can't be pinned to one chunk, so
    # every chunk should see it in its repair error list rather than the
    # issue being silently dropped from every chunk's prompt.
    a = _chunk("Chunk A text.", order=0)
    b = _chunk("Chunk B text.", order=1)
    orphaned_ref_html = (
        '<span class="error mw-ext-cite-error">Gabim citimi: Etiketë ref e pavlefshme</span>'
    )
    mw_client = FakeMediaWikiClient(
        responses=[
            {"text": orphaned_ref_html, "templates": []},
            {"text": "<p>clean</p>", "templates": []},
        ]
    )
    client = FakeOpenRouterClient(["Chunk A fixed.", "Chunk B fixed."])
    stats = RunStats()

    await run_assembly_repair(
        [a, b], _FakeSource(), _config(live_validate=True, max_assembly_repair_rounds=2),
        client, _skill(), None, mw_client, None, stats,
    )

    assert len(client.calls) == 2  # both chunks got a repair call
    for call in client.calls:
        user_message = call[-1]["content"]
        assert "Gabim citimi" in user_message


@pytest.mark.asyncio
async def test_repaired_chunk_is_re_cached(tmp_path: Path):
    # translate_chunk's own cache.set (translator.py) runs pre-repair, when
    # a chunk is first translated. Without re-caching here, a fix made by
    # this assembly-level loop is invisible to future reruns' cache lookups
    # — this proves the fix actually reaches the cache under the same key
    # translate_chunk/translator.py would look it up with.
    chunk = _chunk("{{harvc|last=Smith}}")
    client = FakeOpenRouterClient(["{{Cite book|last=Smith}}"])
    stats = RunStats()
    cache = TranslationCache(tmp_path / "cache.sqlite3")
    config = _config(max_assembly_repair_rounds=3)
    skill = _skill()

    try:
        await run_assembly_repair(
            [chunk], _FakeSource(), config, client, skill, None,
            FakeMediaWikiClient(raise_if_called=True), None, stats,
            cache=cache, facts=None,
        )

        key = compute_key(config.model, chunk.source_lang, config.target_lang, chunk.text, skill.content_hash, "")
        assert cache.get(key) == chunk.translated_text
        assert "harvc" not in cache.get(key)
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_no_cache_arg_skips_re_caching_without_error():
    # cache defaults to None (matches every pre-existing call site above,
    # none of which pass it) -- must not raise just because a repair
    # happened with no cache configured.
    chunk = _chunk("{{harvc|last=Smith}}")
    client = FakeOpenRouterClient(["{{Cite book|last=Smith}}"])
    stats = RunStats()

    _, issues, rounds, _ = await run_assembly_repair(
        [chunk], _FakeSource(), _config(max_assembly_repair_rounds=3), client, _skill(), None,
        FakeMediaWikiClient(raise_if_called=True), None, stats,
    )

    assert issues == []
    assert rounds == 1
