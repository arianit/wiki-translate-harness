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

from wiki_translation_harness.cache import TranslationCache, compute_key
from wiki_translation_harness.models import Chunk, Config, RunStats, ValidationIssue
from wiki_translation_harness.pipeline import (
    _chunk_mentions_ref,
    _ref_names_in_message,
    run_assembly_repair,
)
from wiki_translation_harness.skill_loader import SkillContent


class FakeOpenRouterClient:
    """Matches test_translator.py's fake — repair_chunk (via run_completion)
    calls client.chat_completion(model, messages, temperature, on_retry=...)."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat_completion(self, model, messages, temperature=0.0, on_retry=None, usage_out=None):
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
async def test_qa_skill_forwarded_to_assembly_repair_call():
    chunk = _chunk("Bibliografia.\n{{harvc|last=Smith|c=Ch1}}\n")
    client = FakeOpenRouterClient(["Bibliografia.\n{{Cite book|last=Smith}}\n"])
    stats = RunStats()
    qa_skill = SkillContent(skill_md="Check ref names before delivery.", reference_texts={})

    await run_assembly_repair(
        [chunk], _FakeSource(), _config(max_assembly_repair_rounds=3), client, _skill(), None,
        FakeMediaWikiClient(raise_if_called=True), None, stats, qa_skill=qa_skill,
    )

    repair_system_prompt = client.calls[-1][0]["content"]
    assert "Check ref names before delivery." in repair_system_prompt


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


def test_ref_names_in_message_extracts_quoted_ref_name():
    # Real Cite error text never echoes `name="X"` tag syntax back — confirmed
    # against a live sq.wikipedia.org parse (see test_live_validator.py): it
    # just quotes the bare name ("...refs e quajtura "RefA""). Only
    # orphaned_named_ref/cite_error kinds get read this way — other kinds
    # that happen to quote text too (e.g. unexpanded_template's repr'd
    # title) must not be misread as naming a ref.
    orphaned = ValidationIssue(
        kind="orphaned_named_ref",
        message=(
            "Cite error rendered on the page: Gabim citimi: Etiketë <ref> e "
            'pavlefshme;\nasnjë tekst nuk u dha për refs e quajtura "RefA"'
        ),
    )
    assert _ref_names_in_message(orphaned) == ["RefA"]

    # A finding that doesn't name a ref (e.g. Scribunto error, a leak) gives no names.
    lua_error = ValidationIssue(
        kind="lua_script_error", message="Lua/Scribunto error rendered on the page: Script error"
    )
    assert _ref_names_in_message(lua_error) == []

    # Quoted text, but not a ref-naming kind — must not be read as a ref name.
    missing_template = ValidationIssue(
        kind="unexpanded_template", message="Template 'RefA' does not exist on the target wiki"
    )
    assert _ref_names_in_message(missing_template) == []


def test_chunk_mentions_ref_matches_translated_then_source():
    ref_chunk = _chunk("Vijazimi.\n<ref name=\"RefA\"/>\n", order=0)
    assert _chunk_mentions_ref(ref_chunk, "RefA")
    assert not _chunk_mentions_ref(_chunk("Prozë krejt e pastër.", order=1), "RefA")


@pytest.mark.asyncio
async def test_unlocalized_ref_issue_only_reaches_chunk_mentioning_that_ref():
    # An orphaned named ref can't be pinned to a chunk by line number, but its
    # message names `<ref name="RefA"/>` — the finding should reach ONLY the
    # chunk whose text mentions that ref, not every chunk.
    clean = _chunk("Prozë krejt e pastër.", order=0)
    with_ref = _chunk("Vijazimi.\nKjo qe e dhëna.<ref name=\"RefA\"/>\n", order=1)
    orphaned_ref_html = (
        '<span class="error mw-ext-cite-error">Gabim citimi: Etiketë &lt;ref&gt; e '
        'pavlefshme;\nasnjë tekst nuk u dha për refs e quajtura "RefA"</span>'
    )
    mw_client = FakeMediaWikiClient(
        responses=[
            {"text": orphaned_ref_html, "templates": []},
            {"text": "<p>clean</p>", "templates": []},
        ]
    )
    client = FakeOpenRouterClient(["Chunk with ref fixed."])
    stats = RunStats()

    assembled, issues, rounds, _ = await run_assembly_repair(
        [clean, with_ref], _FakeSource(), _config(live_validate=True, max_assembly_repair_rounds=2),
        client, _skill(), None, mw_client, None, stats,
    )

    assert issues == []
    assert rounds == 1
    assert len(client.calls) == 1  # only the ref-holding chunk got a repair call
    assert with_ref.translated_text == "Chunk with ref fixed."
    assert clean.translated_text == "Prozë krejt e pastër."  # untouched


@pytest.mark.asyncio
async def test_unlocalized_ref_issue_targets_only_first_matching_chunk():
    # Two chunks both mention the ref (e.g. usage in two sections). Broadcasting
    # to both would let each "fix" the orphaned ref by inserting its own
    # definition — producing a define-twice error next round and an
    # oscillating repair loop. The finding must go to the FIRST matching chunk
    # only.
    clean = _chunk("Prozë krejt e pastër.", order=0)
    first = _chunk("Seksioni A.<ref name=\"RefA\"/>\n", order=1)
    second = _chunk("Seksioni B.<ref name=\"RefA\"/>\n", order=2)
    orphaned_ref_html = (
        '<span class="error mw-ext-cite-error">Gabim citimi: Etiketë &lt;ref&gt; e '
        'pavlefshme;\nasnjë tekst nuk u dha për refs e quajtura "RefA"</span>'
    )
    mw_client = FakeMediaWikiClient(
        responses=[
            {"text": orphaned_ref_html, "templates": []},
            {"text": "<p>clean</p>", "templates": []},
        ]
    )
    client = FakeOpenRouterClient(["Seksioni A fixed."])
    stats = RunStats()

    await run_assembly_repair(
        [clean, first, second], _FakeSource(), _config(live_validate=True, max_assembly_repair_rounds=2),
        client, _skill(), None, mw_client, None, stats,
    )

    assert len(client.calls) == 1
    assert first.translated_text == "Seksioni A fixed."
    assert second.translated_text == "Seksioni B.<ref name=\"RefA\"/>\n"  # untouched
    assert clean.translated_text == "Prozë krejt e pastër."  # untouched


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
