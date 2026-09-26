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
from wiki_translation_harness.models import Chunk, Config, RunStats
from wiki_translation_harness.pipeline import (
    run_assembly_repair,
    run_review_pass,
)
from wiki_translation_harness.skill_loader import SkillContent
from wiki_translation_harness.verification import VerifiedFacts


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
    wikitext = "English source text."


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
    assert stats.model_usage["test-model"].calls == 1  # config.model, per _config()'s default


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


# run_review_pass: the semantic-fidelity review pass (review.py), run once
# run_assembly_repair above has already passed clean. The same
# FakeOpenRouterClient drives both the review call (JSON findings) and any
# resulting repair_chunk() call, in call order, since review_client serves
# both roles here exactly as it does in the real pipeline (config.
# resolve_review_model/resolve_review_provider commonly resolve review to
# the same client as complex_model).


@pytest.mark.asyncio
async def test_review_pass_finds_and_fixes_issue():
    chunk = _chunk("Parisi eshte nje qytet i madh.")
    client = FakeOpenRouterClient(
        [
            '[{"kind": "grammar_case", "message": "Wrong case.", "snippet": "Parisi eshte"}]',
            "Parisi është një qytet i madh.",  # repair_chunk's fix
            "[]",  # re-check after repair: clean
        ]
    )
    stats = RunStats()

    assembled, issues, rounds = await run_review_pass(
        [chunk], _FakeSource(), _config(review_max_repair_attempts=2), "Parisi eshte nje qytet i madh.",
        client, "review-model", _skill(), None, FakeMediaWikiClient(raise_if_called=True), stats, VerifiedFacts(),
    )

    assert issues == []
    assert rounds == 1
    assert assembled == "Parisi është një qytet i madh."
    assert chunk.translated_text == "Parisi është një qytet i madh."
    assert stats.review_attempts == 1
    assert stats.review_corrections_applied == 1
    assert stats.review_findings_total == 1
    # 3 chat_completion calls total (2 review calls + 1 repair), all on
    # "review-model" -- confirms review.py's calls and the review-driven
    # repair both get attributed to the same per-model breakdown entry.
    assert stats.model_usage["review-model"].calls == 3
    assert stats.model_usage["review-model"].tokens_in == 300


@pytest.mark.asyncio
async def test_review_pass_localizes_finding_to_owning_chunk_only():
    a = _chunk("Chunk A has a problem here.", order=0)
    b = _chunk("Chunk B is perfectly fine.", order=1)
    client = FakeOpenRouterClient(
        [
            '[{"kind": "semantic_fidelity", "message": "Bad.", "snippet": "Chunk A has a problem here."}]',
            "Chunk A fixed.",
            "[]",
        ]
    )
    stats = RunStats()
    initial = a.translated_text + b.translated_text

    await run_review_pass(
        [a, b], _FakeSource(), _config(), initial, client, "review-model", _skill(), None,
        FakeMediaWikiClient(raise_if_called=True), stats, VerifiedFacts(),
    )

    assert a.translated_text == "Chunk A fixed."
    assert b.translated_text == "Chunk B is perfectly fine."  # untouched


@pytest.mark.asyncio
async def test_review_pass_structural_safety_net_catches_review_driven_regression():
    # The review call itself reports no findings, but the assembled text
    # already has a static defect ({{harvc}}) -- the structural safety-net
    # check (re-running _validate_assembled each round) must still catch
    # and repair it, independent of what the review model said.
    chunk = _chunk("{{harvc|last=Smith}}")
    client = FakeOpenRouterClient(["[]", "{{Cite book|last=Smith}}", "[]"])
    stats = RunStats()

    assembled, issues, rounds = await run_review_pass(
        [chunk], _FakeSource(), _config(review_max_repair_attempts=2), "{{harvc|last=Smith}}",
        client, "review-model", _skill(), None, FakeMediaWikiClient(raise_if_called=True), stats, VerifiedFacts(),
    )

    assert issues == []
    assert rounds == 1
    assert "harvc" not in assembled


@pytest.mark.asyncio
async def test_review_pass_uses_review_model_not_config_model():
    chunk = _chunk("Prozë.")
    client = FakeOpenRouterClient(["[]"])
    stats = RunStats()

    calls_with_model = []
    original_chat_completion = client.chat_completion

    async def _tracking_chat_completion(model, *args, **kwargs):
        calls_with_model.append(model)
        return await original_chat_completion(model, *args, **kwargs)

    client.chat_completion = _tracking_chat_completion

    await run_review_pass(
        [chunk], _FakeSource(), _config(model="draft-model"), "Prozë.", client, "review-model-xyz",
        _skill(), None, FakeMediaWikiClient(raise_if_called=True), stats, VerifiedFacts(),
    )

    assert calls_with_model == ["review-model-xyz"]
