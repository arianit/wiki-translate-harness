from pathlib import Path

from wiki_translate_harness.models import Chunk
from wiki_translate_harness.output import (
    article_already_done,
    assemble_chunks,
    assemble_chunks_with_spans,
    output_path_for,
    sanitize_filename,
    save_article,
)


def test_sanitize_filename_spaces_to_underscores():
    assert sanitize_filename("Paris") == "Paris"
    assert sanitize_filename("Battle of Kosovo") == "Battle_of_Kosovo"


def test_sanitize_filename_strips_unsafe_chars():
    assert "/" not in sanitize_filename("Foo/Bar")
    assert ":" not in sanitize_filename("Foo: Bar")


def test_save_and_detect_done(tmp_path: Path):
    assert not article_already_done(tmp_path, "Paris")
    save_article(tmp_path, "Paris", "'''Paris''' is a city.")
    assert article_already_done(tmp_path, "Paris")
    path = output_path_for(tmp_path, "Paris")
    assert path.read_text(encoding="utf-8") == "'''Paris''' is a city."


def test_save_utf8_preserved(tmp_path: Path):
    text = "Përshëndetje, kjo është shqip: ë, ç, Shqipëria."
    save_article(tmp_path, "Shqipëria", text)
    path = output_path_for(tmp_path, "Shqipëria")
    assert path.read_text(encoding="utf-8") == text


def _chunk(text: str, order: int) -> Chunk:
    return Chunk(article_title="T", section_titles=["S"], order=order, text=text, token_estimate=1, translated_text=text)


def test_assemble_inserts_missing_newline_at_boundary():
    # regression: a model response that doesn't end with a newline must not
    # merge directly into the next chunk's heading
    chunks = [_chunk("...end of section.", 0), _chunk("== Referime ==\n{{reflist}}\n", 1)]
    assembled = assemble_chunks(chunks)
    assert "\n== Referime ==" in assembled
    assert "section.==" not in assembled


def test_assemble_does_not_double_newline_when_already_present():
    chunks = [_chunk("First.\n\n", 0), _chunk("== Second ==\nBody.\n", 1)]
    assembled = assemble_chunks(chunks)
    assert assembled == "First.\n\n== Second ==\nBody.\n"


def test_assemble_with_spans_matches_assemble_chunks_text():
    chunks = [_chunk("...end of section.", 0), _chunk("== Referime ==\n{{reflist}}\n", 1)]
    text, _ = assemble_chunks_with_spans(chunks)
    assert text == assemble_chunks(chunks)


def test_assemble_with_spans_covers_each_chunk_correctly():
    chunks = [
        _chunk("Lead line 1.\nLead line 2.\n", 0),
        _chunk("== History ==\nBody text.\n", 1),
        _chunk("No trailing newline here", 2),
    ]
    text, spans = assemble_chunks_with_spans(chunks)
    lines = text.split("\n")
    assert [c.order for c, _, _ in spans] == [0, 1, 2]

    (_, s0, e0), (_, s1, e1), (_, s2, e2) = spans
    assert "\n".join(lines[s0 - 1 : e0]) == "Lead line 1.\nLead line 2."
    assert "\n".join(lines[s1 - 1 : e1]) == "== History ==\nBody text."
    assert "\n".join(lines[s2 - 1 : e2]) == "No trailing newline here"
    # spans don't overlap and advance strictly forward
    assert e0 < s1 <= e1 < s2 <= e2


def test_assemble_with_spans_empty_chunk_gets_empty_span():
    # An empty chunk contributes no lines: end_line < start_line signals
    # "nothing here" to a caller mapping validation findings back to chunks.
    chunks = [_chunk("A\n", 0), _chunk("", 1), _chunk("B\n", 2)]
    text, spans = assemble_chunks_with_spans(chunks)
    assert text == "A\nB\n"
    (_, s0, e0), (_, s1, e1), (_, s2, e2) = spans
    assert e1 < s1  # empty span for the empty chunk
    assert (s0, e0) == (1, 1)
    assert (s2, e2) == (2, 2)


def test_assemble_with_spans_line_count_changes_between_calls():
    # A chunk's repaired text can have a different line count than its
    # original — spans must reflect the CURRENT text, not go stale, since
    # the assembly-level repair loop recomputes them fresh every round.
    chunk = _chunk("One line.\n", 0)
    other = _chunk("Tail.\n", 1)
    _, spans_before = assemble_chunks_with_spans([chunk, other])
    assert spans_before[1][1] == 2  # other starts at line 2

    chunk.translated_text = "Line one.\nLine two.\nLine three.\n"
    _, spans_after = assemble_chunks_with_spans([chunk, other])
    assert spans_after[1][1] == 4  # other now starts at line 4, not 2
