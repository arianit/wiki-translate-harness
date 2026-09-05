from pathlib import Path

from wiki_translate_harness.models import Chunk, ChunkStatus
from wiki_translate_harness.output import (
    article_already_done,
    assemble_chunks,
    assemble_chunks_partial,
    assemble_chunks_with_spans,
    discard_partial_article,
    output_path_for,
    sanitize_filename,
    save_article,
    save_partial_article,
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


def _chunk_with_status(text: str, order: int, status: ChunkStatus, section: str = "S") -> Chunk:
    return Chunk(
        article_title="T", section_titles=[section], order=order, text=text,
        token_estimate=1, translated_text=text, status=status,
    )


def test_assemble_chunks_partial_keeps_done_chunks_verbatim():
    done = _chunk_with_status("Translated body.\n", 0, ChunkStatus.TRANSLATED)
    cached = _chunk_with_status("Cached body.\n", 1, ChunkStatus.CACHED, section="Cached")
    repaired = _chunk_with_status("Repaired body.\n", 2, ChunkStatus.REPAIRED, section="Repaired")
    assembled = assemble_chunks_partial([done, cached, repaired])
    assert assembled == assemble_chunks([done, cached, repaired])


def test_assemble_chunks_partial_placeholders_pending_and_failed():
    done = _chunk_with_status("Done body.\n", 0, ChunkStatus.TRANSLATED, section="History")
    pending = _chunk_with_status("", 1, ChunkStatus.PENDING, section="Geography")
    failed = _chunk_with_status("garbled", 2, ChunkStatus.FAILED, section="Climate")
    assembled = assemble_chunks_partial([done, pending, failed])
    assert "Done body." in assembled
    assert "garbled" not in assembled  # failed chunk's bad text must not leak through
    assert "<!-- pending: Geography (pending) -->" in assembled
    assert "<!-- pending: Climate (failed) -->" in assembled


def test_assemble_chunks_partial_never_drops_content_for_missing_chunks():
    # Every chunk contributes *something* (real text or a named placeholder)
    # so a partial snapshot never silently reads as shorter than it is.
    chunks = [
        _chunk_with_status("A\n", 0, ChunkStatus.TRANSLATED, section="Alpha"),
        _chunk_with_status("", 1, ChunkStatus.PENDING, section="Beta"),
        _chunk_with_status("C\n", 2, ChunkStatus.CACHED, section="Gamma"),
    ]
    assembled = assemble_chunks_partial(chunks)
    assert "Alpha" not in assembled  # done chunks contribute only their body, not the section title
    assert "Beta" in assembled
    assert "Gamma" not in assembled


def test_save_partial_article_writes_readable_file_no_tmp_left_behind(tmp_path: Path):
    chunks = [_chunk_with_status("Body.\n", 0, ChunkStatus.TRANSLATED)]
    path = save_partial_article(tmp_path, "Mars", chunks)
    assert path == output_path_for(tmp_path, "Mars")
    assert path.read_text(encoding="utf-8") == "Body.\n"
    assert not path.with_name(path.name + ".tmp").exists()


def test_save_partial_article_overwrites_on_rerun(tmp_path: Path):
    save_partial_article(tmp_path, "Mars", [_chunk_with_status("First.\n", 0, ChunkStatus.TRANSLATED)])
    path = save_partial_article(tmp_path, "Mars", [_chunk_with_status("Second.\n", 0, ChunkStatus.TRANSLATED)])
    assert path.read_text(encoding="utf-8") == "Second.\n"


def test_discard_partial_article_removes_file(tmp_path: Path):
    save_partial_article(tmp_path, "Mars", [_chunk_with_status("Body.\n", 0, ChunkStatus.TRANSLATED)])
    assert output_path_for(tmp_path, "Mars").exists()
    discard_partial_article(tmp_path, "Mars")
    assert not output_path_for(tmp_path, "Mars").exists()


def test_discard_partial_article_is_noop_when_missing(tmp_path: Path):
    discard_partial_article(tmp_path, "NeverStarted")  # must not raise
