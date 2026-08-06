from pathlib import Path

from wiki_translate_harness.models import Chunk
from wiki_translate_harness.output import (
    article_already_done,
    assemble_chunks,
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
