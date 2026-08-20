"""Saves final MediaWiki source files. No publishing — local .wiki files only."""

from __future__ import annotations

import re
from pathlib import Path

from wiki_translate_harness.models import Chunk

_UNSAFE_RE = re.compile(r'[\\/:*?"<>|]')


def sanitize_filename(title: str) -> str:
    """Article title -> filesystem-safe base name, MediaWiki-style (spaces -> underscores)."""
    name = title.strip().replace(" ", "_")
    name = _UNSAFE_RE.sub("_", name)
    return name


def _assemble_with_spans(chunks: list[Chunk]) -> tuple[str, list[tuple[int, int]]]:
    """Shared core for assemble_chunks/assemble_chunks_with_spans. Returns
    the joined text plus each chunk's 1-indexed (start_line, end_line)
    span in that text — end_line < start_line for a chunk that contributed
    no lines (empty translated_text)."""
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    current_line = 1
    for chunk in chunks:
        text = chunk.translated_text or ""
        if pieces and pieces[-1] and not pieces[-1].endswith("\n") and text and not text.startswith("\n"):
            pieces.append("\n")
            current_line += 1
        start_line = current_line
        pieces.append(text)
        if text:
            current_line += text.count("\n")
            end_line = current_line - 1 if text.endswith("\n") else current_line
        else:
            end_line = start_line - 1
        spans.append((start_line, end_line))
    return "".join(pieces), spans


def assemble_chunks(chunks: list[Chunk]) -> str:
    """Concatenate translated chunks in order, guaranteeing a newline at every
    chunk boundary. Models don't always faithfully preserve trailing
    whitespace, and a missing newline right before a `==` heading silently
    breaks it (MediaWiki only recognizes a heading at the start of a line) —
    this is a purely mechanical safety net, independent of prompting."""
    return _assemble_with_spans(chunks)[0]


def assemble_chunks_with_spans(chunks: list[Chunk]) -> tuple[str, list[tuple[Chunk, int, int]]]:
    """Like assemble_chunks, but also returns each chunk's 1-indexed
    (start_line, end_line) span in the assembled text — used by the
    assembly-level repair loop (pipeline.py) to map a validation finding's
    line_number back to the chunk that produced it."""
    text, spans = _assemble_with_spans(chunks)
    return text, [(chunk, start, end) for chunk, (start, end) in zip(chunks, spans)]


def output_path_for(output_dir: Path, title: str) -> Path:
    return output_dir / f"{sanitize_filename(title)}.wiki"


def article_already_done(output_dir: Path, title: str) -> bool:
    return output_path_for(output_dir, title).exists()


def save_article(output_dir: Path, title: str, wikitext: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_path_for(output_dir, title)
    path.write_text(wikitext, encoding="utf-8")
    return path
