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


def assemble_chunks(chunks: list[Chunk]) -> str:
    """Concatenate translated chunks in order, guaranteeing a newline at every
    chunk boundary. Models don't always faithfully preserve trailing
    whitespace, and a missing newline right before a `==` heading silently
    breaks it (MediaWiki only recognizes a heading at the start of a line) —
    this is a purely mechanical safety net, independent of prompting."""
    pieces: list[str] = []
    for chunk in chunks:
        text = chunk.translated_text or ""
        if pieces and pieces[-1] and not pieces[-1].endswith("\n") and text and not text.startswith("\n"):
            pieces.append("\n")
        pieces.append(text)
    return "".join(pieces)


def output_path_for(output_dir: Path, title: str) -> Path:
    return output_dir / f"{sanitize_filename(title)}.wiki"


def article_already_done(output_dir: Path, title: str) -> bool:
    return output_path_for(output_dir, title).exists()


def save_article(output_dir: Path, title: str, wikitext: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_path_for(output_dir, title)
    path.write_text(wikitext, encoding="utf-8")
    return path
