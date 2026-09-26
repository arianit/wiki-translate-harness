"""Records articles that exhausted the assembly-level repair loop
(pipeline.py) still holding unresolved validation issues.

Two artifacts, mirroring report.py's {title}.report.md convention:
- `{title}.review.md` — human-readable findings for that one article.
- `needs_human_review.json` — an index of every such article in this
  output_dir, atomic-written the same way statistics.py writes stats.json,
  so a batch runner (or a human) can query "what needs attention" without
  scanning individual .review.md files.

Also, `record_review_flags`/`{title}.review-flags.md` — the non-blocking
counterpart used by pipeline.run_review_pass (the semantic-fidelity review
pass, see review.py) when it still has unresolved findings after its own
repair-attempt cap. Unlike record_needs_human_review above, this does NOT
withhold the `.wiki` file: a fidelity/grammar concern is a human spot-check
prompt, not a publish blocker the way an unresolved structural defect is
(this harness's output is always paste-ready wikitext for a human to review
before it ever reaches the live wiki, never auto-published).
"""

from __future__ import annotations

import time
from pathlib import Path

import orjson

from wiki_translation_harness.models import ValidationIssue
from wiki_translation_harness.output import sanitize_filename

_INDEX_FILENAME = "needs_human_review.json"


def review_path_for(output_dir: Path, title: str) -> Path:
    return output_dir / f"{sanitize_filename(title)}.review.md"


def review_flags_path_for(output_dir: Path, title: str) -> Path:
    return output_dir / f"{sanitize_filename(title)}.review-flags.md"


def _index_path(output_dir: Path) -> Path:
    return output_dir / _INDEX_FILENAME


def _load_index(output_dir: Path) -> list[dict]:
    path = _index_path(output_dir)
    if not path.exists():
        return []
    try:
        data = orjson.loads(path.read_bytes())
    except orjson.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _write_index(output_dir: Path, entries: list[dict]) -> None:
    path = _index_path(output_dir)
    data = orjson.dumps(entries, option=orjson.OPT_INDENT_2)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


def _build_review_markdown(title: str, issues: list[ValidationIssue], repair_rounds: int) -> str:
    lines = [
        f"# Needs human review: {title}",
        "",
        f"Unresolved after {repair_rounds} assembly-level repair round(s). No `.wiki` file was saved.",
        "",
        "| severity | line | explanation | snippet |",
        "|---|---|---|---|",
    ]
    for issue in issues:
        finding = issue.as_finding()
        line = finding["line_number"] if finding["line_number"] is not None else "?"
        snippet = (finding["snippet"] or "").replace("|", "\\|").replace("\n", " ")
        explanation = finding["explanation"].replace("|", "\\|")
        lines.append(f"| {finding['severity']} | {line} | {explanation} | `{snippet}` |")
    lines.append("")
    return "\n".join(lines)


def record_needs_human_review(
    output_dir: Path, title: str, issues: list[ValidationIssue], repair_rounds: int
) -> Path:
    """Writes {title}.review.md and upserts this article's entry in
    needs_human_review.json (keyed by title, so a re-run replaces the
    stale entry rather than accumulating duplicates). Returns the
    .review.md path."""
    output_dir.mkdir(parents=True, exist_ok=True)

    review_text = _build_review_markdown(title, issues, repair_rounds)
    review_path = review_path_for(output_dir, title)
    review_path.write_text(review_text, encoding="utf-8")

    entries = [e for e in _load_index(output_dir) if e.get("title") != title]
    entries.append(
        {
            "title": title,
            "timestamp": time.time(),
            "repair_rounds": repair_rounds,
            "findings": [issue.as_finding() for issue in issues],
        }
    )
    _write_index(output_dir, entries)

    return review_path


def _build_review_flags_markdown(title: str, issues: list[ValidationIssue], rounds: int) -> str:
    lines = [
        f"# Semantic review findings: {title}",
        "",
        f"Unresolved after {rounds} review-repair round(s). The `.wiki` file WAS saved — "
        "these are unresolved translation-fidelity/grammar concerns, not structural "
        "defects, so they don't block delivery. Spot-check before pasting to the live wiki.",
        "",
        "| severity | line | explanation | snippet |",
        "|---|---|---|---|",
    ]
    for issue in issues:
        finding = issue.as_finding()
        line = finding["line_number"] if finding["line_number"] is not None else "?"
        snippet = (finding["snippet"] or "").replace("|", "\\|").replace("\n", " ")
        explanation = finding["explanation"].replace("|", "\\|")
        lines.append(f"| {finding['severity']} | {line} | {explanation} | `{snippet}` |")
    lines.append("")
    return "\n".join(lines)


def record_review_flags(output_dir: Path, title: str, issues: list[ValidationIssue], rounds: int) -> Path:
    """Non-blocking counterpart to record_needs_human_review — writes
    {title}.review-flags.md but does not touch needs_human_review.json (that
    index is reserved for the blocking structural case) and does not
    prevent save_article from running. Returns the .review-flags.md path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    text = _build_review_flags_markdown(title, issues, rounds)
    path = review_flags_path_for(output_dir, title)
    path.write_text(text, encoding="utf-8")
    return path
