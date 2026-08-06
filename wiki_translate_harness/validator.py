"""Post-translation wikitext syntax validation.

Mechanical structural checks only — never semantic/translation-quality
checks. Uses mwparserfromhell to catch hard parse failures, plus explicit
marker-balance counts for the categories the spec calls out (templates,
links, tables, references, comments), since mwparserfromhell's parser is
deliberately lenient and silently treats an unmatched brace/bracket as plain
text rather than raising — the raw counts are what actually surface a
translation that dropped or duplicated a delimiter.
"""

from __future__ import annotations

import re
import zlib

import mwparserfromhell as mwp

from wiki_translate_harness.models import ValidationIssue, ValidationResult

_REF_OPEN_RE = re.compile(r"<ref\b[^>]*>", re.IGNORECASE)
_REF_CLOSE_RE = re.compile(r"</ref\s*>", re.IGNORECASE)
_HEADING_LIKE_RE = re.compile(r"(={2,6})([^\n=]+)\1[ \t]*(?=\n|$)")

# Confirmed in practice, twice: a chunk given little or no real source
# content (see the parser.py fix for the near-empty-trailing-chunk case
# that caused this) didn't just echo back what little it had — the model
# fabricated a plausible-looking replacement section, and separately kept
# leaking the skill's whole-article "end of file" attribution block into
# individual per-chunk output despite the invocation frame telling it not
# to (skill_loader._INVOCATION_FRAME). Both produce content with zero
# legitimate reason to appear in a chunk's translated wikitext, so they're
# checked as exact/near-exact known strings — mechanical detection, not a
# judgment call about translation quality.
_LEAKED_META_MARKERS = [
    "NUK ËSHTË PJESË E ARTIKULLIT",
    "Për faqen e diskutimit (Talk)",
    "{{Përkthyer nga|",
    "Nuk ka tekst burimor",
    "Ju lutem siguroni tekstin",
    "as an ai language model",
    "i cannot translate",
    "i don't have access to",
    "there is no source text",
    "please provide the text",
    "the provided wikitext appears to be",
    "i'll provide a clean version",
    "doesn't form meaningful content",
]


def _check_pair(text: str, open_tok: str, close_tok: str, kind: str, issues: list[ValidationIssue]) -> None:
    o = text.count(open_tok)
    c = text.count(close_tok)
    if o != c:
        issues.append(
            ValidationIssue(
                kind=kind,
                message=f"Unbalanced {kind}: {o} occurrences of '{open_tok}' vs {c} of '{close_tok}'",
            )
        )


def _check_refs(text: str, issues: list[ValidationIssue]) -> None:
    opens = _REF_OPEN_RE.findall(text)
    non_self_closing = [o for o in opens if not o.rstrip().endswith("/>")]
    closes = _REF_CLOSE_RE.findall(text)
    if len(non_self_closing) != len(closes):
        issues.append(
            ValidationIssue(
                kind="reference",
                message=(
                    f"Unbalanced <ref> tags: {len(non_self_closing)} opening vs "
                    f"{len(closes)} closing </ref>"
                ),
            )
        )


def _check_headings_at_line_start(text: str, issues: list[ValidationIssue]) -> None:
    """A `==Heading==`-shaped run not at the start of a line silently fails to
    render as a heading — MediaWiki only recognizes it at line start, and so
    does mwparserfromhell (it just falls back to plain Text, so parsing
    alone won't surface this). Scanned directly on raw text rather than via
    mwparserfromhell's node tree, since a merged/broken heading is exactly
    the case where mwparserfromhell does *not* produce a Heading node."""
    for match in _HEADING_LIKE_RE.finditer(text):
        start = match.start()
        if start > 0 and text[start - 1] != "\n":
            issues.append(
                ValidationIssue(
                    kind="heading",
                    message=f"{match.group(0).strip()!r} does not start at the beginning of a line",
                )
            )


def _check_leaked_meta_commentary(text: str, issues: list[ValidationIssue]) -> None:
    lowered = text.lower()
    for marker in _LEAKED_META_MARKERS:
        if marker.lower() in lowered:
            issues.append(
                ValidationIssue(
                    kind="leaked_commentary",
                    message=f"Output contains {marker!r} — model commentary/meta-text instead of translated wikitext",
                )
            )


# A degenerate repetition loop (the model gets stuck echoing the same line
# or short phrase thousands of times) is a distinct failure mode from
# leaked commentary — no fixed phrase to match against, since the repeated
# content itself varies. Confirmed in practice on a real large chunk: real
# translated wikitext (including large, formatting-heavy tables) compresses
# to roughly 30-47% of its raw size, while two independently observed
# degenerate loops compressed to under 3%. zlib compression ratio is a
# language-agnostic, content-agnostic way to catch this mechanically.
_MIN_LEN_FOR_REPETITION_CHECK = 500
_MAX_COMPRESSION_RATIO = 0.15


def _check_degenerate_repetition(text: str, issues: list[ValidationIssue]) -> None:
    raw = text.encode("utf-8")
    if len(raw) < _MIN_LEN_FOR_REPETITION_CHECK:
        return
    ratio = len(zlib.compress(raw, level=9)) / len(raw)
    if ratio < _MAX_COMPRESSION_RATIO:
        issues.append(
            ValidationIssue(
                kind="degenerate_repetition",
                message=(
                    f"Output compresses to {ratio:.1%} of its raw size — "
                    "a pathological repetition loop, not real translated content"
                ),
            )
        )


def validate_wikitext(text: str) -> ValidationResult:
    issues: list[ValidationIssue] = []

    try:
        mwp.parse(text)
    except Exception as exc:  # mwparserfromhell.parser.ParserError and friends
        issues.append(
            ValidationIssue(kind="parse_error", message=f"mwparserfromhell failed to parse output: {exc}")
        )

    _check_leaked_meta_commentary(text, issues)
    _check_degenerate_repetition(text, issues)
    _check_pair(text, "{{", "}}", "template", issues)
    _check_pair(text, "[[", "]]", "link", issues)
    _check_pair(text, "{|", "|}", "table", issues)
    _check_pair(text, "<!--", "-->", "comment", issues)
    _check_refs(text, issues)
    _check_headings_at_line_start(text, issues)

    return ValidationResult(valid=len(issues) == 0, issues=issues)


def format_errors(result: ValidationResult) -> list[str]:
    return [f"{issue.kind}: {issue.message}" for issue in result.issues]
