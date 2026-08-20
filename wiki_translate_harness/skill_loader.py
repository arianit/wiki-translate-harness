"""Loads the Pi translation skill(s) and turns them into an LLM request.

This is the harness's *only* bridge to translation judgment. The harness
itself contributes no translation guidance — it reads the skill's own
instructions (SKILL.md, optionally its reference files) verbatim and uses
that text as the system prompt for the OpenRouter call. By default this
reads the working tree on disk; if `skill_git_ref` is configured (e.g.
"HEAD"), it instead reads the skill from that committed git revision, so
local uncommitted edits to the skill's own repo can't silently change
translation behavior underneath the harness.

The skill (github.com/arianit/enwiki-sqwiki-translation) is split across
three directories — enwiki-sqwiki-translation (translate), wikiterms
(terminology/link verification), wikiqa (pre-delivery checklist) — that
were written to be followed by an interactive agent with shell/tool access
(curl, grep, Write) and to invoke each other via a Skill tool: it fetches
its own sources, batch-verifies links against Wikidata, checks nearby
sqwiki articles, etc. A single OpenRouter chat-completion call has none of
that: no tools, no internet, no Skill tool, one isolated section at a
time. `skill_path` therefore accepts either one directory or a list of
directories; when it's a list, each directory's SKILL.md (and, if
`include_references` is set, its `references/*.md` files) are concatenated
in the given order, so the model sees the combined content of all three
skills in one system prompt instead of being able to invoke them on
demand. The small "invocation frame" below exists solely to tell the model
what context it is running in and what to skip — it carries no opinion
about *how* to translate. All translation judgment (grammar, terminology,
conventions, what to link, how to transliterate) still comes entirely from
the skill text itself.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)

# Fixed operational framing only — zero translation content. This is what
# makes single-shot, tool-less invocation of an interactive skill possible
# at all; it does not tell the model how to translate anything.
_INVOCATION_FRAME = """\
You are being invoked as an automated, non-interactive translation backend \
by a batch harness. You have no internet access, no shell, and no tools in \
this call — you cannot run curl, grep, or fetch Wikidata/sqwiki pages \
yourself. The harness has already fetched the source article, split it into \
sections, and handles verification, retries, caching, and file output \
outside of this request.

Below are the full working instructions for this translation task (the \
"skill"). Apply every part of them that concerns translation judgment: \
grammar, case/inflection, vocabulary choice, transliteration, citation \
formatting, conventions, and what to preserve or drop. Skip only the parts \
that describe live research actions you cannot perform here (running curl, \
grep, batch Wikidata/API lookups, writing files, spawning subagents, asking \
the user a question) — where such a step would normally decide something \
(e.g. whether a link target exists), make the best judgment call the text \
allows without that lookup, since it is not available in this call.

You will be given exactly one section of one article at a time, not the \
whole article. Maintain terminology consistency using only what is visible \
in this section — the harness is responsible for whole-article consistency \
across sections. Do not treat this section as if it were the complete, \
final output file: skip any instruction in the skill about what to append \
at the end of "every output file" (e.g. a trailing attribution/talk-page \
comment block, an edit-summary reminder, or similar file-level framing) — \
the harness assembles the complete article from many such calls and handles \
end-of-file concerns itself, outside of this request. Adding that block to \
an individual section would corrupt the reassembled article by inserting it \
mid-document.

Output ONLY the translated MediaWiki wikitext for the given input section, \
nothing more and nothing less. No commentary, no explanation, no markdown \
code fences, no preamble or sign-off, no file-level framing.
"""

_REPAIR_FRAME = """\
You are being invoked as an automated, non-interactive syntax-repair backend \
by a batch harness, using the same skill instructions below for style and \
convention context. A previous translation pass produced MediaWiki wikitext \
that failed mechanical validation (unbalanced templates, links, tables, \
references, or comments).

Fix ONLY the listed structural syntax errors. Do not re-translate, do not \
change wording, do not alter meaning, and do not touch anything that is not \
implicated by the listed errors. You have no internet access or tools in \
this call.

Output ONLY the corrected MediaWiki wikitext. No commentary, no explanation, \
no markdown code fences, no preamble or sign-off.
"""


def _strip_frontmatter(text: str) -> str:
    return _FRONTMATTER_RE.sub("", text, count=1)


@dataclass(frozen=True)
class SkillContent:
    skill_md: str
    reference_texts: dict[str, str]

    @property
    def combined(self) -> str:
        parts = [self.skill_md]
        for name, text in sorted(self.reference_texts.items()):
            parts.append(f"\n\n# Reference: {name}\n\n{text}")
        return "".join(parts)

    @property
    def content_hash(self) -> str:
        """Fingerprint of everything that shapes the model's output besides
        the section text itself: the skill body/references plus the fixed
        invocation/repair framing. Used as part of the cache key so editing
        the skill or the framing naturally invalidates stale cache entries
        instead of silently serving translations made under an old prompt."""
        h = hashlib.sha256()
        h.update(self.combined.encode("utf-8"))
        h.update(b"\x00")
        h.update(_INVOCATION_FRAME.encode("utf-8"))
        h.update(b"\x00")
        h.update(_REPAIR_FRAME.encode("utf-8"))
        return h.hexdigest()[:16]


class SkillGitError(Exception):
    pass


def _run_git(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise SkillGitError(f"git command failed ({' '.join(args)}): {result.stderr.strip()}")
    return result.stdout


def _git_root_and_prefix(skill_path: Path) -> tuple[Path, str]:
    """skill_path may be a symlink into a subdirectory of a git repo (as with
    a Claude Code skill symlinked from ~/.claude/skills/). Resolve the repo
    root and skill_path's path prefix within it, so committed content can be
    read via `git show <ref>:<prefix><file>` regardless of cwd/symlinks."""
    root = _run_git(["git", "-C", str(skill_path), "rev-parse", "--show-toplevel"]).strip()
    prefix = _run_git(["git", "-C", str(skill_path), "rev-parse", "--show-prefix"]).strip()
    return Path(root), prefix


def _normalize_skill_paths(skill_path: Path | str | Sequence[Path | str]) -> list[Path]:
    """A single directory is the common case; a list lets skill_path span
    multiple skill directories (e.g. translate + wikiterms + wikiqa) that
    get concatenated into one system prompt, since this harness has no
    Skill-tool equivalent to invoke them on demand at runtime."""
    if isinstance(skill_path, (str, Path)):
        return [Path(skill_path)]
    return [Path(p) for p in skill_path]


def _load_skill_from_git(
    skill_paths: Sequence[Path], git_ref: str, include_references: bool
) -> SkillContent:
    skill_bodies: list[str] = []
    reference_texts: dict[str, str] = {}
    # Only namespace reference filenames by their source skill directory when
    # loading more than one skill — a single skill_path (the common case,
    # and what the pre-split test suite exercises) keeps bare filenames.
    prefix_keys = len(skill_paths) > 1

    for skill_path in skill_paths:
        root, prefix = _git_root_and_prefix(skill_path)

        try:
            raw_skill_md = _run_git(["git", "-C", str(root), "show", f"{git_ref}:{prefix}SKILL.md"])
        except SkillGitError as exc:
            raise SkillGitError(
                f"Could not read SKILL.md at ref {git_ref!r} in {root} "
                f"(path {prefix}SKILL.md within the repo): {exc}"
            ) from exc
        skill_bodies.append(_strip_frontmatter(raw_skill_md).strip())

        if include_references:
            listing = _run_git(
                ["git", "-C", str(root), "ls-tree", "--name-only", git_ref, "--", f"{prefix}references/"]
            )
            for line in listing.splitlines():
                line = line.strip()
                if not line or not line.endswith(".md"):
                    continue
                content = _run_git(["git", "-C", str(root), "show", f"{git_ref}:{line}"]).strip()
                name = Path(line).name
                # Prefixed with the source skill's directory name so that,
                # e.g., wikiterms/sqwiki-verified.md and a same-named file
                # from another skill directory can't collide.
                reference_texts[f"{skill_path.name}/{name}" if prefix_keys else name] = content

    return SkillContent(
        skill_md="\n\n---\n\n".join(skill_bodies), reference_texts=reference_texts
    )


def load_skill(
    skill_path: Path | str | Sequence[Path | str],
    include_references: bool = False,
    git_ref: str | None = None,
) -> SkillContent:
    """Loads the skill(s) from disk. skill_path may be a single directory or
    a list of directories (each containing its own SKILL.md, and optionally
    a references/ directory) — in the list case, each is loaded in the
    given order and concatenated into one SkillContent, since a single-shot
    call has no way to invoke a companion skill on demand the way an
    interactive agent would. If git_ref is set (e.g. "HEAD"), reads the
    skill(s) from that committed git revision instead of the working tree —
    so local uncommitted edits in the skill's own repo don't silently change
    translation behavior underneath the harness. Each directory must then
    sit inside a git working copy of its repo."""
    skill_paths = _normalize_skill_paths(skill_path)

    if git_ref is not None:
        return _load_skill_from_git(skill_paths, git_ref, include_references)

    skill_bodies: list[str] = []
    reference_texts: dict[str, str] = {}
    # Only namespace reference filenames by their source skill directory when
    # loading more than one skill — a single skill_path (the common case,
    # and what the pre-split test suite exercises) keeps bare filenames.
    prefix_keys = len(skill_paths) > 1

    for path in skill_paths:
        skill_md_path = path / "SKILL.md"
        if not skill_md_path.exists():
            raise FileNotFoundError(
                f"Pi skill not found at {skill_md_path}. Set 'skill_path' in config.yaml "
                "to the directory (or list of directories) containing SKILL.md."
            )
        skill_bodies.append(_strip_frontmatter(skill_md_path.read_text(encoding="utf-8")).strip())

        if include_references:
            ref_dir = path / "references"
            if ref_dir.is_dir():
                for ref_file in sorted(ref_dir.glob("*.md")):
                    key = f"{path.name}/{ref_file.name}" if prefix_keys else ref_file.name
                    reference_texts[key] = ref_file.read_text(encoding="utf-8").strip()

    return SkillContent(
        skill_md="\n\n---\n\n".join(skill_bodies), reference_texts=reference_texts
    )


@lru_cache(maxsize=8)
def load_skill_cached(
    skill_path: str | tuple[str, ...], include_references: bool, git_ref: str | None = None
) -> SkillContent:
    paths: list[str] = [skill_path] if isinstance(skill_path, str) else list(skill_path)
    return load_skill(paths, include_references, git_ref)


def build_translation_messages(
    skill: SkillContent,
    source_lang: str,
    target_lang: str,
    article_title: str,
    section_title: str,
    text: str,
    verified_facts_block: str = "",
) -> list[dict[str, str]]:
    """verified_facts_block (see verification.build_verified_facts_block) is
    harness-fetched data — confirmed link/template existence and real
    parameter names — not translation guidance. It's appended as input
    alongside the source text, the same way source_lang/target_lang are."""
    system = f"{_INVOCATION_FRAME}\n\n---\n\n{skill.combined}"
    user = (
        f"Source language: {source_lang}\n"
        f"Target language: {target_lang}\n"
        f"Article title: {article_title}\n"
        f"Section: {section_title}\n\n"
        f"--- BEGIN SOURCE WIKITEXT ---\n{text}\n--- END SOURCE WIKITEXT ---"
    )
    if verified_facts_block:
        user += f"\n\n{verified_facts_block}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_repair_messages(
    skill: SkillContent,
    source_lang: str,
    target_lang: str,
    article_title: str,
    section_title: str,
    invalid_text: str,
    errors: list[str],
) -> list[dict[str, str]]:
    system = f"{_REPAIR_FRAME}\n\n---\n\n{skill.combined}"
    error_list = "\n".join(f"- {e}" for e in errors)
    user = (
        f"Source language: {source_lang}\n"
        f"Target language: {target_lang}\n"
        f"Article title: {article_title}\n"
        f"Section: {section_title}\n\n"
        f"Validation errors found in the translated wikitext below:\n{error_list}\n\n"
        f"--- BEGIN INVALID TRANSLATED WIKITEXT ---\n{invalid_text}\n--- END INVALID TRANSLATED WIKITEXT ---"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
