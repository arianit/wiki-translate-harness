import subprocess
from pathlib import Path

import pytest

from wiki_translate_harness.skill_loader import (
    SkillGitError,
    build_repair_messages,
    build_translation_messages,
    load_skill,
)


def _make_skill_dir(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "enwiki-sqwiki-translation"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: test\n---\n\n# Body\n\nTranslation rules here.\n"
    )
    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "notes.md").write_text("Some verified reference notes.")
    return skill_dir


def test_frontmatter_stripped(tmp_path: Path):
    skill_dir = _make_skill_dir(tmp_path)
    skill = load_skill(skill_dir, include_references=False)
    assert "name: test-skill" not in skill.skill_md
    assert "Translation rules here." in skill.skill_md


def test_references_excluded_by_default(tmp_path: Path):
    skill_dir = _make_skill_dir(tmp_path)
    skill = load_skill(skill_dir, include_references=False)
    assert skill.reference_texts == {}
    assert "verified reference notes" not in skill.combined


def test_references_included_when_requested(tmp_path: Path):
    skill_dir = _make_skill_dir(tmp_path)
    skill = load_skill(skill_dir, include_references=True)
    assert "notes.md" in skill.reference_texts
    assert "verified reference notes" in skill.combined


def test_missing_skill_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_skill(tmp_path / "does-not-exist", include_references=False)


def test_translation_messages_contain_skill_and_input(tmp_path: Path):
    skill_dir = _make_skill_dir(tmp_path)
    skill = load_skill(skill_dir, include_references=False)
    messages = build_translation_messages(skill, "en", "sq", "Paris", "Lead", "Paris is a city.")
    assert messages[0]["role"] == "system"
    assert "Translation rules here." in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Paris is a city." in messages[1]["content"]
    assert "Source language: en" in messages[1]["content"]
    assert "Target language: sq" in messages[1]["content"]


def test_harness_never_embeds_translation_prompt(tmp_path: Path):
    """The system prompt's translation-relevant content must come only from
    the skill file, not from any harness-authored text."""
    skill_dir = _make_skill_dir(tmp_path)
    skill = load_skill(skill_dir, include_references=False)
    messages = build_translation_messages(skill, "en", "sq", "Paris", "Lead", "text")
    system_content = messages[0]["content"]
    # everything besides the skill body should be pure invocation framing —
    # spot check that no grammar/vocabulary guidance appears outside the skill
    framing_only = system_content.split("Translation rules here.")[0]
    assert "case" not in framing_only.lower() or "no tools" in framing_only.lower()


def test_content_hash_stable_for_same_content(tmp_path: Path):
    skill_dir = _make_skill_dir(tmp_path)
    skill_a = load_skill(skill_dir, include_references=False)
    skill_b = load_skill(skill_dir, include_references=False)
    assert skill_a.content_hash == skill_b.content_hash


def test_content_hash_changes_when_skill_body_edited(tmp_path: Path):
    skill_dir = _make_skill_dir(tmp_path)
    before = load_skill(skill_dir, include_references=False)

    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: test\n---\n\n# Body\n\nDifferent rules now.\n"
    )
    after = load_skill(skill_dir, include_references=False)

    assert before.content_hash != after.content_hash


def test_content_hash_changes_when_references_toggled(tmp_path: Path):
    skill_dir = _make_skill_dir(tmp_path)
    without_refs = load_skill(skill_dir, include_references=False)
    with_refs = load_skill(skill_dir, include_references=True)
    assert without_refs.content_hash != with_refs.content_hash


def test_repair_messages_include_errors(tmp_path: Path):
    skill_dir = _make_skill_dir(tmp_path)
    skill = load_skill(skill_dir, include_references=False)
    messages = build_repair_messages(
        skill, "en", "sq", "Paris", "Lead", "[[broken", ["link: unbalanced"]
    )
    assert "link: unbalanced" in messages[1]["content"]
    assert "[[broken" in messages[1]["content"]


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _make_git_skill_repo(tmp_path: Path) -> Path:
    """A repo laid out like the real skill: a subdirectory (mimicking the
    symlink target under ~/.claude/skills/) holding SKILL.md + references,
    with one commit, then an uncommitted local edit on top."""
    repo_root = tmp_path / "skill-repo"
    repo_root.mkdir()
    _git(["init", "-q"], cwd=repo_root)
    _git(["config", "user.email", "test@example.com"], cwd=repo_root)
    _git(["config", "user.name", "Test"], cwd=repo_root)

    skill_dir = repo_root / "enwiki-sqwiki-translation"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: test\n---\n\n# Body\n\nCommitted rules.\n"
    )
    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "notes.md").write_text("Committed reference notes.")

    _git(["add", "-A"], cwd=repo_root)
    _git(["commit", "-q", "-m", "initial"], cwd=repo_root)

    # dirty the working copy after the commit, uncommitted
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: test\n---\n\n# Body\n\nUNCOMMITTED local edit.\n"
    )
    (refs / "notes.md").write_text("UNCOMMITTED reference edit.")

    return skill_dir


def test_git_ref_reads_committed_content_not_working_copy(tmp_path: Path):
    skill_dir = _make_git_skill_repo(tmp_path)
    skill = load_skill(skill_dir, include_references=False, git_ref="HEAD")
    assert "Committed rules." in skill.skill_md
    assert "UNCOMMITTED" not in skill.skill_md


def test_git_ref_none_reads_dirty_working_copy(tmp_path: Path):
    skill_dir = _make_git_skill_repo(tmp_path)
    skill = load_skill(skill_dir, include_references=False, git_ref=None)
    assert "UNCOMMITTED local edit." in skill.skill_md
    assert "Committed rules." not in skill.skill_md


def test_git_ref_reads_committed_references(tmp_path: Path):
    skill_dir = _make_git_skill_repo(tmp_path)
    skill = load_skill(skill_dir, include_references=True, git_ref="HEAD")
    assert "notes.md" in skill.reference_texts
    assert "Committed reference notes." in skill.reference_texts["notes.md"]
    assert "UNCOMMITTED" not in skill.combined


def test_git_ref_bad_revision_raises(tmp_path: Path):
    skill_dir = _make_git_skill_repo(tmp_path)
    with pytest.raises(SkillGitError):
        load_skill(skill_dir, include_references=False, git_ref="not-a-real-ref")


def test_git_ref_content_hash_differs_from_working_copy(tmp_path: Path):
    skill_dir = _make_git_skill_repo(tmp_path)
    committed = load_skill(skill_dir, include_references=False, git_ref="HEAD")
    working = load_skill(skill_dir, include_references=False, git_ref=None)
    assert committed.content_hash != working.content_hash
