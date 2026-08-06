"""SQLite translation memory.

Cache key: SHA256 of (model, source_lang, target_lang, skill content hash,
section text). The model and language pair are folded into the key (not
just the raw section text) so that switching models — e.g. benchmark mode,
comparing several OpenRouter models on the same article — never serves one
model's output as a cache hit for another; content-only keying would
silently zero out cost on every model after the first. The skill content
hash is folded in too so editing the skill file or the harness's fixed
invocation framing invalidates stale entries instead of silently replaying
a translation made under a different prompt.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

import orjson


def compute_key(
    model: str,
    source_lang: str,
    target_lang: str,
    text: str,
    skill_hash: str = "",
    facts_hash: str = "",
) -> str:
    """skill_hash fingerprints the skill body + invocation framing (see
    skill_loader.SkillContent.content_hash) so editing the skill or its
    framing invalidates stale entries instead of silently reusing a
    translation made under a different prompt. facts_hash does the same for
    the harness's pre-verified link/template facts appended to the request
    (see verification.build_verified_facts_block) — as Wikidata/target-wiki
    lookups resolve more titles over time, a chunk should be retranslated
    with the newly-available facts rather than silently reusing an older
    translation made without them."""
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(source_lang.encode("utf-8"))
    h.update(b"\x00")
    h.update(target_lang.encode("utf-8"))
    h.update(b"\x00")
    h.update(skill_hash.encode("utf-8"))
    h.update(b"\x00")
    h.update(facts_hash.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class TranslationCache:
    """Synchronous SQLite-backed cache.

    Calls are local-disk-only and fast; the harness runs a single-threaded
    asyncio event loop, so blocking briefly here (no worker threads) is
    safe and keeps sqlite3's connection out of cross-thread territory.
    """

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS translations (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                source_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                created_at REAL NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.commit()

    def get(self, key: str) -> str | None:
        cur = self._conn.execute(
            "SELECT translated_text FROM translations WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        self._conn.execute(
            "UPDATE translations SET hit_count = hit_count + 1 WHERE key = ?", (key,)
        )
        self._conn.commit()
        return row[0]

    def set(
        self,
        key: str,
        model: str,
        source_lang: str,
        target_lang: str,
        source_text: str,
        translated_text: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO translations (key, model, source_lang, target_lang, source_text, translated_text, created_at, hit_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(key) DO UPDATE SET translated_text = excluded.translated_text
            """,
            (key, model, source_lang, target_lang, source_text, translated_text, time.time()),
        )
        self._conn.commit()

    def size(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM translations")
        return cur.fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TranslationCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class VerificationCache:
    """Persistent store of link/template existence checks and infobox
    parameter lists — the harness's own, growing equivalent of the skill's
    sqwiki-verified.md reference file. Grows across the whole batch and
    across future runs, so the same title/template is never looked up
    against Wikidata (or fetched from the target wiki) twice.
    """

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS link_verification (
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                kind TEXT NOT NULL,
                source_title TEXT NOT NULL,
                target_title TEXT,
                checked_at REAL NOT NULL,
                PRIMARY KEY (source_lang, target_lang, kind, source_title)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS template_params (
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                source_template TEXT NOT NULL,
                target_template TEXT,
                params_json TEXT,
                checked_at REAL NOT NULL,
                PRIMARY KEY (source_lang, target_lang, source_template)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sibling_links (
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                article_title TEXT NOT NULL,
                existing_target_title TEXT,
                links_json TEXT,
                checked_at REAL NOT NULL,
                PRIMARY KEY (source_lang, target_lang, article_title)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sitelink_counts (
                source_lang TEXT NOT NULL,
                source_title TEXT NOT NULL,
                other_wiki_count INTEGER NOT NULL,
                checked_at REAL NOT NULL,
                PRIMARY KEY (source_lang, source_title)
            )
            """
        )
        self._conn.commit()

    def get_many(
        self, source_lang: str, target_lang: str, kind: str, titles: list[str]
    ) -> dict[str, str | None]:
        """Only returns entries already cached; absent keys mean 'not yet
        checked', not 'checked and missing' — check membership, not truthiness."""
        if not titles:
            return {}
        results: dict[str, str | None] = {}
        placeholders = ",".join("?" for _ in titles)
        cur = self._conn.execute(
            f"""
            SELECT source_title, target_title FROM link_verification
            WHERE source_lang = ? AND target_lang = ? AND kind = ? AND source_title IN ({placeholders})
            """,
            (source_lang, target_lang, kind, *titles),
        )
        for source_title, target_title in cur.fetchall():
            results[source_title] = target_title
        return results

    def set_many(
        self, source_lang: str, target_lang: str, kind: str, results: dict[str, str | None]
    ) -> None:
        now = time.time()
        self._conn.executemany(
            """
            INSERT INTO link_verification (source_lang, target_lang, kind, source_title, target_title, checked_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_lang, target_lang, kind, source_title)
            DO UPDATE SET target_title = excluded.target_title, checked_at = excluded.checked_at
            """,
            [(source_lang, target_lang, kind, title, target, now) for title, target in results.items()],
        )
        self._conn.commit()

    def get_template_params(
        self, source_lang: str, target_lang: str, source_template: str
    ) -> list[str] | None:
        cur = self._conn.execute(
            """
            SELECT params_json FROM template_params
            WHERE source_lang = ? AND target_lang = ? AND source_template = ?
            """,
            (source_lang, target_lang, source_template),
        )
        row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        return orjson.loads(row[0])

    def set_template_params(
        self,
        source_lang: str,
        target_lang: str,
        source_template: str,
        target_template: str | None,
        params: list[str],
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO template_params (source_lang, target_lang, source_template, target_template, params_json, checked_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_lang, target_lang, source_template)
            DO UPDATE SET target_template = excluded.target_template, params_json = excluded.params_json, checked_at = excluded.checked_at
            """,
            (source_lang, target_lang, source_template, target_template, orjson.dumps(params).decode(), time.time()),
        )
        self._conn.commit()

    def get_sibling_links(
        self, source_lang: str, target_lang: str, article_title: str
    ) -> tuple[str | None, list[str]] | None:
        """Returns (existing_target_title, sibling_links), or None if this
        article was never checked. existing_target_title is None but the
        tuple itself is non-None when the check ran and found no existing
        target-wiki article — distinct from "not checked yet"."""
        cur = self._conn.execute(
            """
            SELECT existing_target_title, links_json FROM sibling_links
            WHERE source_lang = ? AND target_lang = ? AND article_title = ?
            """,
            (source_lang, target_lang, article_title),
        )
        row = cur.fetchone()
        if row is None:
            return None
        existing_target_title, links_json = row
        links = orjson.loads(links_json) if links_json else []
        return existing_target_title, links

    def set_sibling_links(
        self,
        source_lang: str,
        target_lang: str,
        article_title: str,
        existing_target_title: str | None,
        links: list[str],
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO sibling_links (source_lang, target_lang, article_title, existing_target_title, links_json, checked_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_lang, target_lang, article_title)
            DO UPDATE SET existing_target_title = excluded.existing_target_title, links_json = excluded.links_json, checked_at = excluded.checked_at
            """,
            (source_lang, target_lang, article_title, existing_target_title, orjson.dumps(links).decode(), time.time()),
        )
        self._conn.commit()

    def get_sitelink_counts(self, source_lang: str, titles: list[str]) -> dict[str, int]:
        """Only returns titles already cached — absent keys mean 'not yet
        checked', matching get_many's contract."""
        if not titles:
            return {}
        results: dict[str, int] = {}
        placeholders = ",".join("?" for _ in titles)
        cur = self._conn.execute(
            f"""
            SELECT source_title, other_wiki_count FROM sitelink_counts
            WHERE source_lang = ? AND source_title IN ({placeholders})
            """,
            (source_lang, *titles),
        )
        for source_title, count in cur.fetchall():
            results[source_title] = count
        return results

    def set_sitelink_counts(self, source_lang: str, counts: dict[str, int]) -> None:
        now = time.time()
        self._conn.executemany(
            """
            INSERT INTO sitelink_counts (source_lang, source_title, other_wiki_count, checked_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source_lang, source_title)
            DO UPDATE SET other_wiki_count = excluded.other_wiki_count, checked_at = excluded.checked_at
            """,
            [(source_lang, title, count, now) for title, count in counts.items()],
        )
        self._conn.commit()

    def export_markdown(self, source_lang: str, target_lang: str) -> str:
        """A human-readable snapshot of confirmed entries, in the spirit of
        the skill's own sqwiki-verified.md reference file."""
        lines = [f"# Verified {source_lang} -> {target_lang} facts (harness-collected)\n"]

        lines.append("## Links\n")
        lines.append("| Source title | Target title |")
        lines.append("|---|---|")
        cur = self._conn.execute(
            """
            SELECT source_title, target_title FROM link_verification
            WHERE source_lang = ? AND target_lang = ? AND kind = 'link' AND target_title IS NOT NULL
            ORDER BY source_title
            """,
            (source_lang, target_lang),
        )
        for source_title, target_title in cur.fetchall():
            lines.append(f"| {source_title} | {target_title} |")

        lines.append("\n## Templates\n")
        lines.append("| Source template | Target template |")
        lines.append("|---|---|")
        cur = self._conn.execute(
            """
            SELECT source_title, target_title FROM link_verification
            WHERE source_lang = ? AND target_lang = ? AND kind = 'template' AND target_title IS NOT NULL
            ORDER BY source_title
            """,
            (source_lang, target_lang),
        )
        for source_title, target_title in cur.fetchall():
            lines.append(f"| {source_title} | {target_title} |")

        lines.append("\n## Template parameters\n")
        cur = self._conn.execute(
            """
            SELECT source_template, target_template, params_json FROM template_params
            WHERE source_lang = ? AND target_lang = ? AND params_json IS NOT NULL
            ORDER BY source_template
            """,
            (source_lang, target_lang),
        )
        for source_template, target_template, params_json in cur.fetchall():
            params = orjson.loads(params_json)
            lines.append(f"### `{{{{{source_template}}}}}` -> `{{{{{target_template}}}}}`\n")
            lines.append(", ".join(f"`{p}`" for p in params) + "\n")

        return "\n".join(lines)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "VerificationCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
