"""Typed data models shared across the harness.

No translation logic lives here — only the shapes that carry data between
the fetch / chunk / cache / invoke-skill / validate / repair / save stages.
"""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ArticleStatus(str, Enum):
    PENDING = "pending"
    FETCHED = "fetched"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ChunkStatus(str, Enum):
    PENDING = "pending"
    CACHED = "cached"
    TRANSLATED = "translated"
    REPAIRED = "repaired"
    FAILED = "failed"


class Section(BaseModel):
    """A logical section of an article, as split by parser.py."""

    title: str
    level: int
    order: int
    wikitext: str


class Chunk(BaseModel):
    """A translation unit: one or more merged sections, sized ~1500-2500 tokens.

    Small adjacent sections are merged into one chunk to amortize the fixed
    per-request skill-prompt cost; an oversized section is split internally
    (never inside a template/table/reference/list) and keeps only its own
    title across the resulting sub-chunks.
    """

    article_title: str
    section_titles: list[str]
    order: int
    text: str
    token_estimate: int
    source_lang: str = "en"
    status: ChunkStatus = ChunkStatus.PENDING
    translated_text: str | None = None

    @property
    def section_title(self) -> str:
        return "; ".join(self.section_titles)

    @property
    def chunk_id(self) -> str:
        return f"{self.article_title}::{self.order}"


class ValidationIssue(BaseModel):
    kind: str
    message: str


class ValidationResult(BaseModel):
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.valid


class TranslationResult(BaseModel):
    """Result of one skill invocation (translate or repair)."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cached: bool = False


class ModelPricing(BaseModel):
    model_id: str
    prompt_price_per_token: float
    completion_price_per_token: float


class ArticleSource(BaseModel):
    """Where an article came from and its raw wikitext, prior to chunking."""

    title: str
    wikitext: str
    revid: int | None = None
    revision_timestamp: str | None = None
    source_lang: str = "en"


class ArticleJob(BaseModel):
    """Tracks one article's progress through the pipeline."""

    title: str
    status: ArticleStatus = ArticleStatus.PENDING
    output_path: Path | None = None
    error: str | None = None
    sections_total: int = 0
    sections_done: int = 0


class RunStats(BaseModel):
    """Aggregate counters for stats.json, updated throughout a run."""

    started_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    articles_completed: int = 0
    articles_failed: int = 0
    articles_skipped: int = 0
    sections_translated: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    estimated_cost_usd: float = 0.0
    translation_time_s: float = 0.0
    validation_failures: int = 0
    repair_attempts: int = 0
    repairs_succeeded: int = 0
    repairs_failed: int = 0
    retries: int = 0


class Config(BaseModel):
    """Full harness configuration: config.yaml merged with CLI overrides."""

    provider: str = "openrouter"
    model: str = "deepseek/deepseek-v3"
    workers: int = 4
    temperature: float = 0.0
    max_retries: int = 5
    cache: bool = True
    validate_output: bool = Field(default=True, alias="validate")
    repair: bool = True
    max_repair_attempts: int = 2

    source_lang: str = "en"
    target_lang: str = "sq"

    # Optional override for the MediaWiki API endpoint used for source_lang
    # specifically (e.g. a non-Wikipedia wiki, or a custom mirror). Any other
    # source language encountered (via a `lang:Title` prefix or full URL —
    # see sources.parse_source_ref) resolves generically to
    # https://{lang}.wikipedia.org/w/api.php, not this override.
    source_wiki_api: str | None = None

    chunk_min_tokens: int = 1500
    chunk_max_tokens: int = 2500

    skill_path: Path = Path.home() / ".claude" / "skills" / "enwiki-sqwiki-translation"
    include_skill_references: bool = False
    # If set (e.g. "HEAD"), the skill is read from this git revision instead
    # of the working tree, so local uncommitted edits to the skill's repo
    # don't silently change translation behavior. skill_path must then sit
    # inside a git working copy of the skill's repo. None reads plain files.
    skill_git_ref: str | None = None

    output_dir: Path = Path("output")
    cache_db_path: Path = Path("cache") / "translation_memory.sqlite3"
    log_dir: Path = Path("logs")
    stats_path: Path = Path("stats.json")

    # Harness-side link/template verification against Wikidata + the target
    # wiki (see verification.py) — the harness's own, growing equivalent of
    # the skill's sqwiki-verified.md, fed into each chunk's translation
    # request as pre-checked facts instead of the model having to guess.
    verify_links: bool = True
    verification_db_path: Path = Path("cache") / "verified_facts.sqlite3"
    generate_reports: bool = True
    # Wikidata/target-wiki lookups should complete in a few seconds — this
    # is deliberately its own (short) setting, not request_timeout_s, which
    # is sized for slow LLM completions and would let a single hung lookup
    # run for two minutes before even hitting the outer deadline below.
    wikidata_timeout_s: float = 20.0
    # Hard outer deadline for the whole verification step, independent of
    # httpx's own per-call timeouts — a network call hanging past its
    # configured timeout must never stall an entire batch run.
    verification_timeout_s: float = 60.0

    # Fills any citation missing |language= in the final assembled article:
    # visits the cited URL and reads its declared language, falling back to
    # guessing from the citation title. See citation_language.py.
    fill_citation_languages: bool = True
    max_citation_url_fetches: int = 40
    citation_fetch_concurrency: int = 5
    citation_fetch_timeout_s: float = 10.0
    citation_fill_timeout_s: float = 60.0

    # Deterministic backstop: renames citation parameter *names* that got
    # mistranslated into the target language back to their English CS1
    # equivalents (e.g. |titulli= -> |title=). See citation_language.py.
    fix_citation_param_names: bool = True

    # Reconciles {{sfn}}/{{harvnb}} calls that share the same auto-generated
    # anchor (same author+year+page) but ended up with differently
    # -paraphrased |ps= quotes because the same source citation was split
    # across independently-translated chunks. See citation_language.py.
    dedupe_short_footnotes: bool = True

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Sized for the largest oversized-section chunks (never split further,
    # so a protected table/template can push a single chunk to 10k+ input
    # tokens). Confirmed directly against the raw API (bypassing this
    # harness entirely) that a real 12k-input/7k-output completion took
    # 375s end to end — genuine model inference latency, not a stuck
    # connection. 120s, then 300s, both proved short; 600s leaves real
    # headroom above the slowest observed case.
    request_timeout_s: float = 600.0

    # Wikimedia's User-Agent policy (foundation.wikimedia.org/wiki/Policy:User-Agent_policy)
    # requires automated requests to self-identify with a contact (email or URL) so the
    # operator can be reached — unidentified bulk traffic risks throttling/blocking.
    # wikimedia_contact has no default on purpose; config.py refuses to build a Config
    # without it (or an explicit user_agent override) rather than send anonymous traffic.
    wikimedia_tool_name: str = "wiki-translate-harness"
    wikimedia_contact: str | None = None
    user_agent: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    @field_validator(
        "skill_path",
        "output_dir",
        "cache_db_path",
        "log_dir",
        "stats_path",
        "verification_db_path",
        mode="before",
    )
    @classmethod
    def _expand_user(cls, v: object) -> object:
        if isinstance(v, (str, Path)):
            return Path(v).expanduser()
        return v
