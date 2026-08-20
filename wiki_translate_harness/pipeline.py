"""Concurrent multi-article orchestration: fetch -> chunk -> translate -> assemble -> save.

Ties together mediawiki.py, parser.py, cache.py, translator.py, and
output.py. Contains no translation logic of its own.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from wiki_translate_harness.cache import TranslationCache, VerificationCache
from wiki_translate_harness.engines import LLMEngineClient, build_llm_client
from wiki_translate_harness.citation_language import (
    dedupe_short_footnotes,
    fill_missing_citation_languages,
    fix_citation_param_names,
    fix_sfn_param_names,
)
from wiki_translate_harness.report import build_attribution_block
from wiki_translate_harness.live_validator import validate_wikitext_live
from wiki_translate_harness.mediawiki import MediaWikiClient, MediaWikiClientPool
from wiki_translate_harness.models import (
    ArticleSource,
    Chunk,
    ChunkStatus,
    Config,
    EngineError,
    ModelPricing,
    RunStats,
    ValidationIssue,
    ValidationResult,
)
from wiki_translate_harness.openrouter import RetryCallback
from wiki_translate_harness.output import article_already_done, assemble_chunks, assemble_chunks_with_spans, save_article
from wiki_translate_harness.parser import build_chunks, split_into_sections
from wiki_translate_harness.progress import ProgressReporter
from wiki_translate_harness.report import ArticleReportData, build_article_report, save_report
from wiki_translate_harness.repair import repair_chunk
from wiki_translate_harness.review_queue import record_needs_human_review
from wiki_translate_harness.skill_loader import SkillContent, load_skill
from wiki_translate_harness.sources import ArticleInput, load_article_source
from wiki_translate_harness.statistics import StatsTracker
from wiki_translate_harness.translator import translate_chunk
from wiki_translate_harness.validator import format_errors, validate_wikitext
from wiki_translate_harness.verification import VerifiedFacts, verify_wikitext

logger = logging.getLogger("wiki_translate_harness.pipeline")


class ArticleLimitExceeded(Exception):
    pass

# Bounds concurrent MediaWiki fetches during the planning phase. Not the same
# knob as `workers` (which bounds concurrent OpenRouter translation calls) —
# fetching is cheap and this just keeps us polite to the API.
_FETCH_CONCURRENCY = 8


def _chunk_for_line(spans: list[tuple[Chunk, int, int]], line_number: int | None) -> Chunk | None:
    if line_number is None:
        return None
    for chunk, start, end in spans:
        if start <= line_number <= end:
            return chunk
    return None


async def _validate_assembled(
    text: str,
    target_mw_client: MediaWikiClient,
    article_title: str,
    live_validate: bool,
    live_validate_timeout_s: float,
) -> list[ValidationIssue]:
    """Static checks (validator.py) always run; the live parse-API checks
    (live_validator.py) are an optional enhancement, same treatment as
    verify_wikitext's Wikidata calls elsewhere in this module — a failure
    or timeout is logged and the article proceeds on static checks alone
    rather than blocking the whole run."""
    issues = list(validate_wikitext(text).issues)
    if live_validate:
        try:
            live_result = await asyncio.wait_for(
                validate_wikitext_live(target_mw_client, text, title=article_title),
                timeout=live_validate_timeout_s,
            )
            issues.extend(live_result.issues)
        except Exception as exc:
            logger.warning("Live validation failed for %r: %s", article_title, exc)
    return issues


async def run_assembly_repair(
    chunks: list[Chunk],
    source: ArticleSource,
    config: Config,
    llm_client: LLMEngineClient,
    skill: SkillContent,
    pricing: ModelPricing | None,
    target_mw_client: MediaWikiClient,
    citation_client: httpx.AsyncClient | None,
    stats: RunStats,
    on_retry: RetryCallback | None = None,
) -> tuple[str, list[ValidationIssue], int, dict[str, str]]:
    """Post-processes and validates the assembled article, repairing only
    the chunks implicated by each round's findings, up to
    config.max_assembly_repair_rounds. Mutates chunk.translated_text in
    place for any chunk that gets repaired. This is the whole-article
    counterpart to translator.translate_chunk's per-chunk validate/repair
    loop — kept as a separate loop with its own round budget
    (config.max_assembly_repair_rounds) since these checks (live parse-API
    errors, table span mismatches, etc.) only make sense once every chunk
    is in place.

    Returns (assembled_text, remaining_issues, rounds_used,
    citation_languages_filled) — remaining_issues is empty iff the article
    is valid, whether immediately or after repair; the caller decides what
    "still has issues after the cap" means (needs_human_review)."""
    citation_languages_filled: dict[str, str] = {}

    async def _post_process(text: str) -> str:
        # Deterministic string-level fixes, re-run after every repair round
        # (not just once) since a freshly reassembled `text` hasn't had
        # them applied yet — each fix is written to be a no-op on
        # already-fixed input, so repeating them is safe, just not free.
        nonlocal citation_languages_filled
        if citation_client is not None:
            try:
                citation_result = await asyncio.wait_for(
                    fill_missing_citation_languages(
                        text,
                        citation_client,
                        max_url_fetches=config.max_citation_url_fetches,
                        concurrency=config.citation_fetch_concurrency,
                        fetch_timeout=config.citation_fetch_timeout_s,
                    ),
                    timeout=config.citation_fill_timeout_s,
                )
                text = citation_result.patched_wikitext
                citation_languages_filled = citation_result.filled
                if citation_result.filled:
                    logger.info(
                        "Filled |language= for %d/%d citation(s) in %r",
                        len(citation_result.filled),
                        citation_result.attempted,
                        source.title,
                    )
            except Exception as exc:
                logger.warning("Citation language fill failed for %r: %s", source.title, exc)

        if config.fix_citation_param_names:
            try:
                param_fix_result = fix_citation_param_names(text)
                text = param_fix_result.patched_wikitext
                if param_fix_result.renamed:
                    logger.info(
                        "Renamed %d mistranslated citation parameter name(s) in %r: %s",
                        len(param_fix_result.renamed),
                        source.title,
                        param_fix_result.renamed,
                    )
            except Exception as exc:
                logger.warning("Citation parameter name fix failed for %r: %s", source.title, exc)

        try:
            sfn_fix_result = fix_sfn_param_names(text)
            text = sfn_fix_result.patched_wikitext
            if sfn_fix_result.renamed:
                logger.info(
                    "Renamed %d sfn parameter(s) in %r: %s",
                    len(sfn_fix_result.renamed),
                    source.title,
                    sfn_fix_result.renamed,
                )
        except Exception as exc:
            logger.warning("Sfn parameter name fix failed for %r: %s", source.title, exc)

        if config.dedupe_short_footnotes:
            try:
                dedupe_result = dedupe_short_footnotes(text)
                text = dedupe_result.patched_wikitext
                if dedupe_result.canonicalized:
                    logger.info(
                        "Reconciled %d short-footnote identity group(s) with diverging |ps= in %r",
                        len(dedupe_result.canonicalized),
                        source.title,
                    )
            except Exception as exc:
                logger.warning("Short-footnote dedup failed for %r: %s", source.title, exc)

        return text

    assembled = await _post_process(assemble_chunks(chunks))
    combined_issues = await _validate_assembled(
        assembled, target_mw_client, source.title, config.live_validate, config.live_validate_timeout_s
    )

    rounds_used = 0
    while combined_issues and rounds_used < config.max_assembly_repair_rounds:
        rounds_used += 1
        _, spans = assemble_chunks_with_spans(chunks)

        localized: dict[int, list[str]] = {}
        unlocalized: list[str] = []
        for issue in combined_issues:
            finding = issue.as_finding()
            text_line = f"{finding['severity']}: {finding['explanation']}"
            if finding["snippet"]:
                text_line += f" (near: {finding['snippet']})"
            chunk = _chunk_for_line(spans, issue.line_number)
            if chunk is not None:
                localized.setdefault(id(chunk), []).append(text_line)
            else:
                # Can't be pinned to one chunk (e.g. most live-API findings
                # — a rendered error message doesn't literally appear in
                # the source wikitext to locate). Given to every chunk
                # rather than dropped, since an issue no chunk ever sees
                # can never be fixed within the round cap.
                unlocalized.append(text_line)

        for chunk in chunks:
            errors_for_chunk = localized.get(id(chunk), []) + unlocalized
            if not errors_for_chunk:
                continue
            try:
                stats.repair_attempts += 1
                repair_result = await repair_chunk(
                    llm_client,
                    skill,
                    config.model,
                    config.temperature,
                    chunk.source_lang,
                    config.target_lang,
                    chunk.article_title,
                    chunk.section_title,
                    chunk.translated_text or "",
                    errors_for_chunk,
                    pricing,
                    on_retry=on_retry,
                )
                stats.tokens_in += repair_result.prompt_tokens
                stats.tokens_out += repair_result.completion_tokens
                stats.estimated_cost_usd += repair_result.cost_usd
                stats.translation_time_s += repair_result.latency_s
                chunk.translated_text = repair_result.text
            except Exception as exc:
                logger.warning(
                    "Assembly-level repair failed for %r chunk %d (round %d): %s",
                    source.title, chunk.order, rounds_used, exc,
                )

        assembled = await _post_process(assemble_chunks(chunks))
        combined_issues = await _validate_assembled(
            assembled, target_mw_client, source.title, config.live_validate, config.live_validate_timeout_s
        )

    return assembled, combined_issues, rounds_used, citation_languages_filled


async def _plan_article(
    mw_pool: MediaWikiClientPool,
    item: ArticleInput,
    config: Config,
    fetch_sem: asyncio.Semaphore,
    wikidata_client: httpx.AsyncClient | None,
    verification_cache: VerificationCache | None,
) -> tuple[ArticleSource, list[Chunk], VerifiedFacts] | None:
    effective_lang = item.source_lang or config.source_lang
    async with fetch_sem:
        try:
            mw_client = mw_pool.get(effective_lang)
            source = await load_article_source(mw_client, item, effective_lang)
        except Exception as exc:
            logger.error("Failed to fetch %r (lang=%s): %s", item.title, effective_lang, exc)
            return None

        facts = VerifiedFacts()
        if wikidata_client is not None:
            try:
                target_mw_client = mw_pool.get(config.target_lang)
                # Hard outer deadline, independent of httpx's own per-call
                # timeouts: verification is an optional enhancement (the
                # model still translates fine without it), so it must never
                # be able to stall an entire batch run indefinitely if a
                # network call hangs past its configured timeout for any
                # reason — confirmed to happen in practice.
                facts = await asyncio.wait_for(
                    verify_wikitext(
                        source.title,
                        source.wikitext,
                        effective_lang,
                        config.target_lang,
                        wikidata_client,
                        target_mw_client,
                        verification_cache,
                    ),
                    timeout=config.verification_timeout_s,
                )
            except Exception as exc:
                logger.warning("Link/template verification failed for %r: %s", item.title, exc)

    sections = split_into_sections(source.wikitext)
    chunks = build_chunks(
        source.title,
        sections,
        config.chunk_min_tokens,
        config.chunk_max_tokens,
        source_lang=effective_lang,
    )
    return source, chunks, facts


async def run_pipeline(
    config: Config,
    inputs: list[ArticleInput],
    force: bool = False,
    reporter: ProgressReporter | None = None,
    stats_tracker: StatsTracker | None = None,
) -> StatsTracker:
    stats_tracker = stats_tracker if stats_tracker is not None else StatsTracker()
    stats = stats_tracker.stats

    skill = load_skill(config.skill_path, config.include_skill_references, config.skill_git_ref)
    cache = TranslationCache(config.cache_db_path) if config.cache else None
    verification_cache = VerificationCache(config.verification_db_path) if config.verify_links else None

    mw_pool = MediaWikiClientPool(config.user_agent, config.source_lang, config.source_wiki_api)
    llm_client, effective_model = build_llm_client(config)
    if effective_model != config.model:
        config = config.model_copy(update={"model": effective_model})
    wikidata_client = (
        httpx.AsyncClient(headers={"User-Agent": config.user_agent}, timeout=config.wikidata_timeout_s)
        if config.verify_links
        else None
    )
    # Separate from wikidata_client: this one fetches arbitrary third-party
    # citation URLs, independently toggleable from verify_links.
    citation_client = (
        httpx.AsyncClient(headers={"User-Agent": config.user_agent}) if config.fill_citation_languages else None
    )

    try:
        pricing = await llm_client.get_pricing_for(config.model)

        pending_items: list[ArticleInput] = []
        for item in inputs:
            if not force and article_already_done(config.output_dir, item.title):
                stats.articles_skipped += 1
                logger.info("Skipping %r: output already exists", item.title)
                continue
            pending_items.append(item)

        fetch_sem = asyncio.Semaphore(_FETCH_CONCURRENCY)
        planned = await asyncio.gather(
            *(
                _plan_article(mw_pool, item, config, fetch_sem, wikidata_client, verification_cache)
                for item in pending_items
            )
        )

        article_plans: list[tuple[ArticleSource, list[Chunk], VerifiedFacts]] = []
        for plan in planned:
            if plan is None:
                stats.articles_failed += 1
                continue
            article_plans.append(plan)

        total_chunks = sum(len(chunks) for _, chunks, _ in article_plans)
        estimated_cost = 0.0
        if pricing is not None:
            for _, chunks, _ in article_plans:
                for chunk in chunks:
                    # Heuristic: assume completion length is roughly comparable to
                    # prompt length for a translation task (source + target text
                    # are similar order of magnitude). Actual cost uses real usage.
                    estimated_cost += chunk.token_estimate * (
                        pricing.prompt_price_per_token + pricing.completion_price_per_token
                    )

        if reporter is not None:
            reporter.set_plan(len(article_plans), total_chunks, estimated_cost)

        slot_queue: asyncio.Queue[int] = asyncio.Queue()
        for i in range(config.workers):
            slot_queue.put_nowait(i)

        def on_retry(attempt: int, reason: str, delay: float) -> None:
            # Matches RetryCallback's (attempt, reason, delay) signature from
            # openrouter.py's _backoff — model isn't passed through that call
            # chain, so it's taken from the enclosing config instead.
            stats.retries += 1
            logger.warning("Retry %d for %s: %s (sleeping %.1fs)", attempt, config.model, reason, delay)
            if reporter is not None:
                reporter.on_retry(config.model, attempt, reason, delay)

        async def process_article(source: ArticleSource, chunks: list[Chunk], facts: VerifiedFacts) -> None:
            article_stats = {
                'input_tokens': 0,
                'output_tokens': 0,
                'start_time': time.monotonic(),
                'chunks_done': 0,
                'failed': False,
                'limit_exceeded': False,
            }

            async def process_chunk(chunk: Chunk, facts: VerifiedFacts) -> None:
                slot_id = await slot_queue.get()
                try:
                    if reporter is not None:
                        reporter.on_chunk_start(slot_id, chunk.article_title, chunk.section_title)
                    try:
                        outcome = await translate_chunk(
                            chunk,
                            config,
                            llm_client,
                            skill,
                            cache,
                            pricing,
                            stats,
                            on_retry=on_retry,
                            verified_facts=facts,
                        )
                    except EngineError as exc:
                        chunk.status = ChunkStatus.FAILED
                        logger.error(
                            "Engine call failed for %s chunk %d: %s", chunk.article_title, chunk.order, exc
                        )
                        article_stats['failed'] = True
                        return
                    if not outcome.validation.valid:
                        logger.error(
                            "Validation failed for %s chunk %d after repair attempts: %s",
                            chunk.article_title,
                            chunk.order,
                            "; ".join(format_errors(outcome.validation)),
                        )
                        article_stats['failed'] = True
                        return
                    # Update article stats
                    article_stats['input_tokens'] += outcome.prompt_tokens
                    article_stats['output_tokens'] += outcome.completion_tokens
                    article_stats['chunks_done'] += 1
                    # Check token ratio limit
                    if config.max_token_ratio > 0 and article_stats['input_tokens'] > 0:
                        ratio = article_stats['output_tokens'] / article_stats['input_tokens']
                        if ratio > config.max_token_ratio:
                            logger.error(
                                "Article %r token ratio exceeded: output/input = %.2f > %.2f",
                                source.title, ratio, config.max_token_ratio
                            )
                            article_stats['limit_exceeded'] = True
                            raise ArticleLimitExceeded(f"Token ratio {ratio:.2f} exceeds limit {config.max_token_ratio}")
                    # Check total token limit
                    if config.max_article_tokens > 0:
                        total_tokens = article_stats['input_tokens'] + article_stats['output_tokens']
                        if total_tokens > config.max_article_tokens:
                            logger.error(
                                "Article %r total tokens exceeded: %d > %d",
                                source.title, total_tokens, config.max_article_tokens
                            )
                            article_stats['limit_exceeded'] = True
                            raise ArticleLimitExceeded(f"Total tokens {total_tokens} exceeds limit {config.max_article_tokens}")
                finally:
                    if reporter is not None:
                        reporter.on_chunk_done(slot_id)
                    slot_queue.put_nowait(slot_id)
                    stats_tracker.write(config.stats_path)

            # Execute chunk translation with overall article timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(process_chunk(c, facts) for c in chunks)),
                    timeout=config.article_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Article %r translation timed out after %.1f seconds",
                    source.title, config.article_timeout_s
                )
                # Mark remaining pending chunks as failed
                for c in chunks:
                    if c.status == ChunkStatus.PENDING:
                        c.status = ChunkStatus.FAILED
                article_stats['failed'] = True
            except ArticleLimitExceeded:
                # Already logged, mark remaining chunks as failed
                for c in chunks:
                    if c.status == ChunkStatus.PENDING:
                        c.status = ChunkStatus.FAILED
                article_stats['failed'] = True
            except Exception as exc:
                logger.error("Unexpected error processing article %r: %s", source.title, exc)
                article_stats['failed'] = True
                for c in chunks:
                    if c.status == ChunkStatus.PENDING:
                        c.status = ChunkStatus.FAILED

            # Determine article outcome
            failed_chunks = [c for c in chunks if c.status == ChunkStatus.FAILED]
            if failed_chunks or article_stats['failed']:
                stats.articles_failed += 1
                logger.error(
                    "Article %r failed: %d of %d sections failed",
                    source.title, len(failed_chunks), len(chunks)
                )
                if reporter is not None:
                    reporter.on_article_done()
                return

            # All chunks succeeded, proceed with assembly and post-processing
            target_mw_client = mw_pool.get(config.target_lang)
            assembled, combined_issues, rounds_used, citation_languages_filled = await run_assembly_repair(
                chunks, source, config, llm_client, skill, pricing, target_mw_client, citation_client, stats, on_retry
            )

            if combined_issues:
                stats.articles_needs_human_review += 1
                review_path = record_needs_human_review(config.output_dir, source.title, combined_issues, rounds_used)
                logger.error(
                    "Article %r needs human review after %d assembly-level repair round(s) — see %s: %s",
                    source.title,
                    rounds_used,
                    review_path,
                    "; ".join(f"{i.kind}: {i.message}" for i in combined_issues),
                )
                if reporter is not None:
                    reporter.on_article_done()
                return

            # Add attribution block as HTML comment at the bottom of the file
            attribution = build_attribution_block(source)
            if attribution:
                assembled = f"{assembled}\n\n{attribution}"

            path = save_article(config.output_dir, source.title, assembled)
            stats.articles_completed += 1
            logger.info("Saved %s", path)

            if config.generate_reports:
                report_text = build_article_report(
                    ArticleReportData(
                        source=source,
                        chunks=chunks,
                        facts=facts,
                        citation_languages_filled=citation_languages_filled,
                    ),
                    assembled,
                )
                report_path = save_report(config.output_dir, source.title, report_text)
                logger.info("Saved %s", report_path)

            if reporter is not None:
                reporter.on_article_done()

        await asyncio.gather(
            *(process_article(source, chunks, facts) for source, chunks, facts in article_plans)
        )

    finally:
        await mw_pool.aclose()
        await llm_client.aclose()
        if wikidata_client is not None:
            await wikidata_client.aclose()
        if citation_client is not None:
            await citation_client.aclose()
        if cache is not None:
            cache.close()
        if verification_cache is not None:
            verification_cache.close()
        stats_tracker.write(config.stats_path)

    return stats_tracker
