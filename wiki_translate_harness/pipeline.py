"""Concurrent multi-article orchestration: fetch -> chunk -> translate -> assemble -> save.

Ties together mediawiki.py, parser.py, cache.py, translator.py, and
output.py. Contains no translation logic of its own.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from wiki_translate_harness.cache import TranslationCache, VerificationCache
from wiki_translate_harness.citation_language import (
    dedupe_short_footnotes,
    fill_missing_citation_languages,
    fix_citation_param_names,
    fix_sfn_param_names,
)
from wiki_translate_harness.report import build_attribution_block
from wiki_translate_harness.mediawiki import MediaWikiClientPool
from wiki_translate_harness.models import ArticleSource, Chunk, ChunkStatus, Config
from wiki_translate_harness.openrouter import OpenRouterClient, OpenRouterError
from wiki_translate_harness.output import article_already_done, assemble_chunks, save_article
from wiki_translate_harness.parser import build_chunks, split_into_sections
from wiki_translate_harness.progress import ProgressReporter
from wiki_translate_harness.report import ArticleReportData, build_article_report, save_report
from wiki_translate_harness.skill_loader import load_skill
from wiki_translate_harness.sources import ArticleInput, load_article_source
from wiki_translate_harness.statistics import StatsTracker
from wiki_translate_harness.translator import translate_chunk
from wiki_translate_harness.validator import format_errors, validate_wikitext
from wiki_translate_harness.verification import VerifiedFacts, verify_wikitext

logger = logging.getLogger("wiki_translate_harness.pipeline")

# Bounds concurrent MediaWiki fetches during the planning phase. Not the same
# knob as `workers` (which bounds concurrent OpenRouter translation calls) —
# fetching is cheap and this just keeps us polite to the API.
_FETCH_CONCURRENCY = 8


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
    or_client = OpenRouterClient(
        api_key=config.openrouter_api_key,
        base_url=config.openrouter_base_url,
        user_agent=config.user_agent,
        timeout=config.request_timeout_s,
        max_retries=config.max_retries,
    )
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
        pricing = await or_client.get_pricing_for(config.model)

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

        async def process_chunk(chunk: Chunk, facts: VerifiedFacts) -> None:
            slot_id = await slot_queue.get()
            try:
                if reporter is not None:
                    reporter.on_chunk_start(slot_id, chunk.article_title, chunk.section_title)
                try:
                    outcome = await translate_chunk(
                        chunk,
                        config,
                        or_client,
                        skill,
                        cache,
                        pricing,
                        stats,
                        on_retry=on_retry,
                        verified_facts=facts,
                    )
                except OpenRouterError as exc:
                    chunk.status = ChunkStatus.FAILED
                    logger.error(
                        "OpenRouter call failed for %s chunk %d: %s", chunk.article_title, chunk.order, exc
                    )
                    return
                if not outcome.validation.valid:
                    logger.error(
                        "Validation failed for %s chunk %d after repair attempts: %s",
                        chunk.article_title,
                        chunk.order,
                        "; ".join(format_errors(outcome.validation)),
                    )
            finally:
                if reporter is not None:
                    reporter.on_chunk_done(slot_id)
                slot_queue.put_nowait(slot_id)
                stats_tracker.write(config.stats_path)

        async def process_article(source: ArticleSource, chunks: list[Chunk], facts: VerifiedFacts) -> None:
            await asyncio.gather(*(process_chunk(c, facts) for c in chunks))

            failed = [c for c in chunks if c.status == ChunkStatus.FAILED]
            if failed:
                stats.articles_failed += 1
                logger.error(
                    "Article %r failed: %d of %d sections failed validation",
                    source.title,
                    len(failed),
                    len(chunks),
                )
                if reporter is not None:
                    reporter.on_article_done()
                return

            assembled = assemble_chunks(chunks)

            citation_languages_filled: dict[str, str] = {}
            if citation_client is not None:
                try:
                    # Same hard-deadline reasoning as verification above:
                    # this is an optional enhancement over citations the
                    # model already produced, and must never be able to
                    # stall the whole run if a fetch hangs indefinitely.
                    citation_result = await asyncio.wait_for(
                        fill_missing_citation_languages(
                            assembled,
                            citation_client,
                            max_url_fetches=config.max_citation_url_fetches,
                            concurrency=config.citation_fetch_concurrency,
                            fetch_timeout=config.citation_fetch_timeout_s,
                        ),
                        timeout=config.citation_fill_timeout_s,
                    )
                    assembled = citation_result.patched_wikitext
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
                # Deterministic backstop for the model mistranslating CS1
                # citation parameter *names* into the target language
                # (confirmed in practice: |date=/|title=/|website=/
                # |access-date= all renamed to Albanian on one citation in
                # a citation-dense article, silently dropping those fields
                # on render). Synchronous, no I/O — no timeout needed.
                try:
                    param_fix_result = fix_citation_param_names(assembled)
                    assembled = param_fix_result.patched_wikitext
                    if param_fix_result.renamed:
                        logger.info(
                            "Renamed %d mistranslated citation parameter name(s) in %r: %s",
                            len(param_fix_result.renamed),
                            source.title,
                            param_fix_result.renamed,
                        )
                except Exception as exc:
                    logger.warning("Citation parameter name fix failed for %r: %s", source.title, exc)

            # Fix sfn/harvnb parameter names mistranslated into Albanian
            # (|f= -> |p=, |ff= -> |pp=, positional "f. 161" -> |p=161).
            try:
                sfn_fix_result = fix_sfn_param_names(assembled)
                assembled = sfn_fix_result.patched_wikitext
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
                # The same source citation split across independently
                # -translated chunks can come back with its |ps= quote
                # paraphrased slightly differently each time, which breaks
                # {{sfn}}/{{harvnb}}'s shared auto-generated anchor
                # (MediaWiki requires byte-identical content across all
                # uses of that anchor). Synchronous, no I/O.
                try:
                    dedupe_result = dedupe_short_footnotes(assembled)
                    assembled = dedupe_result.patched_wikitext
                    if dedupe_result.canonicalized:
                        logger.info(
                            "Reconciled %d short-footnote identity group(s) with diverging |ps= in %r",
                            len(dedupe_result.canonicalized),
                            source.title,
                        )
                except Exception as exc:
                    logger.warning("Short-footnote dedup failed for %r: %s", source.title, exc)

            final_check = validate_wikitext(assembled)
            if not final_check.valid:
                stats.articles_failed += 1
                logger.error(
                    "Article %r failed final assembly validation: %s",
                    source.title,
                    "; ".join(format_errors(final_check)),
                )
                if reporter is not None:
                    reporter.on_article_done()
                return

            # Add attribution block as HTML comment at the top of the file
            attribution = build_attribution_block(source)
            if attribution:
                assembled = f"<!--\n{attribution}\n-->\n\n{assembled}"
            
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
        await or_client.aclose()
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
