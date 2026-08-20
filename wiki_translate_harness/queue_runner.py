"""Queue mode: drain articles from the shared wiki-translate-queue repo
(github.com/arianit/wiki-translate-queue) instead of a manually-supplied
--title/--titles/--category. Delegates the actual translation to the same
run_pipeline() the CLI's normal modes use; this module only owns picking
articles off the shared queue and reporting results back to it.

See that repo's README for the claim/work/finish protocol this implements
via its queue_lib.py (dynamically imported from the cloned repo — this
package doesn't vendor a copy, to avoid two sources of truth).
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from wiki_translate_harness.config import Config
from wiki_translate_harness.sources import ArticleInput, parse_source_ref
from wiki_translate_harness.statistics import StatsTracker

logger = logging.getLogger("wiki_translate_harness.queue_runner")

DEFAULT_QUEUE_REPO_DIR = Path("~/code/wiki-translate-queue").expanduser()


def load_queue_lib(repo_dir: Path):
    spec = importlib.util.spec_from_file_location("queue_lib", repo_dir / "queue_lib.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def run_queue_mode(
    config: Config,
    queue_repo_dir: Path = DEFAULT_QUEUE_REPO_DIR,
    max_articles: int = 10,
    stale_hours: float = 3.0,
) -> StatsTracker:
    from wiki_translate_harness.pipeline import run_pipeline  # local import: mirrors cli.py

    queue_lib = load_queue_lib(queue_repo_dir)
    queue_lib.sync_to_remote(queue_repo_dir)

    stats_tracker = StatsTracker()
    stats = stats_tracker.stats
    processed = 0

    while processed < max_articles:
        claimed = queue_lib.claim_next_pending(queue_repo_dir, stale_hours=stale_hours)
        if claimed is None:
            logger.info("Shared queue has no pending articles.")
            break
        line_no, url = claimed
        lang, title = parse_source_ref(url)
        logger.info("Claimed queue line %d: %s", line_no, url)

        article_stats = StatsTracker()
        try:
            await run_pipeline(
                config,
                [ArticleInput(title=title, source_lang=lang)],
                force=True,  # the queue is the source of truth for done/failed, not output_dir presence
                stats_tracker=article_stats,
            )
        except Exception as exc:  # noqa: BLE001 - must still record FAILED and move on to the next article
            logger.exception("Queue article %s crashed", url)
            try:
                queue_lib.finish_line(queue_repo_dir, line_no, url, "FAILED", reason=str(exc)[:200])
            except queue_lib.QueueSyncError as sync_exc:
                logger.error("Could not push FAILED result for %s: %s", url, sync_exc)
            stats.articles_failed += 1
            processed += 1
            continue

        if article_stats.stats.articles_completed >= 1:
            status, reason = "DONE", None
            stats.articles_completed += 1
        else:
            status, reason = "FAILED", "translation did not complete (see run.log for this article)"
            stats.articles_failed += 1
        try:
            queue_lib.finish_line(queue_repo_dir, line_no, url, status, reason=reason)
        except queue_lib.QueueSyncError as sync_exc:
            logger.error("Could not push %s result for %s: %s", status, url, sync_exc)
        processed += 1

    return stats_tracker
