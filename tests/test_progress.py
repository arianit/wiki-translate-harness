"""ProgressReporter's on_event hook — the mechanism queue_runner.py uses to
get per-chunk/per-article text lines into logs/run.log for non-interactive
runs, independent of the Rich Live table (which is never entered here, same
as queue_runner.py's usage -- confirms refresh() no-ops without a Live)."""

from wiki_translation_harness.models import RunStats
from wiki_translation_harness.progress import ProgressReporter


def _reporter(on_event=None) -> ProgressReporter:
    return ProgressReporter(RunStats(), workers=2, on_event=on_event)


def test_on_chunk_done_emits_before_clearing_slot():
    events: list[str] = []
    reporter = _reporter(on_event=events.append)
    reporter.on_chunk_start(0, "Mars", "Geography")
    events.clear()
    reporter.on_chunk_done(0)
    assert len(events) == 1
    # slot.article/section must still be in the message, even though
    # on_chunk_done clears them on the slot right after
    assert "Mars" in events[0]
    assert "Geography" in events[0]
    assert reporter.slots[0].article == ""  # confirms the clear still happens


def test_on_chunk_done_includes_cumulative_stats():
    stats = RunStats(sections_translated=3, cache_hits=1, estimated_cost_usd=0.05)
    events: list[str] = []
    reporter = ProgressReporter(stats, workers=1, on_event=events.append)
    reporter.on_chunk_start(0, "Mars", "Geography")
    reporter.on_chunk_done(0)
    assert "3 sections" in events[-1]
    assert "1 cache hits" in events[-1]
    assert "0.0500" in events[-1]


def test_on_article_done_emits_title_and_outcome():
    events: list[str] = []
    reporter = _reporter(on_event=events.append)
    reporter.on_article_done("Neptune", "failed")
    assert "Neptune" in events[0]
    assert "failed" in events[0]


