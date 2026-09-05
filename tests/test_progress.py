"""ProgressReporter's on_event hook — the mechanism queue_runner.py uses to
get per-chunk/per-article text lines into logs/run.log for non-interactive
runs, independent of the Rich Live table (which is never entered here, same
as queue_runner.py's usage -- confirms refresh() no-ops without a Live)."""

from wiki_translate_harness.models import RunStats
from wiki_translate_harness.progress import ProgressReporter


def _reporter(on_event=None) -> ProgressReporter:
    return ProgressReporter(RunStats(), workers=2, on_event=on_event)


def test_on_event_none_by_default_no_crash():
    reporter = _reporter()
    reporter.on_chunk_start(0, "Mars", "Geography")
    reporter.on_chunk_done(0)
    reporter.on_article_done("Mars", "completed")
    reporter.on_retry("test-model", 1, "rate limit", 2.5)


def test_on_chunk_start_emits_article_and_section():
    events: list[str] = []
    reporter = _reporter(on_event=events.append)
    reporter.on_chunk_start(0, "Mars", "Geography")
    assert len(events) == 1
    assert "Mars" in events[0]
    assert "Geography" in events[0]


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


def test_on_article_done_still_increments_articles_done():
    reporter = _reporter()
    reporter.on_article_done("Neptune", "completed")
    assert reporter.state.articles_done == 1


def test_on_retry_emits_same_text_as_last_retry_state():
    events: list[str] = []
    reporter = _reporter(on_event=events.append)
    reporter.on_retry("test-model", 2, "rate limit", 1.5)
    assert events[0] == reporter.state.last_retry


def test_refresh_is_noop_without_live_display():
    # queue_runner.py constructs a ProgressReporter but never enters it as a
    # context manager -- refresh() must not try to render/raise.
    reporter = _reporter()
    reporter.on_chunk_start(0, "Mars", "Geography")  # calls refresh() internally
    assert reporter._live is None
