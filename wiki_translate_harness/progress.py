"""Rich live progress display: current article/section, worker status, ETA, cost."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table

from wiki_translate_harness.models import RunStats


@dataclass
class WorkerSlot:
    slot_id: int
    article: str = ""
    section: str = ""
    action: str = "idle"  # idle | translating | repairing


@dataclass
class ProgressState:
    total_articles: int = 0
    articles_done: int = 0
    total_chunks: int = 0
    chunks_done: int = 0
    estimated_total_cost_usd: float = 0.0
    last_retry: str = ""
    _start_time: float = field(default_factory=time.monotonic)

    @property
    def eta_seconds(self) -> float | None:
        if self.chunks_done == 0 or self.total_chunks == 0:
            return None
        elapsed = time.monotonic() - self._start_time
        rate = self.chunks_done / elapsed
        remaining = self.total_chunks - self.chunks_done
        return remaining / rate if rate > 0 else None


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class ProgressReporter:
    def __init__(
        self,
        stats: RunStats,
        workers: int,
        console: Console | None = None,
        on_event: Callable[[str], None] | None = None,
    ):
        self.stats = stats
        self.state = ProgressState()
        self.slots = [WorkerSlot(slot_id=i) for i in range(workers)]
        self.console = console or Console()
        self._live: Live | None = None
        # Optional one-line-per-event text sink, independent of the Rich
        # Live table below — e.g. queue_runner.py wires this to logger.info
        # so non-interactive runs (which never enter/render Live) still get
        # human-readable per-chunk/per-article progress in logs/run.log.
        self.on_event = on_event

    def __enter__(self) -> "ProgressReporter":
        self._live = Live(self._render(), console=self.console, refresh_per_second=4, transient=False)
        self._live.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._live is not None:
            self._live.update(self._render())
            self._live.__exit__(*exc)

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    # --- event hooks called by pipeline.py ---

    def set_plan(self, total_articles: int, total_chunks: int, estimated_cost_usd: float) -> None:
        self.state.total_articles = total_articles
        self.state.total_chunks = total_chunks
        self.state.estimated_total_cost_usd = estimated_cost_usd
        self.refresh()

    def on_chunk_start(self, slot_id: int, article: str, section: str) -> None:
        if self.on_event:
            self.on_event(f"worker {slot_id}: start '{article}' — {section}")
        slot = self.slots[slot_id]
        slot.article, slot.section, slot.action = article, section, "translating"
        self.refresh()

    def on_repair_start(self, slot_id: int) -> None:
        self.slots[slot_id].action = "repairing"
        self.refresh()

    def on_chunk_done(self, slot_id: int) -> None:
        slot = self.slots[slot_id]
        if self.on_event:
            self.on_event(
                f"worker {slot_id}: done '{slot.article}' — {slot.section} | "
                f"cumulative: {self.stats.sections_translated} sections, "
                f"{self.stats.cache_hits} cache hits, ${self.stats.estimated_cost_usd:.4f}"
            )
        slot.action = "idle"
        slot.article, slot.section = "", ""
        self.state.chunks_done += 1
        self.refresh()

    def on_article_done(self, article: str, outcome: str) -> None:
        if self.on_event:
            self.on_event(
                f"article '{article}' {outcome} ({self.state.articles_done + 1}/{self.state.total_articles})"
            )
        self.state.articles_done += 1
        self.refresh()

    def on_retry(self, model: str, attempt: int, reason: str, delay: float) -> None:
        self.state.last_retry = f"retry {attempt} ({reason}) in {delay:.1f}s [{model}]"
        if self.on_event:
            self.on_event(self.state.last_retry)
        self.refresh()

    # --- rendering ---

    def _render(self) -> Group:
        header = Table.grid(padding=(0, 2))
        header.add_column()
        header.add_column()
        header.add_column()
        header.add_column()
        header.add_row(
            f"Articles: {self.state.articles_done}/{self.state.total_articles}",
            f"Sections: {self.state.chunks_done}/{self.state.total_chunks}",
            f"ETA: {_fmt_duration(self.state.eta_seconds)}",
            f"Cost: ${self.stats.estimated_cost_usd:.4f} / ~${self.state.estimated_total_cost_usd:.4f} est.",
        )

        worker_table = Table(show_header=True, header_style="bold")
        worker_table.add_column("Worker")
        worker_table.add_column("Article")
        worker_table.add_column("Section")
        worker_table.add_column("Status")
        for slot in self.slots:
            worker_table.add_row(
                str(slot.slot_id),
                slot.article or "-",
                slot.section or "-",
                slot.action,
            )

        footer = self.state.last_retry or ""
        parts = [header, worker_table]
        if footer:
            parts.append(footer)
        return Group(*parts)
