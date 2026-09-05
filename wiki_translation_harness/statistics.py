"""Run statistics accumulation and atomic stats.json writes."""

from __future__ import annotations

import time
from pathlib import Path

import orjson

from wiki_translation_harness.models import RunStats


class StatsTracker:
    def __init__(self) -> None:
        self.stats = RunStats()

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.stats.started_at

    def write(self, path: Path) -> None:
        self.stats.updated_at = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = orjson.dumps(self.stats.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_bytes(data)
        tmp_path.replace(path)
