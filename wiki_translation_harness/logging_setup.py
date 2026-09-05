"""Wires logs/run.log (all activity) and logs/errors.log (errors only)."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: Path, console_level: int = logging.WARNING) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("wiki_translation_harness")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    run_handler = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    run_handler.setLevel(logging.INFO)
    run_handler.setFormatter(fmt)
    logger.addHandler(run_handler)

    error_handler = logging.FileHandler(log_dir / "errors.log", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(fmt)
    logger.addHandler(error_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger
