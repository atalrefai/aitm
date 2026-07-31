"""Structured JSON logging utilities (Principle 1.5)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog

from atis.config import get_path, load_engine_config


def _configure_once() -> None:
    if getattr(_configure_once, "_done", False):
        return

    cfg = load_engine_config().get("logging", {})
    level_name = str(cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    json_logs = bool(cfg.get("json_logs", True))

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_logs:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    try:
        log_dir = get_path("logs")
    except Exception:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "atis.jsonl", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _configure_once._done = True  # type: ignore[attr-defined]


def get_logger(name: str = "atis", **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Return a structured logger bound with optional context."""
    _configure_once()
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger
