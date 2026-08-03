"""Persistent history of discovered MT5 pattern overlays."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from atis.config import PROJECT_ROOT
from atis.shared.mt5_pattern_overlay.models import PatternOverlay


def history_path() -> Path:
    path = PROJECT_ROOT / "logs" / "live" / "mt5_overlay" / "patterns_history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_history(overlays: Iterable[PatternOverlay], *, event: str = "detect") -> None:
    path = history_path()
    ts = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as fh:
        for ov in overlays:
            rec = {
                "event": event,
                "logged_at": ts,
                **ov.to_dict(),
            }
            # Drop bulky object geometry from long-term history (keep anchors + meta)
            rec.pop("objects", None)
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def read_history(
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    path = history_path()
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if symbol and rec.get("symbol") != symbol:
                continue
            if timeframe and str(rec.get("timeframe", "")).upper() != str(timeframe).upper():
                continue
            rows.append(rec)
    return rows[-limit:]
