"""Per-timeframe discovery checkpoints for cancel/resume safety."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atis.config import PROJECT_ROOT, get_path


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def checkpoint_dir(symbol: str, timeframe: str) -> Path:
    try:
        root = get_path("data_patterns")
    except KeyError:
        root = PROJECT_ROOT / "data" / "patterns"
    path = root / symbol / timeframe / "_checkpoint"
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_path(symbol: str, timeframe: str) -> Path:
    return checkpoint_dir(symbol, timeframe) / "discovery_state.json"


def load_checkpoint(symbol: str, timeframe: str) -> dict[str, Any] | None:
    path = checkpoint_path(symbol, timeframe)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_checkpoint(symbol: str, timeframe: str, state: dict[str, Any]) -> Path:
    path = checkpoint_path(symbol, timeframe)
    payload = {
        **state,
        "symbol": symbol,
        "timeframe": timeframe,
        "updated_at": _utc(),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def clear_checkpoint(symbol: str, timeframe: str) -> None:
    path = checkpoint_path(symbol, timeframe)
    if path.exists():
        path.unlink()


def stage_done(state: dict[str, Any] | None, stage: str) -> bool:
    if not state:
        return False
    return stage in (state.get("completed_stages") or [])


def mark_stage(state: dict[str, Any], stage: str, **extra: Any) -> dict[str, Any]:
    done = list(state.get("completed_stages") or [])
    if stage not in done:
        done.append(stage)
    state["completed_stages"] = done
    state["last_stage"] = stage
    state.update(extra)
    return state
