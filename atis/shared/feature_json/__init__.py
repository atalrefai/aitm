"""JSONL export/import of bar+pattern matrices for deep pattern discovery.

Each timeframe is stored as a standalone file:

    data/features/{symbol}/{timeframe}/discovery_bars.jsonl

One JSON object per bar (full history, no truncation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from atis.config import get_path

CORE_COLS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
    "symbol",
    "timeframe",
    "pat_bias",
    "pat_strength",
    "chart_pattern_score",
    "structure_hh_hl",
)


def discovery_bars_path(symbol: str, timeframe: str) -> Path:
    return get_path("data_features") / symbol / timeframe / "discovery_bars.jsonl"


def _pattern_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in CORE_COLS:
            cols.append(c)
            continue
        if c.startswith(("pat_", "cmp_", "disc_")) and c not in {"pat_bias", "pat_strength"}:
            cols.append(c)
    # Preserve original column order as much as possible
    ordered = [c for c in df.columns if c in set(cols)]
    return ordered


def _json_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def export_discovery_bars_jsonl(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> Path:
    """Write the full bar/pattern matrix for one timeframe as JSONL."""
    path = discovery_bars_path(symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = _pattern_columns(df)
    frame = df.loc[:, cols].copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = frame["timestamp"].astype(str)
    if "symbol" not in frame.columns:
        frame["symbol"] = symbol
    if "timeframe" not in frame.columns:
        frame["timeframe"] = timeframe

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in frame.itertuples(index=False, name=None):
            payload = {cols[i]: _json_cell(row[i]) for i in range(len(cols))}
            # ensure identity fields exist even if not in source cols order
            payload["symbol"] = symbol
            payload["timeframe"] = timeframe
            fh.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")
    tmp.replace(path)
    return path


def load_discovery_bars_jsonl(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Load the full JSONL discovery file for one timeframe (no row skip)."""
    path = discovery_bars_path(symbol, timeframe)
    if not path.exists():
        return None
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def ensure_discovery_bars_jsonl(
    symbol: str,
    timeframe: str,
    *,
    source_df: pd.DataFrame | None = None,
    force: bool = False,
) -> tuple[Path, pd.DataFrame]:
    """
    Ensure a full JSONL exists for the timeframe, then load it entirely.

    If ``source_df`` is provided (typically from parquet), it is exported first
    when the JSONL is missing or ``force`` is True.
    """
    path = discovery_bars_path(symbol, timeframe)
    if force or not path.exists():
        if source_df is None:
            parquet = get_path("data_features") / symbol / timeframe / "features.parquet"
            if not parquet.exists():
                raise FileNotFoundError(f"No features for {symbol}/{timeframe}")
            source_df = pd.read_parquet(parquet)
        export_discovery_bars_jsonl(source_df, symbol, timeframe)

    df = load_discovery_bars_jsonl(symbol, timeframe)
    if df is None:
        raise FileNotFoundError(f"Failed to load discovery JSONL for {symbol}/{timeframe}")
    return path, df
