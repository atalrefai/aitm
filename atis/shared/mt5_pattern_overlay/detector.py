"""Extract live pattern overlays from a featured OHLC frame."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from atis.shared.feature_engine.patterns import (
    PATTERN_CATALOG,
    bearish_keys,
    bullish_keys,
    pattern_category_map,
    pattern_labels,
)
from atis.shared.mt5_pattern_overlay.geometry import build_geometry, _ts_unix
from atis.shared.mt5_pattern_overlay.models import (
    STRUCTURE_KEYS,
    PatternOverlay,
    PatternStatus,
    TradeLink,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pattern_id(symbol: str, timeframe: str, key: str, bar_time: int) -> str:
    raw = f"{symbol}|{timeframe}|{key}|{bar_time}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{key}_{timeframe}_{digest}"


def _confidence(
    row: pd.Series,
    key: str,
    kb_stats: dict[str, dict[str, Any]] | None,
) -> float:
    """Blend pattern strength with optional KB confidence."""
    strength = float(row["pat_strength"]) if "pat_strength" in row.index and pd.notna(row["pat_strength"]) else 0.0
    chart = (
        float(row["chart_pattern_score"])
        if "chart_pattern_score" in row.index and pd.notna(row["chart_pattern_score"])
        else 0.0
    )
    base = 0.45 + 0.12 * min(abs(strength), 3.0) + 0.05 * min(abs(chart), 5.0)
    if kb_stats and key in kb_stats:
        kb_c = kb_stats[key].get("confidence")
        if kb_c is not None:
            try:
                base = 0.5 * base + 0.5 * float(kb_c)
            except (TypeError, ValueError):
                pass
    return float(max(0.05, min(0.99, base)))


def _expected_direction(bias: str) -> str:
    if bias == "bullish":
        return "buy"
    if bias == "bearish":
        return "sell"
    return "neutral"


def _pattern_columns(df: pd.DataFrame) -> list[str]:
    labels = pattern_labels()
    skip = {"pat_bias", "pat_strength", "chart_pattern_score", "structure_hh_hl", "trendline_slope"}
    cols: list[str] = []
    for c in df.columns:
        if c in skip:
            continue
        if c in labels or c.startswith(("pat_", "cmp_", "disc_")):
            cols.append(c)
    return cols


def extract_active_overlays(
    featured: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    lookback_bars: int = 8,
    max_patterns: int = 40,
    kb_stats: dict[str, dict[str, Any]] | None = None,
    previous: dict[str, PatternOverlay] | None = None,
) -> list[PatternOverlay]:
    """
    Scan recent bars for pattern hits and build drawable overlays.

    Status rules:
    - hit on last closed bar → confirmed
    - hit on earlier bar still in lookback → confirmed (historical on chart)
    - previously active, opposite structural break now → invalidated
    - trade link preserved from ``previous`` when same id
    """
    if featured is None or featured.empty:
        return []
    df = featured.copy()
    if "timestamp" not in df.columns:
        raise ValueError("featured frame requires timestamp column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.reset_index(drop=True)

    labels = pattern_labels()
    cats = pattern_category_map()
    bulls = bullish_keys()
    bears = bearish_keys()
    cols = _pattern_columns(df)
    n = len(df)
    start = max(0, n - max(1, lookback_bars))
    last_i = n - 1

    # Invalidation signals on latest bar
    last = df.iloc[last_i]
    invalidate_bull = any(
        c in df.columns and float(last.get(c, 0) or 0) == 1.0
        for c in ("pat_bos_down", "pat_choch_bear", "pat_breakout_down")
    )
    invalidate_bear = any(
        c in df.columns and float(last.get(c, 0) or 0) == 1.0
        for c in ("pat_bos_up", "pat_choch_bull", "pat_breakout_up")
    )

    found: dict[str, PatternOverlay] = {}
    for i in range(start, n):
        row = df.iloc[i]
        bar_time = _ts_unix(row["timestamp"])
        for col in cols:
            try:
                if float(row.get(col, 0) or 0) != 1.0:
                    continue
            except (TypeError, ValueError):
                continue
            meta = PATTERN_CATALOG.get(col, {})
            name = labels.get(col, meta.get("name", col))
            cat = cats.get(col, meta.get("category", "candle"))
            if col in STRUCTURE_KEYS:
                cat = "structure" if cat == "chart" else cat
            if col in bulls:
                bias = "bullish"
            elif col in bears:
                bias = "bearish"
            else:
                bias = str(meta.get("bias", "neutral"))
            pid = _pattern_id(symbol, timeframe, col, bar_time)
            status = PatternStatus.CONFIRMED if i == last_i else PatternStatus.CONFIRMED
            # Forming heuristic: pattern on last bar but weak strength
            strength = (
                float(row["pat_strength"])
                if "pat_strength" in row.index and pd.notna(row["pat_strength"])
                else 0.0
            )
            if i == last_i and strength < 0.35 and cat == "candle":
                status = PatternStatus.FORMING

            # Invalidate opposing patterns that are older than the break
            if previous and pid in previous:
                prev = previous[pid]
                if prev.status == PatternStatus.LINKED:
                    status = PatternStatus.LINKED
                if prev.trade.ticket or prev.trade.side:
                    status = PatternStatus.LINKED

            if bias == "bullish" and invalidate_bull and i < last_i:
                status = PatternStatus.INVALIDATED
            if bias == "bearish" and invalidate_bear and i < last_i:
                status = PatternStatus.INVALIDATED

            ov = PatternOverlay(
                id=pid,
                key=col,
                name=str(name),
                category=str(cat),
                bias=bias,
                status=status,
                timeframe=str(timeframe).upper(),
                symbol=symbol,
                detected_at=_utc_now_iso() if i == last_i else str(row["timestamp"]),
                bar_time=bar_time,
                bar_index=i,
                confidence=_confidence(row, col, kb_stats),
                strength=strength,
                conditions=str(meta.get("conditions", "")),
                expected_direction=_expected_direction(bias),
                trade=(previous.get(pid).trade if previous and pid in previous else TradeLink()),
                meta={"bar_iso": str(row["timestamp"]), "close": float(row["close"])},
            )
            if previous and pid in previous and previous[pid].status == PatternStatus.LINKED:
                ov.status = PatternStatus.LINKED
                ov.trade = previous[pid].trade
            ov.objects = build_geometry(ov, df)
            found[pid] = ov

    # Carry forward linked patterns that fell out of lookback (still show until cleared)
    if previous:
        for pid, prev in previous.items():
            if pid in found:
                continue
            if prev.status == PatternStatus.LINKED:
                found[pid] = prev

    # Prefer latest / strongest
    overlays = sorted(
        found.values(),
        key=lambda o: (o.bar_time, o.confidence, o.strength),
        reverse=True,
    )
    return overlays[: max_patterns]


def top_signal_patterns(
    overlays: list[PatternOverlay],
    *,
    side: str,
    limit: int = 5,
) -> list[PatternOverlay]:
    """Pick patterns that explain a buy/sell decision."""
    want = "bullish" if side == "buy" else "bearish"
    ranked = [
        o
        for o in overlays
        if o.status in {PatternStatus.CONFIRMED, PatternStatus.LINKED, PatternStatus.FORMING}
        and (o.bias == want or o.expected_direction == side)
    ]
    ranked.sort(key=lambda o: (o.confidence, o.strength, o.bar_time), reverse=True)
    if ranked:
        return ranked[:limit]
    # Fallback: any non-invalidated recent patterns
    alts = [o for o in overlays if o.status != PatternStatus.INVALIDATED]
    alts.sort(key=lambda o: (o.confidence, o.bar_time), reverse=True)
    return alts[:limit]
