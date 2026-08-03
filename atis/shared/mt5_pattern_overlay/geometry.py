"""Build MT5 drawing primitives for detected patterns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from atis.shared.mt5_pattern_overlay.models import DrawObject, PatternOverlay, PatternStatus


def _ts_unix(ts: Any) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.timestamp())


def _obj_name(pattern_id: str, kind: str, idx: int = 0) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in pattern_id)[:48]
    return f"ATIS_{safe}_{kind}{idx}"


def _recent_swings(
    df: pd.DataFrame,
    end_i: int,
    lookback: int = 40,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Approximate swing highs/lows in a lookback window (causal, local extrema)."""
    start = max(0, end_i - lookback)
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    for i in range(start + 2, end_i):
        if high[i] >= high[i - 1] and high[i] >= high[i - 2] and high[i] >= high[min(i + 1, end_i)]:
            if i + 1 <= end_i and high[i] >= high[min(i + 1, end_i)]:
                highs.append((i, float(high[i])))
        if low[i] <= low[i - 1] and low[i] <= low[i - 2] and low[i] <= low[min(i + 1, end_i)]:
            if i + 1 <= end_i and low[i] <= low[min(i + 1, end_i)]:
                lows.append((i, float(low[i])))
    return highs, lows


def build_geometry(overlay: PatternOverlay, df: pd.DataFrame) -> list[DrawObject]:
    """Create drawing objects for a pattern overlay using OHLC context."""
    i = int(overlay.bar_index)
    if i < 0 or i >= len(df):
        return []
    color = overlay.color()
    tip = overlay.tooltip_text()
    bar_t = _ts_unix(df["timestamp"].iloc[i])
    hi = float(df["high"].iloc[i])
    lo = float(df["low"].iloc[i])
    close = float(df["close"].iloc[i])
    rng = max(hi - lo, abs(close) * 1e-4, 1e-6)
    pad = rng * 0.15
    objs: list[DrawObject] = []

    # Direction arrow
    bullish = overlay.bias == "bullish" or overlay.expected_direction == "buy"
    bearish = overlay.bias == "bearish" or overlay.expected_direction == "sell"
    arrow_code = 233 if bullish else (234 if bearish else 159)
    arrow_price = lo - pad if bullish else (hi + pad if bearish else hi + pad)
    objs.append(
        DrawObject(
            type="arrow",
            name=_obj_name(overlay.id, "arr"),
            color=color,
            time=bar_t,
            price=arrow_price,
            arrow_code=arrow_code,
            width=2 if overlay.status != PatternStatus.FORMING else 1,
            tooltip=tip,
        )
    )

    # Label
    label_price = arrow_price - pad if bullish else arrow_price + pad
    objs.append(
        DrawObject(
            type="text",
            name=_obj_name(overlay.id, "lbl"),
            color=color,
            time=bar_t,
            price=label_price,
            text=overlay.label_text(),
            fontsize=8,
            tooltip=tip,
        )
    )

    key = overlay.key
    highs, lows = _recent_swings(df, i)

    # Family-specific geometry
    if key in {"pat_double_top", "pat_triple_top", "pat_head_shoulders", "pat_equal_highs"}:
        pts = highs[-3:] if "triple" in key or "head" in key else highs[-2:]
        for n, (idx, px) in enumerate(pts):
            t = _ts_unix(df["timestamp"].iloc[idx])
            objs.append(
                DrawObject(
                    type="arrow",
                    name=_obj_name(overlay.id, "ah", n),
                    color=color,
                    time=t,
                    price=px + pad * 0.5,
                    arrow_code=251,
                    width=1,
                    tooltip=f"Pivot high #{n + 1}",
                )
            )
            overlay.anchors.append({"role": f"high_{n + 1}", "time": t, "price": px})
        if len(pts) >= 2:
            t1 = _ts_unix(df["timestamp"].iloc[pts[0][0]])
            t2 = _ts_unix(df["timestamp"].iloc[pts[-1][0]])
            neck = float(min(df["low"].iloc[pts[0][0] : pts[-1][0] + 1]))
            objs.append(
                DrawObject(
                    type="trendline",
                    name=_obj_name(overlay.id, "neck"),
                    color=color,
                    t1=t1,
                    p1=neck,
                    t2=bar_t,
                    p2=neck,
                    width=1,
                    style=2,
                    tooltip="Neckline / support break",
                )
            )
            objs.append(
                DrawObject(
                    type="trendline",
                    name=_obj_name(overlay.id, "top"),
                    color=color,
                    t1=t1,
                    p1=pts[0][1],
                    t2=t2,
                    p2=pts[-1][1],
                    width=1,
                    tooltip="Resistance rail",
                )
            )

    elif key in {"pat_double_bottom", "pat_triple_bottom", "pat_inv_head_shoulders", "pat_equal_lows"}:
        pts = lows[-3:] if "triple" in key or "head" in key else lows[-2:]
        for n, (idx, px) in enumerate(pts):
            t = _ts_unix(df["timestamp"].iloc[idx])
            objs.append(
                DrawObject(
                    type="arrow",
                    name=_obj_name(overlay.id, "al", n),
                    color=color,
                    time=t,
                    price=px - pad * 0.5,
                    arrow_code=251,
                    width=1,
                    tooltip=f"Pivot low #{n + 1}",
                )
            )
            overlay.anchors.append({"role": f"low_{n + 1}", "time": t, "price": px})
        if len(pts) >= 2:
            t1 = _ts_unix(df["timestamp"].iloc[pts[0][0]])
            t2 = _ts_unix(df["timestamp"].iloc[pts[-1][0]])
            neck = float(max(df["high"].iloc[pts[0][0] : pts[-1][0] + 1]))
            objs.append(
                DrawObject(
                    type="trendline",
                    name=_obj_name(overlay.id, "neck"),
                    color=color,
                    t1=t1,
                    p1=neck,
                    t2=bar_t,
                    p2=neck,
                    width=1,
                    style=2,
                    tooltip="Neckline / resistance break",
                )
            )

    elif any(x in key for x in ("triangle", "wedge", "flag", "pennant", "channel", "broadening")):
        if len(highs) >= 2 and len(lows) >= 2:
            h1, h2 = highs[-2], highs[-1]
            l1, l2 = lows[-2], lows[-1]
            objs.append(
                DrawObject(
                    type="trendline",
                    name=_obj_name(overlay.id, "upper"),
                    color=color,
                    t1=_ts_unix(df["timestamp"].iloc[h1[0]]),
                    p1=h1[1],
                    t2=_ts_unix(df["timestamp"].iloc[h2[0]]),
                    p2=h2[1],
                    width=1,
                    tooltip="Upper rail",
                )
            )
            objs.append(
                DrawObject(
                    type="trendline",
                    name=_obj_name(overlay.id, "lower"),
                    color=color,
                    t1=_ts_unix(df["timestamp"].iloc[l1[0]]),
                    p1=l1[1],
                    t2=_ts_unix(df["timestamp"].iloc[l2[0]]),
                    p2=l2[1],
                    width=1,
                    tooltip="Lower rail",
                )
            )

    elif any(x in key for x in ("rectangle", "cup_handle", "rounding")):
        look = max(5, min(i, 30))
        t1 = _ts_unix(df["timestamp"].iloc[i - look])
        box_hi = float(np.nanmax(df["high"].iloc[i - look : i + 1].to_numpy()))
        box_lo = float(np.nanmin(df["low"].iloc[i - look : i + 1].to_numpy()))
        objs.append(
            DrawObject(
                type="rectangle",
                name=_obj_name(overlay.id, "box"),
                color=color,
                t1=t1,
                p1=box_hi,
                t2=bar_t,
                p2=box_lo,
                width=1,
                fill=False,
                tooltip="Formation zone",
            )
        )

    elif key in {
        "pat_bos_up",
        "pat_bos_down",
        "pat_choch_bull",
        "pat_choch_bear",
        "pat_breakout_up",
        "pat_breakout_down",
        "pat_liquidity_sweep_high",
        "pat_liquidity_sweep_low",
    }:
        level = hi if "high" in key or key.endswith("_up") or "bull" in key else lo
        if "down" in key or "bear" in key or "sweep_high" in key:
            level = hi if "sweep_high" in key else lo
        look = max(3, min(i, 15))
        t1 = _ts_unix(df["timestamp"].iloc[i - look])
        objs.append(
            DrawObject(
                type="trendline",
                name=_obj_name(overlay.id, "lvl"),
                color=color,
                t1=t1,
                p1=level,
                t2=bar_t,
                p2=level,
                width=2,
                style=0,
                tooltip="Structure level",
            )
        )

    elif overlay.category == "candle" or overlay.category == "compound":
        # Highlight the candle body zone
        t_prev = bar_t if i == 0 else _ts_unix(df["timestamp"].iloc[max(0, i - 1)])
        objs.append(
            DrawObject(
                type="rectangle",
                name=_obj_name(overlay.id, "bar"),
                color=color,
                t1=t_prev,
                p1=hi,
                t2=bar_t,
                p2=lo,
                width=1,
                fill=False,
                tooltip=tip,
            )
        )

    # Trade-link annotation
    if overlay.status == PatternStatus.LINKED and (overlay.trade.side or overlay.trade.ticket):
        objs.append(
            DrawObject(
                type="text",
                name=_obj_name(overlay.id, "trade"),
                color="FFFFFF",
                time=bar_t,
                price=close,
                text=(
                    f"→ {str(overlay.trade.side or '').upper()} "
                    f"#{overlay.trade.ticket or 'paper'}"
                ),
                fontsize=9,
                tooltip=overlay.tooltip_text(),
            )
        )

    return objs
