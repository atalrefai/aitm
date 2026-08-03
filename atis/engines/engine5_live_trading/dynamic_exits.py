"""Model-driven dynamic Stop-Loss / Take-Profit levels.

Exit geometry is derived from prediction outputs (expected move, confidence,
risk), *near* market structure (local swings / S-R), and live volatility (ATR).

Levels stay close to entry — scaled to actual ATR — not fixed pip offsets and
not distant historical support/resistance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ExitLevels:
    sl: float
    tp: float
    side: str
    price: float
    sl_distance: float
    tp_distance: float
    reward_risk: float
    method: str
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(x: Any, default: float | None = None) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    return v


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _row_level(featured: pd.DataFrame | None, col: str) -> float | None:
    if featured is None or col not in featured.columns:
        return None
    return _finite(featured[col].iloc[-1])


def _within_atr(level: float | None, price: float, atr: float, max_atr: float) -> float | None:
    """Keep only structure levels that sit near the entry (≤ max_atr × ATR)."""
    if level is None or atr <= 0:
        return None
    dist = abs(float(level) - price)
    if dist <= 1e-12 or dist > max_atr * atr:
        return None
    return float(level)


def _local_swing_levels(
    featured: pd.DataFrame | None,
    *,
    price: float,
    atr: float,
    lookback: int,
    max_atr: float,
) -> tuple[float | None, float | None]:
    """Nearest micro-structure from recent bar extremes (closer than far ffill S/R)."""
    if featured is None or lookback < 2:
        return None, None
    need = {"high", "low"}
    if not need.issubset(featured.columns):
        return None, None
    n = min(int(lookback), len(featured))
    if n < 2:
        return None, None
    tail = featured.iloc[-n:]
    local_hi = _finite(tail["high"].max())
    local_lo = _finite(tail["low"].min())
    support = _within_atr(local_lo if local_lo is not None and local_lo < price else None, price, atr, max_atr)
    resist = _within_atr(local_hi if local_hi is not None and local_hi > price else None, price, atr, max_atr)
    return support, resist


def _nearest_level(*levels: float | None, price: float) -> float | None:
    """Pick the level closest to price (ignores None)."""
    valid = [float(x) for x in levels if x is not None]
    if not valid:
        return None
    return min(valid, key=lambda lvl: abs(lvl - price))


def aggregate_prediction_exits(
    votes: list[dict[str, Any]],
    fused_pred: int,
) -> dict[str, Any]:
    """Confidence-weighted expected_return / risk from TF votes that agree with fused side."""
    agreeing: list[tuple[float, float, float]] = []  # conf, exp_ret, risk
    for v in votes:
        if "error" in v:
            continue
        pred = int(v.get("pred") or 0)
        if fused_pred == 0 or pred != fused_pred:
            continue
        conf = _finite(v.get("conf"), 0.0) or 0.0
        exp = _finite(v.get("expected_return"))
        risk = _finite(v.get("risk_score"))
        if exp is None and risk is None:
            continue
        agreeing.append((max(conf, 1e-6), float(exp or 0.0), float(risk if risk is not None else 0.5)))

    if not agreeing:
        return {}

    w_sum = sum(c for c, _, _ in agreeing)
    exp_w = sum(c * e for c, e, _ in agreeing) / w_sum
    risk_w = sum(c * r for c, _, r in agreeing) / w_sum
    return {
        "expected_return": float(exp_w),
        "risk_score": float(risk_w),
        "exit_vote_count": len(agreeing),
    }


def compute_dynamic_sl_tp(
    *,
    price: float,
    side: str,
    atr_value: float,
    confidence: float,
    featured: pd.DataFrame | None = None,
    expected_return: float | None = None,
    risk_score: float | None = None,
    cfg: dict[str, Any] | None = None,
) -> ExitLevels:
    """Compute near-entry SL/TP from model forecasts + local structure + ATR.

    Raises ValueError when levels cannot be formed or R/R gate fails.
    """
    cfg = dict(cfg or {})
    side = side.lower().strip()
    if side not in {"buy", "sell"}:
        raise ValueError(f"invalid_side:{side}")
    price = float(price)
    atr = float(atr_value)
    conf = _clamp(float(confidence), 0.0, 1.0)
    if not np.isfinite(price) or price <= 0:
        raise ValueError("bad_price")
    if not np.isfinite(atr) or atr <= 0:
        raise ValueError("bad_atr")

    risk = _clamp(_finite(risk_score, 0.5) or 0.5, 0.0, 1.0)
    exp_ret = _finite(expected_return)

    min_rr = float(cfg.get("min_rr", 1.15))
    tp_conf_min = float(cfg.get("tp_confidence_min", 0.50))
    tp_conf_max = float(cfg.get("tp_confidence_max", 0.85))
    risk_dampen = float(cfg.get("risk_dampen", 0.40))
    # Tight ATR fallbacks — targets stay proportional to live volatility.
    tp_fb_min = float(cfg.get("tp_atr_fallback_min", 0.9))
    tp_fb_max = float(cfg.get("tp_atr_fallback_max", 2.0))
    tp_clamp_min = float(cfg.get("tp_atr_clamp_min", 0.6))
    tp_clamp_max = float(cfg.get("tp_atr_clamp_max", 2.4))
    sl_base = float(cfg.get("sl_atr_base", 0.85))
    sl_risk_extra = float(cfg.get("sl_atr_risk_extra", 0.55))
    sl_conf_tighten = float(cfg.get("sl_atr_conf_tighten", 0.35))
    sl_clamp_min = float(cfg.get("sl_atr_clamp_min", 0.45))
    sl_clamp_max = float(cfg.get("sl_atr_clamp_max", 1.8))
    struct_buf = float(cfg.get("structure_buffer_atr", 0.12))
    struct_sl_max = float(cfg.get("structure_sl_max_atr", 1.6))
    struct_tp_max = float(cfg.get("structure_tp_max_atr", 2.0))
    struct_reach_max = float(cfg.get("structure_reach_max_atr", 2.2))
    struct_sl_min_vs_vol = float(cfg.get("structure_sl_min_vs_vol", 0.45))
    local_lookback = int(cfg.get("local_swing_lookback", 12))
    struct_tp_snap = bool(cfg.get("structure_tp_snap", True))
    skip_rr = bool(cfg.get("skip_if_rr_below_min", True))

    raw_support = _row_level(featured, "support_level")
    raw_resist = _row_level(featured, "resist_level")
    # Discard distant ffill swings — only near-entry structure is actionable.
    chart_support = _within_atr(raw_support if raw_support is not None and raw_support < price else None,
                                price, atr, struct_reach_max)
    chart_resist = _within_atr(raw_resist if raw_resist is not None and raw_resist > price else None,
                               price, atr, struct_reach_max)
    local_support, local_resist = _local_swing_levels(
        featured, price=price, atr=atr, lookback=local_lookback, max_atr=struct_reach_max
    )
    # Prefer the nearer of chart S/R and recent bar extremes.
    support = _nearest_level(chart_support, local_support, price=price)
    resist = _nearest_level(chart_resist, local_resist, price=price)
    trend_strength = _row_level(featured, "trend_strength")
    adx_v = _row_level(featured, "adx")

    # --- Expected move (TP magnitude) ---
    method_bits: list[str] = []
    aligned_exp: float | None = None
    if exp_ret is not None and abs(exp_ret) > 1e-12:
        # Sign must agree with trade side; otherwise ignore magnitude head.
        if (side == "buy" and exp_ret > 0) or (side == "sell" and exp_ret < 0):
            aligned_exp = abs(exp_ret)
            method_bits.append("model_expected_return")
        else:
            method_bits.append("expected_return_misaligned")

    if aligned_exp is not None:
        raw_move = aligned_exp * price
    else:
        # Confidence / trend-scaled ATR fallback (still dynamic, not fixed 2.5×).
        trend_boost = 0.0
        if trend_strength is not None:
            trend_boost = _clamp(abs(trend_strength) / 50.0, 0.0, 0.25)
        elif adx_v is not None:
            trend_boost = _clamp((adx_v - 15.0) / 40.0, 0.0, 0.25)
        atr_mult = tp_fb_min + (tp_fb_max - tp_fb_min) * conf + trend_boost * (tp_fb_max - tp_fb_min)
        raw_move = atr_mult * atr
        method_bits.append("atr_confidence_fallback")

    reach = tp_conf_min + (tp_conf_max - tp_conf_min) * conf
    reach *= 1.0 - risk_dampen * risk
    reach = _clamp(reach, 0.35, 1.0)
    tp_dist = raw_move * reach

    # Structure snap for TP — only when the opposite level is near entry.
    structure_tp: float | None = None
    if struct_tp_snap:
        if side == "buy" and resist is not None and resist > price:
            structure_tp = resist - struct_buf * atr
        elif side == "sell" and support is not None and support < price:
            structure_tp = support + struct_buf * atr
        if structure_tp is not None:
            struct_dist = abs(structure_tp - price)
            max_struct = struct_tp_max * atr
            lo = 0.40 * tp_dist
            hi = min(1.35 * tp_dist, max_struct)
            if lo <= struct_dist <= hi and struct_dist <= max_struct:
                tp_dist = struct_dist
                method_bits.append("structure_tp")
            elif struct_dist > max_struct:
                method_bits.append("structure_tp_rejected_far")

    tp_dist = _clamp(tp_dist, tp_clamp_min * atr, tp_clamp_max * atr)

    # --- Stop Loss (thesis invalidation, near entry) ---
    vol_sl_mult = sl_base + sl_risk_extra * risk - sl_conf_tighten * max(conf - 0.5, 0.0)
    vol_sl_mult = _clamp(vol_sl_mult, sl_clamp_min, sl_clamp_max)
    vol_sl_dist = vol_sl_mult * atr
    method_bits.append("atr_risk_sl")

    structure_sl_dist: float | None = None
    if side == "buy" and support is not None and support < price:
        cand = (price - support) + struct_buf * atr
        if sl_clamp_min * atr <= cand <= struct_sl_max * atr:
            structure_sl_dist = cand
            method_bits.append("structure_sl")
    elif side == "sell" and resist is not None and resist > price:
        cand = (resist - price) + struct_buf * atr
        if sl_clamp_min * atr <= cand <= struct_sl_max * atr:
            structure_sl_dist = cand
            method_bits.append("structure_sl")

    # Prefer structural invalidation when it clears a modest noise floor vs ATR stop.
    used_structure_sl = False
    if structure_sl_dist is not None and structure_sl_dist >= struct_sl_min_vs_vol * vol_sl_dist:
        sl_dist = structure_sl_dist
        used_structure_sl = True
    else:
        sl_dist = vol_sl_dist
        if structure_sl_dist is not None:
            method_bits.append("structure_sl_rejected_too_tight")

    sl_dist = _clamp(sl_dist, sl_clamp_min * atr, sl_clamp_max * atr)

    rr = tp_dist / max(sl_dist, 1e-12)
    if rr < min_rr:
        # If structure stop is too far for the model's expected move, fall back to
        # volatility invalidation so TP remains model-driven and R/R stays coherent.
        if used_structure_sl:
            sl_dist = _clamp(vol_sl_dist, sl_clamp_min * atr, sl_clamp_max * atr)
            rr = tp_dist / max(sl_dist, 1e-12)
            used_structure_sl = False
            method_bits = [m for m in method_bits if m != "structure_sl"]
            method_bits.append("structure_sl_rr_fallback_atr")

        if rr < min_rr:
            needed_sl = tp_dist / max(min_rr, 1e-9)
            min_allowed = sl_clamp_min * atr
            if needed_sl >= min_allowed and needed_sl < sl_dist:
                sl_dist = max(needed_sl, min_allowed)
                rr = tp_dist / max(sl_dist, 1e-12)
                method_bits.append("sl_tightened_for_rr")
            elif skip_rr:
                raise ValueError(f"rr_below_min:{rr:.3f}<{min_rr}")

    if side == "buy":
        sl = price - sl_dist
        tp = price + tp_dist
    else:
        sl = price + sl_dist
        tp = price - tp_dist

    if sl <= 0 or tp <= 0:
        raise ValueError("non_positive_sl_tp")

    return ExitLevels(
        sl=float(sl),
        tp=float(tp),
        side=side,
        price=price,
        sl_distance=float(sl_dist),
        tp_distance=float(tp_dist),
        reward_risk=float(rr),
        method="+".join(method_bits),
        meta={
            "expected_return": exp_ret,
            "aligned_expected_return": aligned_exp,
            "risk_score": risk,
            "confidence": conf,
            "atr": atr,
            "raw_move": float(raw_move),
            "reach_fraction": float(reach),
            "support_level": support,
            "resist_level": resist,
            "raw_support_level": raw_support,
            "raw_resist_level": raw_resist,
            "local_support": local_support,
            "local_resist": local_resist,
            "trend_strength": trend_strength,
            "adx": adx_v,
            "min_rr": min_rr,
            "vol_sl_mult": float(vol_sl_mult),
            "sl_atr_mult": float(sl_dist / atr),
            "tp_atr_mult": float(tp_dist / atr),
        },
    )
