"""
Advanced candlestick + structural + compound pattern detection (causal, no look-ahead).

All confirmations at index i use only bars <= i.
PATTERN_CATALOG is the single source of truth for labels / bias / category.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Catalog — keys, Arabic/English labels, category, directional bias
# ---------------------------------------------------------------------------

PATTERN_CATALOG: dict[str, dict[str, str]] = {
    # --- single candle ---
    "pat_doji": {"name": "Doji", "category": "candle", "bias": "neutral", "conditions": "body_pct<0.1"},
    "pat_long_legged_doji": {"name": "Long-Legged Doji", "category": "candle", "bias": "neutral", "conditions": "body_pct<0.1 & long wicks"},
    "pat_dragonfly_doji": {"name": "Dragonfly Doji", "category": "candle", "bias": "bullish", "conditions": "doji + long lower wick"},
    "pat_gravestone_doji": {"name": "Gravestone Doji", "category": "candle", "bias": "bearish", "conditions": "doji + long upper wick"},
    "pat_spinning_top": {"name": "Spinning Top", "category": "candle", "bias": "neutral", "conditions": "0.1<body_pct<0.3 & both wicks"},
    "pat_high_wave": {"name": "High Wave", "category": "candle", "bias": "neutral", "conditions": "very long both wicks"},
    "pat_rickshaw_man": {"name": "Rickshaw Man", "category": "candle", "bias": "neutral", "conditions": "long-legged doji mid-range"},
    "pat_marubozu_bull": {"name": "Bullish Marubozu", "category": "candle", "bias": "bullish", "conditions": "bull & body_pct>0.9"},
    "pat_marubozu_bear": {"name": "Bearish Marubozu", "category": "candle", "bias": "bearish", "conditions": "bear & body_pct>0.9"},
    "pat_hammer": {"name": "Hammer", "category": "candle", "bias": "bullish", "conditions": "lower>2*body & short upper & prior downtrend (close<EMA20)"},
    "pat_inverted_hammer": {"name": "Inverted Hammer", "category": "candle", "bias": "bullish", "conditions": "upper>2*body & short lower & prior downtrend"},
    "pat_hanging_man": {"name": "Hanging Man", "category": "candle", "bias": "bearish", "conditions": "hammer shape & prior uptrend (close>EMA20)"},
    "pat_shooting_star": {"name": "Shooting Star", "category": "candle", "bias": "bearish", "conditions": "long upper wick & prior uptrend"},
    "pat_belt_hold_bull": {"name": "Bullish Belt Hold", "category": "candle", "bias": "bullish", "conditions": "opens near low, closes strong up"},
    "pat_belt_hold_bear": {"name": "Bearish Belt Hold", "category": "candle", "bias": "bearish", "conditions": "opens near high, closes strong down"},
    "pat_inside_bar": {"name": "Inside Bar", "category": "candle", "bias": "neutral", "conditions": "range inside prior bar"},
    "pat_outside_bar": {"name": "Outside Bar", "category": "candle", "bias": "neutral", "conditions": "engulfs prior range"},
    "pat_nr7": {"name": "NR7 Narrow Range", "category": "candle", "bias": "neutral", "conditions": "narrowest range of last 7"},
    "pat_wide_range": {"name": "Wide Range Bar", "category": "candle", "bias": "neutral", "conditions": "range > 2x avg"},
    # --- two candle ---
    "pat_bull_engulf": {"name": "Bullish Engulfing", "category": "candle", "bias": "bullish", "conditions": "bull body engulfs prior bear"},
    "pat_bear_engulf": {"name": "Bearish Engulfing", "category": "candle", "bias": "bearish", "conditions": "bear body engulfs prior bull"},
    "pat_bull_harami": {"name": "Bullish Harami", "category": "candle", "bias": "bullish", "conditions": "small bull inside prior bear"},
    "pat_bear_harami": {"name": "Bearish Harami", "category": "candle", "bias": "bearish", "conditions": "small bear inside prior bull"},
    "pat_piercing": {"name": "Piercing Line", "category": "candle", "bias": "bullish", "conditions": "bull closes >50% into prior bear"},
    "pat_dark_cloud": {"name": "Dark Cloud Cover", "category": "candle", "bias": "bearish", "conditions": "bear closes >50% into prior bull"},
    "pat_tweezer_bottom": {"name": "Tweezer Bottom", "category": "candle", "bias": "bullish", "conditions": "matching lows after decline"},
    "pat_tweezer_top": {"name": "Tweezer Top", "category": "candle", "bias": "bearish", "conditions": "matching highs after rally"},
    "pat_meeting_bull": {"name": "Bullish Meeting Lines", "category": "candle", "bias": "bullish", "conditions": "bear then bull close near equal; moderate bodies (quiet)"},
    "pat_meeting_bear": {"name": "Bearish Meeting Lines", "category": "candle", "bias": "bearish", "conditions": "bull then bear close near equal; moderate bodies (quiet)"},
    "pat_matching_low": {"name": "Matching Low", "category": "candle", "bias": "bullish", "conditions": "two bear candles same close"},
    "pat_matching_high": {"name": "Matching High", "category": "candle", "bias": "bearish", "conditions": "two bull candles same close"},
    "pat_kicking_bull": {"name": "Bullish Kicking", "category": "candle", "bias": "bullish", "conditions": "bear marubozu → gap-up bull marubozu"},
    "pat_kicking_bear": {"name": "Bearish Kicking", "category": "candle", "bias": "bearish", "conditions": "bull marubozu → gap-down bear marubozu"},
    "pat_counterattack_bull": {"name": "Bullish Counterattack", "category": "candle", "bias": "bullish", "conditions": "strong bear then gap-down open (>0.08%) + strong bull close equal"},
    "pat_counterattack_bear": {"name": "Bearish Counterattack", "category": "candle", "bias": "bearish", "conditions": "strong bull then gap-up open (>0.08%) + strong bear close equal"},
    "pat_separating_bull": {"name": "Bullish Separating Lines", "category": "candle", "bias": "bullish", "conditions": "bear then bull open=prior open"},
    "pat_separating_bear": {"name": "Bearish Separating Lines", "category": "candle", "bias": "bearish", "conditions": "bull then bear open=prior open"},
    "pat_thrusting": {"name": "Thrusting", "category": "candle", "bias": "bearish", "conditions": "bull closes into prior bear <50%"},
    "pat_on_neck": {"name": "On-Neck", "category": "candle", "bias": "bearish", "conditions": "bull closes near prior low"},
    "pat_in_neck": {"name": "In-Neck", "category": "candle", "bias": "bearish", "conditions": "bull closes slightly into prior body"},
    # --- three / multi candle ---
    "pat_morning_star": {"name": "Morning Star", "category": "candle", "bias": "bullish", "conditions": "bear + small + strong bull"},
    "pat_evening_star": {"name": "Evening Star", "category": "candle", "bias": "bearish", "conditions": "bull + small + strong bear"},
    "pat_abandoned_baby_bull": {"name": "Bullish Abandoned Baby", "category": "candle", "bias": "bullish", "conditions": "gapped doji then gap-up bull"},
    "pat_abandoned_baby_bear": {"name": "Bearish Abandoned Baby", "category": "candle", "bias": "bearish", "conditions": "gapped doji then gap-down bear"},
    "pat_three_soldiers": {"name": "Three White Soldiers", "category": "candle", "bias": "bullish", "conditions": "3 rising bull bodies"},
    "pat_three_crows": {"name": "Three Black Crows", "category": "candle", "bias": "bearish", "conditions": "3 falling bear bodies"},
    "pat_three_inside_up": {"name": "Three Inside Up", "category": "candle", "bias": "bullish", "conditions": "bull harami + confirm close"},
    "pat_three_inside_down": {"name": "Three Inside Down", "category": "candle", "bias": "bearish", "conditions": "bear harami + confirm close"},
    "pat_three_outside_up": {"name": "Three Outside Up", "category": "candle", "bias": "bullish", "conditions": "bull engulf + higher close"},
    "pat_three_outside_down": {"name": "Three Outside Down", "category": "candle", "bias": "bearish", "conditions": "bear engulf + lower close"},
    "pat_rising_three": {"name": "Rising Three Methods", "category": "candle", "bias": "bullish", "conditions": "bull + 3 small down + bull"},
    "pat_falling_three": {"name": "Falling Three Methods", "category": "candle", "bias": "bearish", "conditions": "bear + 3 small up + bear"},
    "pat_stick_sandwich": {"name": "Stick Sandwich", "category": "candle", "bias": "bullish", "conditions": "bear / bull / bear same close"},
    "pat_unique_three_river": {"name": "Unique Three River", "category": "candle", "bias": "bullish", "conditions": "long bear + hammer-like + small bull"},
    "pat_two_crows": {"name": "Two Crows", "category": "candle", "bias": "bearish", "conditions": "bull + gap bear + bear into body"},
    "pat_upside_gap_two_crows": {"name": "Upside Gap Two Crows", "category": "candle", "bias": "bearish", "conditions": "bull + 2 gap bears"},
    "pat_advance_block": {"name": "Advance Block", "category": "candle", "bias": "bearish", "conditions": "3 bulls weakening"},
    "pat_deliberation": {"name": "Deliberation", "category": "candle", "bias": "bearish", "conditions": "2 strong bulls + small top"},
    "pat_ladder_bottom": {"name": "Ladder Bottom", "category": "candle", "bias": "bullish", "conditions": "3 bears + inverted hammer + bull"},
    "pat_tasuki_up": {"name": "Upside Tasuki Gap", "category": "candle", "bias": "bullish", "conditions": "bull gap + partial fill"},
    "pat_tasuki_down": {"name": "Downside Tasuki Gap", "category": "candle", "bias": "bearish", "conditions": "bear gap + partial fill"},
    # --- chart / structural ---
    "pat_double_top": {"name": "Double Top", "category": "chart", "bias": "bearish", "conditions": "2 similar highs + break mid trough"},
    "pat_double_bottom": {"name": "Double Bottom", "category": "chart", "bias": "bullish", "conditions": "2 similar lows + break mid peak"},
    "pat_triple_top": {"name": "Triple Top", "category": "chart", "bias": "bearish", "conditions": "3 similar highs + support break"},
    "pat_triple_bottom": {"name": "Triple Bottom", "category": "chart", "bias": "bullish", "conditions": "3 similar lows + resist break"},
    "pat_head_shoulders": {"name": "Head & Shoulders", "category": "chart", "bias": "bearish", "conditions": "3 highs middle highest + neck break"},
    "pat_inv_head_shoulders": {"name": "Inverse Head & Shoulders", "category": "chart", "bias": "bullish", "conditions": "3 lows middle lowest + neck break"},
    "pat_triangle_asc": {"name": "Ascending Triangle", "category": "chart", "bias": "bullish", "conditions": "flat highs + rising lows"},
    "pat_triangle_desc": {"name": "Descending Triangle", "category": "chart", "bias": "bearish", "conditions": "flat lows + falling highs"},
    "pat_triangle_sym": {"name": "Symmetrical Triangle", "category": "chart", "bias": "neutral", "conditions": "falling highs + rising lows"},
    "pat_wedge_rising": {"name": "Rising Wedge", "category": "chart", "bias": "bearish", "conditions": "rising highs+lows converging"},
    "pat_wedge_falling": {"name": "Falling Wedge", "category": "chart", "bias": "bullish", "conditions": "falling highs+lows converging"},
    "pat_flag_bull": {"name": "Bull Flag", "category": "chart", "bias": "bullish", "conditions": "impulse up + tight consolidation"},
    "pat_flag_bear": {"name": "Bear Flag", "category": "chart", "bias": "bearish", "conditions": "impulse down + tight consolidation"},
    "pat_channel_up": {"name": "Ascending Channel", "category": "chart", "bias": "bullish", "conditions": "parallel rising swing rails"},
    "pat_channel_down": {"name": "Descending Channel", "category": "chart", "bias": "bearish", "conditions": "parallel falling swing rails"},
    "pat_breakout_up": {"name": "Breakout Up", "category": "chart", "bias": "bullish", "conditions": "close > prior 20-bar high"},
    "pat_breakout_down": {"name": "Breakout Down", "category": "chart", "bias": "bearish", "conditions": "close < prior 20-bar low"},
    "pat_bos_up": {"name": "Break of Structure Up", "category": "chart", "bias": "bullish", "conditions": "close breaks last swing high"},
    "pat_bos_down": {"name": "Break of Structure Down", "category": "chart", "bias": "bearish", "conditions": "close breaks last swing low"},
    "pat_choch_bull": {"name": "Change of Character Bull", "category": "chart", "bias": "bullish", "conditions": "down structure then BOS up"},
    "pat_choch_bear": {"name": "Change of Character Bear", "category": "chart", "bias": "bearish", "conditions": "up structure then BOS down"},
    "pat_equal_highs": {"name": "Equal Highs", "category": "chart", "bias": "bearish", "conditions": "two swing highs within 0.15%"},
    "pat_equal_lows": {"name": "Equal Lows", "category": "chart", "bias": "bullish", "conditions": "two swing lows within 0.15%"},
    "pat_liquidity_sweep_high": {"name": "Liquidity Sweep High", "category": "chart", "bias": "bearish", "conditions": "pierce prior high then close back"},
    "pat_liquidity_sweep_low": {"name": "Liquidity Sweep Low", "category": "chart", "bias": "bullish", "conditions": "pierce prior low then close back"},
    "pat_rounding_bottom": {"name": "Rounding Bottom", "category": "chart", "bias": "bullish", "conditions": "U-shaped swing lows"},
    "pat_rounding_top": {"name": "Rounding Top", "category": "chart", "bias": "bearish", "conditions": "inverted-U swing highs"},
    # --- advanced chart / wave / harmonic (Phase-1 catalog gaps) ---
    "pat_cup_handle": {
        "name": "Cup and Handle",
        "category": "chart",
        "bias": "bullish",
        "conditions": "U-cup recovery + shallow handle pullback + close>handle_high",
    },
    "pat_pennant_bull": {
        "name": "Bull Pennant",
        "category": "chart",
        "bias": "bullish",
        "conditions": "impulse_up + converging consolidation + close>flag_high",
    },
    "pat_pennant_bear": {
        "name": "Bear Pennant",
        "category": "chart",
        "bias": "bearish",
        "conditions": "impulse_down + converging consolidation + close<flag_low",
    },
    "pat_rectangle_bull": {
        "name": "Rectangle Breakout Up",
        "category": "chart",
        "bias": "bullish",
        "conditions": "flat_highs & flat_lows range then close>range_high",
    },
    "pat_rectangle_bear": {
        "name": "Rectangle Breakout Down",
        "category": "chart",
        "bias": "bearish",
        "conditions": "flat_highs & flat_lows range then close<range_low",
    },
    "pat_broadening_up": {
        "name": "Broadening Formation Up",
        "category": "chart",
        "bias": "bullish",
        "conditions": "diverging swing highs+lows then close>last_high",
    },
    "pat_broadening_down": {
        "name": "Broadening Formation Down",
        "category": "chart",
        "bias": "bearish",
        "conditions": "diverging swing highs+lows then close<last_low",
    },
    "pat_compression": {
        "name": "Volatility Compression",
        "category": "chart",
        "bias": "neutral",
        "conditions": "ATR(5)/ATR(20)<0.55 & range shrinking",
    },
    "pat_compression_breakout_up": {
        "name": "Compression Breakout Up",
        "category": "chart",
        "bias": "bullish",
        "conditions": "compression then close>prior_20_high",
    },
    "pat_compression_breakout_down": {
        "name": "Compression Breakout Down",
        "category": "chart",
        "bias": "bearish",
        "conditions": "compression then close<prior_20_low",
    },
    "pat_harmonic_gartley_bull": {
        "name": "Bullish Gartley",
        "category": "chart",
        "bias": "bullish",
        "conditions": "XA/AB≈0.618 BC≈0.382-0.886 AD≈0.786 XA at D",
    },
    "pat_harmonic_gartley_bear": {
        "name": "Bearish Gartley",
        "category": "chart",
        "bias": "bearish",
        "conditions": "mirror Gartley ratios at D",
    },
    "pat_harmonic_bat_bull": {
        "name": "Bullish Bat",
        "category": "chart",
        "bias": "bullish",
        "conditions": "AB≈0.382-0.5 XA AD≈0.886 XA",
    },
    "pat_harmonic_bat_bear": {
        "name": "Bearish Bat",
        "category": "chart",
        "bias": "bearish",
        "conditions": "mirror Bat ratios at D",
    },
    "pat_harmonic_butterfly_bull": {
        "name": "Bullish Butterfly",
        "category": "chart",
        "bias": "bullish",
        "conditions": "AB≈0.786 XA AD≈1.27 XA",
    },
    "pat_harmonic_butterfly_bear": {
        "name": "Bearish Butterfly",
        "category": "chart",
        "bias": "bearish",
        "conditions": "mirror Butterfly ratios at D",
    },
    "pat_harmonic_crab_bull": {
        "name": "Bullish Crab",
        "category": "chart",
        "bias": "bullish",
        "conditions": "AB≈0.382-0.618 XA AD≈1.618 XA",
    },
    "pat_harmonic_crab_bear": {
        "name": "Bearish Crab",
        "category": "chart",
        "bias": "bearish",
        "conditions": "mirror Crab ratios at D",
    },
    "pat_elliott_impulse_bull": {
        "name": "Elliott Impulse Bull (1-5)",
        "category": "chart",
        "bias": "bullish",
        "conditions": "5 alternating swings up with wave3 longest & wave4>wave1",
    },
    "pat_elliott_impulse_bear": {
        "name": "Elliott Impulse Bear (1-5)",
        "category": "chart",
        "bias": "bearish",
        "conditions": "5 alternating swings down with wave3 longest & wave4<wave1",
    },
    "pat_wolfe_bull": {
        "name": "Bullish Wolfe Wave",
        "category": "chart",
        "bias": "bullish",
        "conditions": "5-point converging channel; point5 pierce then reverse up",
    },
    "pat_wolfe_bear": {
        "name": "Bearish Wolfe Wave",
        "category": "chart",
        "bias": "bearish",
        "conditions": "5-point converging channel; point5 pierce then reverse down",
    },
}

# Known high-value compound templates (overlapping same-bar)
COMPOUND_TEMPLATES: list[tuple[str, str, str, str]] = [
    ("cmp_hammer_support", "Hammer @ Support", "pat_hammer", "near_support"),
    ("cmp_shooting_resist", "Shooting Star @ Resistance", "pat_shooting_star", "near_resist"),
    ("cmp_engulf_bos_up", "Bull Engulf + BOS Up", "pat_bull_engulf", "pat_bos_up"),
    ("cmp_engulf_bos_down", "Bear Engulf + BOS Down", "pat_bear_engulf", "pat_bos_down"),
    ("cmp_morning_double_bot", "Morning Star + Double Bottom", "pat_morning_star", "pat_double_bottom"),
    ("cmp_evening_double_top", "Evening Star + Double Top", "pat_evening_star", "pat_double_top"),
    ("cmp_breakout_flag_bull", "Breakout + Bull Flag", "pat_breakout_up", "pat_flag_bull"),
    ("cmp_breakout_flag_bear", "Breakout + Bear Flag", "pat_breakout_down", "pat_flag_bear"),
    ("cmp_pin_sweep_low", "Hammer + Sweep Low", "pat_hammer", "pat_liquidity_sweep_low"),
    ("cmp_pin_sweep_high", "Shooting + Sweep High", "pat_shooting_star", "pat_liquidity_sweep_high"),
    ("cmp_inside_nr7", "Inside Bar + NR7", "pat_inside_bar", "pat_nr7"),
    ("cmp_soldiers_breakout", "Soldiers + Breakout Up", "pat_three_soldiers", "pat_breakout_up"),
    ("cmp_crows_breakout", "Crows + Breakout Down", "pat_three_crows", "pat_breakout_down"),
]


# Explicit aliases kept for Engine3/Engine4 column compatibility (not duplicate detectors).
PATTERN_ALIASES: dict[str, str] = {
    # Historical: counterattack previously mirrored meeting; keys remain independent
    # after rule split — aliases reserved for future merges.
}


def pattern_labels() -> dict[str, str]:
    labels = {k: v["name"] for k, v in PATTERN_CATALOG.items()}
    for key, name, _, _ in COMPOUND_TEMPLATES:
        labels[key] = name
    return labels


def pattern_category_map() -> dict[str, str]:
    cats = {k: v["category"] for k, v in PATTERN_CATALOG.items()}
    for key, _, _, _ in COMPOUND_TEMPLATES:
        cats[key] = "compound"
    return cats


def bullish_keys() -> set[str]:
    keys = {k for k, v in PATTERN_CATALOG.items() if v["bias"] == "bullish"}
    keys.update(
        {
            "cmp_hammer_support",
            "cmp_engulf_bos_up",
            "cmp_morning_double_bot",
            "cmp_breakout_flag_bull",
            "cmp_pin_sweep_low",
            "cmp_soldiers_breakout",
        }
    )
    return keys


def bearish_keys() -> set[str]:
    keys = {k for k, v in PATTERN_CATALOG.items() if v["bias"] == "bearish"}
    keys.update(
        {
            "cmp_shooting_resist",
            "cmp_engulf_bos_down",
            "cmp_evening_double_top",
            "cmp_breakout_flag_bear",
            "cmp_pin_sweep_high",
            "cmp_crows_breakout",
        }
    )
    return keys


# ---------------------------------------------------------------------------
# Candlestick helpers
# ---------------------------------------------------------------------------

def _candle_parts(df: pd.DataFrame) -> dict[str, pd.Series]:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    bull = c > o
    bear = c < o
    return {
        "o": o,
        "h": h,
        "l": l,
        "c": c,
        "body": body,
        "rng": rng,
        "upper": upper,
        "lower": lower,
        "bull": bull,
        "bear": bear,
        "body_pct": body / rng,
        "avg_body": body.rolling(14, min_periods=5).mean(),
        "avg_rng": rng.rolling(14, min_periods=5).mean(),
    }


def candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Expanded Japanese candlestick set with binary flags + strength scores."""
    p = _candle_parts(df)
    o, h, l, c = p["o"], p["h"], p["l"], p["c"]
    body, rng, upper, lower = p["body"], p["rng"], p["upper"], p["lower"]
    bull, bear = p["bull"], p["bear"]
    body_pct = p["body_pct"]
    avg_body = p["avg_body"]
    avg_rng = p["avg_rng"]

    out = pd.DataFrame(index=df.index)

    # --- single ---
    out["pat_doji"] = (body_pct < 0.1).astype(int)
    out["pat_long_legged_doji"] = ((body_pct < 0.1) & (upper > body) & (lower > body)).astype(int)
    out["pat_dragonfly_doji"] = ((body_pct < 0.1) & (lower > 2 * upper) & (upper / rng < 0.1)).astype(int)
    out["pat_gravestone_doji"] = ((body_pct < 0.1) & (upper > 2 * lower) & (lower / rng < 0.1)).astype(int)
    out["pat_spinning_top"] = ((body_pct > 0.1) & (body_pct < 0.3) & (upper > body) & (lower > body)).astype(int)
    out["pat_high_wave"] = ((upper > 2 * body) & (lower > 2 * body) & (body_pct < 0.25)).astype(int)
    out["pat_rickshaw_man"] = (
        (body_pct < 0.1) & (upper > avg_rng * 0.4) & (lower > avg_rng * 0.4)
    ).astype(int)
    out["pat_marubozu_bull"] = (bull & (body_pct > 0.9)).astype(int)
    out["pat_marubozu_bear"] = (bear & (body_pct > 0.9)).astype(int)

    # Causal trend context: prior bar vs EMA20 (no look-ahead)
    ema20 = c.ewm(span=20, adjust=False, min_periods=10).mean()
    prior_down = c.shift(1) < ema20.shift(1)
    prior_up = c.shift(1) > ema20.shift(1)
    hammer_shape = (lower > 2 * body) & (upper < body * 0.5) & (body_pct < 0.35)
    inv_hammer_shape = (upper > 2 * body) & (lower < body * 0.5) & (body_pct < 0.35)
    out["pat_hammer"] = (hammer_shape & prior_down).astype(int)
    out["pat_inverted_hammer"] = (inv_hammer_shape & prior_down).astype(int)
    out["pat_hanging_man"] = (hammer_shape & prior_up).astype(int)
    out["pat_shooting_star"] = (inv_hammer_shape & prior_up).astype(int)
    out["pat_belt_hold_bull"] = (bull & (lower / rng < 0.05) & (body_pct > 0.7)).astype(int)
    out["pat_belt_hold_bear"] = (bear & (upper / rng < 0.05) & (body_pct > 0.7)).astype(int)
    out["pat_inside_bar"] = ((h <= h.shift(1)) & (l >= l.shift(1))).astype(int)
    out["pat_outside_bar"] = ((h >= h.shift(1)) & (l <= l.shift(1))).astype(int)
    roll_min_rng = rng.rolling(7, min_periods=7).min()
    out["pat_nr7"] = ((rng <= roll_min_rng) & rng.notna()).astype(int)
    out["pat_wide_range"] = (rng > 2 * avg_rng).astype(int)

    # --- two ---
    po, pc = o.shift(1), c.shift(1)
    pb = body.shift(1)
    out["pat_bull_engulf"] = ((pc < po) & bull & (c >= po) & (o <= pc) & (body > pb)).astype(int)
    out["pat_bear_engulf"] = ((pc > po) & bear & (c <= po) & (o >= pc) & (body > pb)).astype(int)
    out["pat_bull_harami"] = ((pc < po) & bull & (o > pc) & (c < po) & (body < pb * 0.7)).astype(int)
    out["pat_bear_harami"] = ((pc > po) & bear & (o < pc) & (c > po) & (body < pb * 0.7)).astype(int)
    out["pat_piercing"] = ((pc < po) & bull & (o < pc) & (c > (po + pc) / 2) & (c < po)).astype(int)
    out["pat_dark_cloud"] = ((pc > po) & bear & (o > pc) & (c < (po + pc) / 2) & (c > po)).astype(int)
    out["pat_tweezer_bottom"] = ((pc < po) & bull & ((l - l.shift(1)).abs() / c < 0.0008)).astype(int)
    out["pat_tweezer_top"] = ((pc > po) & bear & ((h - h.shift(1)).abs() / c < 0.0008)).astype(int)
    close_eq = (c - pc).abs() / c < 0.0006
    prior_body_pct = pb / rng.shift(1).replace(0, np.nan)
    # Meeting lines: quiet — moderate bodies, equal closes, no aggressive gap
    meeting_moderate = (
        (body_pct > 0.12)
        & (body_pct < 0.55)
        & (prior_body_pct > 0.12)
        & (prior_body_pct < 0.55)
    )
    quiet_open = (o - pc).abs() / c < 0.0012  # open near prior close (no meaningful gap)
    out["pat_meeting_bull"] = ((pc < po) & bull & close_eq & meeting_moderate & quiet_open).astype(int)
    out["pat_meeting_bear"] = ((pc > po) & bear & close_eq & meeting_moderate & quiet_open).astype(int)
    out["pat_matching_low"] = ((pc < po) & bear & close_eq).astype(int)
    out["pat_matching_high"] = ((pc > po) & bull & close_eq).astype(int)
    gap_up = o > h.shift(1)
    gap_dn = o < l.shift(1)
    out["pat_kicking_bull"] = (
        (out["pat_marubozu_bear"].shift(1) == 1) & gap_up & (out["pat_marubozu_bull"] == 1)
    ).astype(int)
    out["pat_kicking_bear"] = (
        (out["pat_marubozu_bull"].shift(1) == 1) & gap_dn & (out["pat_marubozu_bear"] == 1)
    ).astype(int)
    # Counterattack: strong bodies + meaningful gap against prior direction + equal closes
    # (mutually exclusive with meeting's quiet_open)
    strong_bodies = (body > avg_body * 0.85) & (pb > avg_body.shift(1) * 0.85)
    slight_gap_dn = (pc - o) / c > 0.0008  # opens meaningfully below prior close
    slight_gap_up = (o - pc) / c > 0.0008
    out["pat_counterattack_bull"] = (
        (pc < po) & bull & close_eq & strong_bodies & slight_gap_dn
    ).astype(int)
    out["pat_counterattack_bear"] = (
        (pc > po) & bear & close_eq & strong_bodies & slight_gap_up
    ).astype(int)
    open_eq = (o - po).abs() / c < 0.0006
    out["pat_separating_bull"] = ((pc < po) & bull & open_eq).astype(int)
    out["pat_separating_bear"] = ((pc > po) & bear & open_eq).astype(int)
    out["pat_thrusting"] = (
        (pc < po) & bull & (o < pc) & (c > pc) & (c < (po + pc) / 2)
    ).astype(int)
    out["pat_on_neck"] = ((pc < po) & bull & ((c - l.shift(1)).abs() / c < 0.0008)).astype(int)
    out["pat_in_neck"] = (
        (pc < po) & bull & (c > l.shift(1)) & (c < (po + pc) / 2)
    ).astype(int)

    # --- three / multi ---
    mid_small = body.shift(1) / rng.shift(1) < 0.3
    out["pat_morning_star"] = (
        (pc.shift(1) < po.shift(1)) & mid_small & bull & (c > ((po.shift(1) + pc.shift(1)) / 2))
    ).astype(int)
    out["pat_evening_star"] = (
        (pc.shift(1) > po.shift(1)) & mid_small & bear & (c < ((po.shift(1) + pc.shift(1)) / 2))
    ).astype(int)
    doji_mid = body.shift(1) / rng.shift(1) < 0.1
    out["pat_abandoned_baby_bull"] = (
        (pc.shift(1) < po.shift(1))
        & doji_mid
        & (l.shift(1) < l.shift(2))
        & (l.shift(1) < l)
        & bull
        & (o > h.shift(1))
    ).astype(int)
    out["pat_abandoned_baby_bear"] = (
        (pc.shift(1) > po.shift(1))
        & doji_mid
        & (h.shift(1) > h.shift(2))
        & (h.shift(1) > h)
        & bear
        & (o < l.shift(1))
    ).astype(int)
    out["pat_three_soldiers"] = (
        bull & bull.shift(1) & bull.shift(2)
        & (c > c.shift(1)) & (c.shift(1) > c.shift(2))
        & (body > avg_body * 0.8) & (body.shift(1) > avg_body.shift(1) * 0.8)
    ).astype(int)
    out["pat_three_crows"] = (
        bear & bear.shift(1) & bear.shift(2)
        & (c < c.shift(1)) & (c.shift(1) < c.shift(2))
        & (body > avg_body * 0.8) & (body.shift(1) > avg_body.shift(1) * 0.8)
    ).astype(int)
    out["pat_three_inside_up"] = ((out["pat_bull_harami"].shift(1) == 1) & bull & (c > pc.shift(1))).astype(int)
    out["pat_three_inside_down"] = ((out["pat_bear_harami"].shift(1) == 1) & bear & (c < pc.shift(1))).astype(int)
    out["pat_three_outside_up"] = ((out["pat_bull_engulf"].shift(1) == 1) & bull & (c > c.shift(1))).astype(int)
    out["pat_three_outside_down"] = ((out["pat_bear_engulf"].shift(1) == 1) & bear & (c < c.shift(1))).astype(int)
    out["pat_rising_three"] = (
        (out["pat_marubozu_bull"].shift(4) == 1)
        & bear.shift(3) & bear.shift(2) & bear.shift(1)
        & (h.shift(3) < h.shift(4)) & (l.shift(1) > l.shift(4))
        & bull & (c > c.shift(4))
    ).astype(int)
    out["pat_falling_three"] = (
        (out["pat_marubozu_bear"].shift(4) == 1)
        & bull.shift(3) & bull.shift(2) & bull.shift(1)
        & (l.shift(3) > l.shift(4)) & (h.shift(1) < h.shift(4))
        & bear & (c < c.shift(4))
    ).astype(int)
    out["pat_stick_sandwich"] = (
        bear.shift(2) & bull.shift(1) & bear
        & ((c - c.shift(2)).abs() / c < 0.0008)
    ).astype(int)
    out["pat_unique_three_river"] = (
        (pc.shift(1) < po.shift(1)) & (body.shift(1) > avg_body.shift(1))
        & (lower > body) & (body_pct < 0.4)
        & bull & (body < avg_body * 0.6)
    ).astype(int)
    out["pat_two_crows"] = (
        (c.shift(2) > o.shift(2))
        & (o.shift(1) > h.shift(2))
        & (c.shift(1) < o.shift(1))
        & bear
        & (c < o.shift(1))
        & (c > o.shift(2))
    ).astype(int)
    out["pat_upside_gap_two_crows"] = (
        (c.shift(2) > o.shift(2))
        & (o.shift(1) > h.shift(2))
        & (c.shift(1) < o.shift(1))
        & bear
        & (o > c.shift(1))
        & (c < o.shift(1))
        & (c > o.shift(2))
    ).astype(int)
    out["pat_advance_block"] = (
        bull & bull.shift(1) & bull.shift(2)
        & (c > c.shift(1)) & (c.shift(1) > c.shift(2))
        & (body < body.shift(1)) & (body.shift(1) < body.shift(2))
        & (upper > upper.shift(1))
    ).astype(int)
    out["pat_deliberation"] = (
        bull.shift(2) & bull.shift(1) & bull
        & (body.shift(2) > avg_body.shift(2)) & (body.shift(1) > avg_body.shift(1))
        & (body < avg_body * 0.5) & (c >= c.shift(1))
    ).astype(int)
    out["pat_ladder_bottom"] = (
        bear.shift(4) & bear.shift(3) & bear.shift(2)
        & (out["pat_inverted_hammer"].shift(1) == 1)
        & bull
    ).astype(int)
    out["pat_tasuki_up"] = (
        bull.shift(2) & (o.shift(1) > h.shift(2)) & bull.shift(1)
        & bear & (o < c.shift(1)) & (c > o.shift(1))
    ).astype(int)
    out["pat_tasuki_down"] = (
        bear.shift(2) & (o.shift(1) < l.shift(2)) & bear.shift(1)
        & bull & (o > c.shift(1)) & (c < o.shift(1))
    ).astype(int)

    strength = (body / avg_body.replace(0, np.nan)).clip(0, 3).fillna(0)
    out["pat_strength"] = strength.round(3)

    bull_cols = [k for k, v in PATTERN_CATALOG.items() if v["bias"] == "bullish" and k in out.columns]
    bear_cols = [k for k, v in PATTERN_CATALOG.items() if v["bias"] == "bearish" and k in out.columns]
    out["pat_bias"] = (out[bull_cols].sum(axis=1) - out[bear_cols].sum(axis=1)).clip(-5, 5).astype(int)

    return out.fillna(0)


# ---------------------------------------------------------------------------
# Structural / chart patterns
# ---------------------------------------------------------------------------

def _causal_swings(
    high: pd.Series,
    low: pd.Series,
    left: int = 3,
    right: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(high)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    sh_idx = np.full(n, -1, dtype=int)
    sl_idx = np.full(n, -1, dtype=int)
    for i in range(left + right, n):
        center = i - right
        wh = high.iloc[center - left : center + right + 1]
        wl = low.iloc[center - left : center + right + 1]
        if high.iloc[center] >= wh.max():
            sh[i] = float(high.iloc[center])
            sh_idx[i] = center
        if low.iloc[center] <= wl.min():
            sl[i] = float(low.iloc[center])
            sl_idx[i] = center
    return sh, sl, sh_idx, sl_idx


def _ratio_near(value: float, target: float, tol: float = 0.12) -> bool:
    if target == 0 or np.isnan(value):
        return False
    return abs(value - target) / abs(target) <= tol


def _ratio_in(value: float, lo: float, hi: float) -> bool:
    return (not np.isnan(value)) and lo <= value <= hi


def _alternating_pivots(
    highs: list[tuple[int, float]],
    lows: list[tuple[int, float]],
    *,
    max_points: int = 5,
) -> list[tuple[int, float, str]]:
    """Merge swing highs/lows chronologically and keep an alternating HL sequence."""
    merged = [(i, p, "H") for i, p in highs] + [(i, p, "L") for i, p in lows]
    merged.sort(key=lambda x: x[0])
    alt: list[tuple[int, float, str]] = []
    for pt in merged:
        if not alt or alt[-1][2] != pt[2]:
            alt.append(pt)
        else:
            # Keep more extreme same-type pivot
            prev = alt[-1]
            if pt[2] == "H" and pt[1] >= prev[1]:
                alt[-1] = pt
            elif pt[2] == "L" and pt[1] <= prev[1]:
                alt[-1] = pt
    return alt[-max_points:]


def _mark_harmonic_patterns(
    out: dict[str, np.ndarray],
    i: int,
    pivots: list[tuple[int, float, str]],
    close_i: float,
) -> None:
    """XABCD harmonic approximations at completion of D (causal)."""
    if len(pivots) < 5:
        return
    x, a, b, c, d = pivots[-5:]
    xa = abs(a[1] - x[1])
    ab = abs(b[1] - a[1])
    bc = abs(c[1] - b[1])
    ad = abs(d[1] - a[1])
    if xa < 1e-12:
        return
    ab_xa = ab / xa
    bc_ab = bc / max(ab, 1e-12)
    ad_xa = ad / xa
    bullish_d = x[2] == "H" and a[2] == "L" and b[2] == "H" and c[2] == "L" and d[2] == "H"
    # Prefer bullish completion when last pivot is a low (buy zone D)
    bull_completion = x[2] == "L" and a[2] == "H" and b[2] == "L" and c[2] == "H" and d[2] == "L"
    bear_completion = x[2] == "H" and a[2] == "L" and b[2] == "H" and c[2] == "L" and d[2] == "H"

    def _fire(key: str) -> None:
        # Confirm only when price is near D (within 0.25%)
        if abs(close_i - d[1]) / max(abs(d[1]), 1e-9) < 0.0025:
            out[key][i] = 1

    # Gartley
    if _ratio_near(ab_xa, 0.618) and _ratio_in(bc_ab, 0.382, 0.886) and _ratio_near(ad_xa, 0.786):
        if bull_completion:
            _fire("pat_harmonic_gartley_bull")
        if bear_completion:
            _fire("pat_harmonic_gartley_bear")
    # Bat
    if _ratio_in(ab_xa, 0.382, 0.5) and _ratio_near(ad_xa, 0.886, 0.1):
        if bull_completion:
            _fire("pat_harmonic_bat_bull")
        if bear_completion:
            _fire("pat_harmonic_bat_bear")
    # Butterfly
    if _ratio_near(ab_xa, 0.786) and _ratio_near(ad_xa, 1.27, 0.12):
        if bull_completion:
            _fire("pat_harmonic_butterfly_bull")
        if bear_completion:
            _fire("pat_harmonic_butterfly_bear")
    # Crab
    if _ratio_in(ab_xa, 0.382, 0.618) and _ratio_near(ad_xa, 1.618, 0.12):
        if bull_completion:
            _fire("pat_harmonic_crab_bull")
        if bear_completion:
            _fire("pat_harmonic_crab_bear")
    _ = bullish_d  # reserved for future XA-direction filter


def _mark_elliott_wolfe(
    out: dict[str, np.ndarray],
    i: int,
    pivots: list[tuple[int, float, str]],
    close_i: float,
    high_i: float,
    low_i: float,
) -> None:
    """Simplified 5-wave Elliott impulse and Wolfe wave heuristics."""
    if len(pivots) < 5:
        return
    p = pivots[-5:]
    prices = [pt[1] for pt in p]
    # Bull impulse: L H L H L with rising structure and wave3 longest
    if p[0][2] == "L" and p[1][2] == "H" and p[2][2] == "L" and p[3][2] == "H" and p[4][2] == "L":
        w1 = prices[1] - prices[0]
        w2 = prices[1] - prices[2]
        w3 = prices[3] - prices[2]
        w4 = prices[3] - prices[4]
        if (
            w1 > 0
            and w3 > w1
            and w3 > w2
            and prices[4] > prices[0]
            and prices[2] > prices[0]
            and close_i > prices[3]
        ):
            out["pat_elliott_impulse_bull"][i] = 1
        # Wolfe bull: converging 1-3-5 / 2-4; point5 under-extension then reclaim
        if prices[4] < prices[2] < prices[0] and prices[3] < prices[1] and close_i > prices[4] and low_i <= prices[4]:
            out["pat_wolfe_bull"][i] = 1
    # Bear impulse: H L H L H
    if p[0][2] == "H" and p[1][2] == "L" and p[2][2] == "H" and p[3][2] == "L" and p[4][2] == "H":
        w1 = prices[0] - prices[1]
        w2 = prices[2] - prices[1]
        w3 = prices[2] - prices[3]
        w4 = prices[4] - prices[3]
        if (
            w1 > 0
            and w3 > w1
            and w3 > w2
            and prices[4] < prices[0]
            and prices[2] < prices[0]
            and close_i < prices[3]
        ):
            out["pat_elliott_impulse_bear"][i] = 1
        if prices[4] > prices[2] > prices[0] and prices[3] > prices[1] and close_i < prices[4] and high_i >= prices[4]:
            out["pat_wolfe_bear"][i] = 1


def swing_support_resistance(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    sh, sl, _, _ = _causal_swings(high, low, left, right)
    resist_s = pd.Series(sh, index=df.index).ffill()
    support_s = pd.Series(sl, index=df.index).ffill()
    out = pd.DataFrame(index=df.index)
    out["resist_level"] = resist_s
    out["support_level"] = support_s
    out["dist_to_resist"] = (resist_s - close) / close
    out["dist_to_support"] = (close - support_s) / close
    out["swing_high"] = pd.Series(sh, index=df.index)
    out["swing_low"] = pd.Series(sl, index=df.index)
    out["near_support"] = (out["dist_to_support"].abs() < 0.0015).astype(int)
    out["near_resist"] = (out["dist_to_resist"].abs() < 0.0015).astype(int)
    return out


def structural_patterns(
    df: pd.DataFrame,
    left: int = 3,
    right: int = 3,
    score_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    n = len(df)
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    sh, sl, sh_i, sl_i = _causal_swings(df["high"], df["low"], left, right)

    keys = [
        "pat_double_top", "pat_double_bottom", "pat_triple_top", "pat_triple_bottom",
        "pat_head_shoulders", "pat_inv_head_shoulders",
        "pat_triangle_asc", "pat_triangle_desc", "pat_triangle_sym",
        "pat_wedge_rising", "pat_wedge_falling",
        "pat_flag_bull", "pat_flag_bear",
        "pat_channel_up", "pat_channel_down",
        "pat_breakout_up", "pat_breakout_down",
        "pat_bos_up", "pat_bos_down",
        "pat_choch_bull", "pat_choch_bear",
        "pat_equal_highs", "pat_equal_lows",
        "pat_liquidity_sweep_high", "pat_liquidity_sweep_low",
        "pat_rounding_bottom", "pat_rounding_top",
        "pat_cup_handle",
        "pat_pennant_bull", "pat_pennant_bear",
        "pat_rectangle_bull", "pat_rectangle_bear",
        "pat_broadening_up", "pat_broadening_down",
        "pat_compression", "pat_compression_breakout_up", "pat_compression_breakout_down",
        "pat_harmonic_gartley_bull", "pat_harmonic_gartley_bear",
        "pat_harmonic_bat_bull", "pat_harmonic_bat_bear",
        "pat_harmonic_butterfly_bull", "pat_harmonic_butterfly_bear",
        "pat_harmonic_crab_bull", "pat_harmonic_crab_bear",
        "pat_elliott_impulse_bull", "pat_elliott_impulse_bear",
        "pat_wolfe_bull", "pat_wolfe_bear",
        "structure_hh_hl", "trendline_slope", "chart_pattern_score",
    ]
    out = {k: np.zeros(n, dtype=float) for k in keys}
    # Precompute ATR proxies for compression (causal rolling)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr5 = pd.Series(tr).rolling(5, min_periods=3).mean().to_numpy()
    atr20 = pd.Series(tr).rolling(20, min_periods=8).mean().to_numpy()

    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    prev_structure = 0.0

    for i in range(n):
        if not np.isnan(sh[i]) and sh_i[i] >= 0:
            highs.append((int(sh_i[i]), float(sh[i])))
            if len(highs) > 12:
                highs = highs[-12:]
        if not np.isnan(sl[i]) and sl_i[i] >= 0:
            lows.append((int(sl_i[i]), float(sl[i])))
            if len(lows) > 12:
                lows = lows[-12:]

        if len(highs) >= 2 and len(lows) >= 2:
            hh = highs[-1][1] > highs[-2][1]
            hl = lows[-1][1] > lows[-2][1]
            lh = highs[-1][1] < highs[-2][1]
            ll = lows[-1][1] < lows[-2][1]
            if hh and hl:
                out["structure_hh_hl"][i] = 1.0
            elif lh and ll:
                out["structure_hh_hl"][i] = -1.0

        if len(highs) >= 2 and len(lows) >= 1:
            h1, h2 = highs[-2], highs[-1]
            if abs(h1[1] - h2[1]) / max(h2[1], 1e-9) < 0.0025 and h2[0] > h1[0]:
                mid_lows = [p for idx, p in lows if h1[0] < idx < h2[0]]
                if mid_lows and close[i] < min(mid_lows):
                    out["pat_double_top"][i] = 1
                if abs(h1[1] - h2[1]) / max(h2[1], 1e-9) < 0.0015:
                    out["pat_equal_highs"][i] = 1

        if len(lows) >= 2 and len(highs) >= 1:
            l1, l2 = lows[-2], lows[-1]
            if abs(l1[1] - l2[1]) / max(l2[1], 1e-9) < 0.0025 and l2[0] > l1[0]:
                mid_highs = [p for idx, p in highs if l1[0] < idx < l2[0]]
                if mid_highs and close[i] > max(mid_highs):
                    out["pat_double_bottom"][i] = 1
                if abs(l1[1] - l2[1]) / max(l2[1], 1e-9) < 0.0015:
                    out["pat_equal_lows"][i] = 1

        if len(highs) >= 3 and len(lows) >= 1:
            a, b, c3 = highs[-3], highs[-2], highs[-1]
            if (
                abs(a[1] - b[1]) / max(b[1], 1e-9) < 0.003
                and abs(b[1] - c3[1]) / max(c3[1], 1e-9) < 0.003
            ):
                troughs = [p for idx, p in lows if a[0] < idx < c3[0]]
                if troughs and close[i] < min(troughs):
                    out["pat_triple_top"][i] = 1

        if len(lows) >= 3 and len(highs) >= 1:
            a, b, c3 = lows[-3], lows[-2], lows[-1]
            if (
                abs(a[1] - b[1]) / max(b[1], 1e-9) < 0.003
                and abs(b[1] - c3[1]) / max(c3[1], 1e-9) < 0.003
            ):
                peaks = [p for idx, p in highs if a[0] < idx < c3[0]]
                if peaks and close[i] > max(peaks):
                    out["pat_triple_bottom"][i] = 1

        if len(highs) >= 3 and len(lows) >= 2:
            s1, head, s2 = highs[-3], highs[-2], highs[-1]
            if head[1] > s1[1] and head[1] > s2[1] and abs(s1[1] - s2[1]) / head[1] < 0.01 and s1[0] < head[0] < s2[0]:
                neck = [p for idx, p in lows if s1[0] < idx < s2[0]]
                if neck and close[i] < min(neck):
                    out["pat_head_shoulders"][i] = 1

        if len(lows) >= 3 and len(highs) >= 2:
            s1, head, s2 = lows[-3], lows[-2], lows[-1]
            if head[1] < s1[1] and head[1] < s2[1] and abs(s1[1] - s2[1]) / max(abs(head[1]), 1e-9) < 0.01 and s1[0] < head[0] < s2[0]:
                neck = [p for idx, p in highs if s1[0] < idx < s2[0]]
                if neck and close[i] > max(neck):
                    out["pat_inv_head_shoulders"][i] = 1

        if len(highs) >= 3 and len(lows) >= 3:
            hx = np.array([h[0] for h in highs[-3:]], dtype=float)
            hy = np.array([h[1] for h in highs[-3:]], dtype=float)
            lx = np.array([v[0] for v in lows[-3:]], dtype=float)
            ly = np.array([v[1] for v in lows[-3:]], dtype=float)
            hs = np.polyfit(hx, hy, 1)[0] if hx[-1] != hx[0] else 0.0
            ls = np.polyfit(lx, ly, 1)[0] if lx[-1] != lx[0] else 0.0
            out["trendline_slope"][i] = float((hs + ls) / 2)
            if abs(hs) < abs(hy.mean()) * 1e-5 and ls > 0:
                out["pat_triangle_asc"][i] = 1
            elif abs(ls) < abs(ly.mean()) * 1e-5 and hs < 0:
                out["pat_triangle_desc"][i] = 1
            elif hs < 0 and ls > 0:
                out["pat_triangle_sym"][i] = 1
            if hs > 0 and ls > 0 and hs < ls * 0.7:
                out["pat_wedge_rising"][i] = 1
            if hs < 0 and ls < 0 and abs(ls) < abs(hs) * 0.7:
                out["pat_wedge_falling"][i] = 1
            if abs(hs - ls) / max(abs(hy.mean()), 1e-9) < 1e-5 and hs > 0:
                out["pat_channel_up"][i] = 1
            if abs(hs - ls) / max(abs(hy.mean()), 1e-9) < 1e-5 and hs < 0:
                out["pat_channel_down"][i] = 1

        if i >= 20:
            impulse = close[i - 15] / close[i - 20] - 1.0
            cons = np.std(close[i - 8 : i + 1]) / max(close[i], 1e-9)
            if impulse > 0.008 and cons < 0.002:
                out["pat_flag_bull"][i] = 1
            if impulse < -0.008 and cons < 0.002:
                out["pat_flag_bear"][i] = 1
            recent_high = np.max(high[i - 20 : i])
            recent_low = np.min(low[i - 20 : i])
            if close[i] > recent_high:
                out["pat_breakout_up"][i] = 1
            if close[i] < recent_low:
                out["pat_breakout_down"][i] = 1
            # liquidity sweeps
            if high[i] > recent_high and close[i] < recent_high:
                out["pat_liquidity_sweep_high"][i] = 1
            if low[i] < recent_low and close[i] > recent_low:
                out["pat_liquidity_sweep_low"][i] = 1

        if highs and close[i] > highs[-1][1]:
            out["pat_bos_up"][i] = 1
        if lows and close[i] < lows[-1][1]:
            out["pat_bos_down"][i] = 1
        if prev_structure < 0 and out["pat_bos_up"][i] == 1:
            out["pat_choch_bull"][i] = 1
        if prev_structure > 0 and out["pat_bos_down"][i] == 1:
            out["pat_choch_bear"][i] = 1
        if out["structure_hh_hl"][i] != 0:
            prev_structure = out["structure_hh_hl"][i]

        # Rounding bottom/top via quadratic fit on last 8 closes
        if i >= 30:
            y = close[i - 29 : i + 1]
            x = np.arange(len(y), dtype=float)
            coef = np.polyfit(x, y, 2)
            if coef[0] > 0 and close[i] > close[i - 15]:
                out["pat_rounding_bottom"][i] = 1
            if coef[0] < 0 and close[i] < close[i - 15]:
                out["pat_rounding_top"][i] = 1

        # Cup & Handle: deep U recovery + shallow handle + breakout
        if i >= 40 and len(lows) >= 2 and len(highs) >= 2:
            cup = close[i - 39 : i - 9]
            cup_x = np.arange(len(cup), dtype=float)
            cup_coef = np.polyfit(cup_x, cup, 2)
            left_rim = float(np.max(close[i - 39 : i - 30]))
            right_rim = float(np.max(close[i - 20 : i - 10]))
            cup_low = float(np.min(cup))
            depth = (left_rim + right_rim) / 2.0 - cup_low
            handle = close[i - 9 : i + 1]
            handle_high = float(np.max(high[i - 9 : i + 1]))
            handle_low = float(np.min(low[i - 9 : i + 1]))
            handle_depth = handle_high - handle_low
            if (
                cup_coef[0] > 0
                and depth > 0
                and handle_depth < 0.45 * depth
                and abs(left_rim - right_rim) / max(right_rim, 1e-9) < 0.02
                and close[i] > handle_high
            ):
                out["pat_cup_handle"][i] = 1

        # Pennants: impulse + converging consolidation (tighter than flag)
        if i >= 25 and len(highs) >= 3 and len(lows) >= 3:
            impulse_up = close[i - 18] / max(close[i - 24], 1e-9) - 1.0
            impulse_dn = close[i - 18] / max(close[i - 24], 1e-9) - 1.0
            hx = np.array([h[0] for h in highs[-3:]], dtype=float)
            hy = np.array([h[1] for h in highs[-3:]], dtype=float)
            lx = np.array([v[0] for v in lows[-3:]], dtype=float)
            ly = np.array([v[1] for v in lows[-3:]], dtype=float)
            hs = np.polyfit(hx, hy, 1)[0] if hx[-1] != hx[0] else 0.0
            ls = np.polyfit(lx, ly, 1)[0] if lx[-1] != lx[0] else 0.0
            converging = hs < 0 and ls > 0
            cons_w = (float(np.max(high[i - 8 : i + 1])) - float(np.min(low[i - 8 : i + 1]))) / max(close[i], 1e-9)
            if impulse_up > 0.01 and converging and cons_w < 0.006 and close[i] > float(np.max(high[i - 8 : i])):
                out["pat_pennant_bull"][i] = 1
            if impulse_dn < -0.01 and converging and cons_w < 0.006 and close[i] < float(np.min(low[i - 8 : i])):
                out["pat_pennant_bear"][i] = 1

        # Rectangles: flat range then breakout
        if i >= 24:
            win_h = high[i - 23 : i]
            win_l = low[i - 23 : i]
            rh = float(np.max(win_h))
            rl = float(np.min(win_l))
            top_hits = int(np.sum(win_h >= rh * 0.9985))
            bot_hits = int(np.sum(win_l <= rl * 1.0015))
            width = (rh - rl) / max(close[i], 1e-9)
            if top_hits >= 3 and bot_hits >= 3 and 0.002 < width < 0.025:
                if close[i] > rh:
                    out["pat_rectangle_bull"][i] = 1
                if close[i] < rl:
                    out["pat_rectangle_bear"][i] = 1

        # Broadening (megaphone): diverging rails
        if len(highs) >= 3 and len(lows) >= 3:
            hx = np.array([h[0] for h in highs[-3:]], dtype=float)
            hy = np.array([h[1] for h in highs[-3:]], dtype=float)
            lx = np.array([v[0] for v in lows[-3:]], dtype=float)
            ly = np.array([v[1] for v in lows[-3:]], dtype=float)
            hs = np.polyfit(hx, hy, 1)[0] if hx[-1] != hx[0] else 0.0
            ls = np.polyfit(lx, ly, 1)[0] if lx[-1] != lx[0] else 0.0
            if hs > 0 and ls < 0:
                if close[i] > highs[-1][1]:
                    out["pat_broadening_up"][i] = 1
                if close[i] < lows[-1][1]:
                    out["pat_broadening_down"][i] = 1

        # Compression + breakout
        if i >= 20 and atr20[i] and not np.isnan(atr20[i]) and atr20[i] > 0:
            compressing = atr5[i] / atr20[i] < 0.55
            if compressing:
                out["pat_compression"][i] = 1
                recent_high = float(np.max(high[i - 20 : i]))
                recent_low = float(np.min(low[i - 20 : i]))
                if close[i] > recent_high:
                    out["pat_compression_breakout_up"][i] = 1
                if close[i] < recent_low:
                    out["pat_compression_breakout_down"][i] = 1

        # Harmonics / Elliott / Wolfe from last 5 alternating pivots
        pivots = _alternating_pivots(highs, lows, max_points=5)
        if len(pivots) >= 5:
            _mark_harmonic_patterns(out, i, pivots, close[i])
            _mark_elliott_wolfe(out, i, pivots, close[i], high[i], low[i])

        score = _chart_pattern_score_at(out, i, score_weights)
        out["chart_pattern_score"][i] = float(score)

    return pd.DataFrame(out, index=df.index)


# Bullish / bearish contributors to chart_pattern_score (includes all harmonics)
_CHART_SCORE_BULL = (
    "pat_double_bottom",
    "pat_triple_bottom",
    "pat_inv_head_shoulders",
    "pat_flag_bull",
    "pat_breakout_up",
    "pat_bos_up",
    "pat_choch_bull",
    "pat_wedge_falling",
    "pat_liquidity_sweep_low",
    "pat_triangle_asc",
    "pat_cup_handle",
    "pat_pennant_bull",
    "pat_rectangle_bull",
    "pat_compression_breakout_up",
    "pat_harmonic_gartley_bull",
    "pat_harmonic_bat_bull",
    "pat_harmonic_butterfly_bull",
    "pat_harmonic_crab_bull",
    "pat_elliott_impulse_bull",
    "pat_wolfe_bull",
)
_CHART_SCORE_BEAR = (
    "pat_double_top",
    "pat_triple_top",
    "pat_head_shoulders",
    "pat_flag_bear",
    "pat_breakout_down",
    "pat_bos_down",
    "pat_choch_bear",
    "pat_wedge_rising",
    "pat_liquidity_sweep_high",
    "pat_triangle_desc",
    "pat_pennant_bear",
    "pat_rectangle_bear",
    "pat_compression_breakout_down",
    "pat_harmonic_gartley_bear",
    "pat_harmonic_bat_bear",
    "pat_harmonic_butterfly_bear",
    "pat_harmonic_crab_bear",
    "pat_elliott_impulse_bear",
    "pat_wolfe_bear",
)


def chart_score_weights_from_rankings(
    ranking_items: list[dict[str, Any]] | None,
) -> dict[str, float]:
    """Derive per-pattern score weights from rankings quality/success (safe fallback 1.0)."""
    weights: dict[str, float] = {}
    keys = set(_CHART_SCORE_BULL) | set(_CHART_SCORE_BEAR)
    if not ranking_items:
        return {k: 1.0 for k in keys}
    by_key = {
        str(r.get("pattern_key") or r.get("key") or ""): r
        for r in ranking_items
        if r.get("pattern_key") or r.get("key")
    }
    for k in keys:
        row = by_key.get(k)
        if not row:
            weights[k] = 1.0
            continue
        q = row.get("quality_score")
        sr = row.get("success_rate")
        try:
            qf = float(q) if q is not None else 0.5
        except (TypeError, ValueError):
            qf = 0.5
        try:
            srf = float(sr) if sr is not None else 0.5
        except (TypeError, ValueError):
            srf = 0.5
        # Map ~[0.4, 0.8] quality/success into ~[0.5, 1.5] weight
        w = 0.5 + max(0.0, min(1.0, qf)) + max(0.0, min(1.0, srf - 0.5))
        weights[k] = float(max(0.25, min(2.0, w)))
    return weights


def _chart_pattern_score_at(
    out: dict[str, np.ndarray],
    i: int,
    score_weights: dict[str, float] | None,
) -> float:
    w = score_weights or {}
    score = 0.0
    for key in _CHART_SCORE_BULL:
        score += float(out[key][i]) * float(w.get(key, 1.0))
    for key in _CHART_SCORE_BEAR:
        score -= float(out[key][i]) * float(w.get(key, 1.0))
    score += float(out["structure_hh_hl"][i])
    return score


def compound_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Overlapping / composite patterns from co-firing base signals."""
    out = pd.DataFrame(index=df.index)
    for key, _name, a, b in COMPOUND_TEMPLATES:
        if a in df.columns and b in df.columns:
            out[key] = ((df[a] == 1) & (df[b] == 1)).astype(int)
        else:
            out[key] = 0
    return out


def discover_rare_compounds(
    df: pd.DataFrame,
    pattern_cols: list[str],
    *,
    min_count: int = 5,
    max_new: int = 40,
    max_order: int = 3,
    candidate_cap: int = 18,
) -> list[dict[str, Any]]:
    """
    Data-driven discovery of co-occurring pattern compounds (same bar).
    Supports pair and triple discoveries, sorted by order / lift / count.
    """
    cols = [c for c in pattern_cols if c in df.columns]
    if len(cols) < 2:
        return []
    mat = df[cols].fillna(0).astype(int)
    totals = mat.sum()
    n = len(mat)
    viable_cols = [c for c in cols if int(totals[c]) >= min_count]
    viable_cols.sort(key=lambda c: int(totals[c]), reverse=True)
    viable_cols = viable_cols[: max(2, candidate_cap)]
    candidates: list[dict[str, Any]] = []
    for order in range(2, max(2, max_order) + 1):
        if len(viable_cols) < order:
            break
        min_lift = 1.5 if order == 2 else 2.0
        prefix = "disc" if order == 2 else f"disc{order}"
        for legs in combinations(viable_cols, order):
            both = int((mat[list(legs)].sum(axis=1) == order).sum())
            if both < min_count:
                continue
            expected = 1.0
            for leg in legs:
                expected *= float(totals[leg]) / max(n, 1)
            p_joint = both / max(n, 1)
            lift = p_joint / max(expected, 1e-12)
            if lift < min_lift:
                continue
            pretty = [
                leg.replace("pat_", "").replace("cmp_", "").replace("disc_", "")
                for leg in legs
            ]
            candidates.append(
                {
                    "key": f"{prefix}_{'__'.join(legs)}",
                    "name": " + ".join(pretty),
                    "legs": list(legs),
                    "occurrences": both,
                    "lift": float(lift),
                    "category": "compound",
                    "conditions": " AND ".join(f"{leg}=1" for leg in legs) + f" (lift={lift:.2f})",
                }
            )
    candidates.sort(
        key=lambda x: (len(x["legs"]), x["lift"], x["occurrences"]),
        reverse=True,
    )
    return candidates[:max_new]


def pattern_summary_row(row: pd.Series, labels: dict[str, str]) -> dict[str, Any]:
    active = []
    for col, name in labels.items():
        if col in row.index and float(row[col]) == 1.0:
            active.append(name)
    bias = int(row["pat_bias"]) if "pat_bias" in row.index and pd.notna(row["pat_bias"]) else 0
    strength = float(row["pat_strength"]) if "pat_strength" in row.index and pd.notna(row["pat_strength"]) else 0.0
    chart = float(row["chart_pattern_score"]) if "chart_pattern_score" in row.index and pd.notna(row["chart_pattern_score"]) else 0.0
    structure = float(row["structure_hh_hl"]) if "structure_hh_hl" in row.index and pd.notna(row["structure_hh_hl"]) else 0.0
    return {
        "active": active,
        "bias": bias,
        "strength": strength,
        "chart_score": chart,
        "structure": structure,
    }
