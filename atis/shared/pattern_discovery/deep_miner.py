"""Deep Pattern Mining — discover novel NewN motifs beyond same-bar compounds.

All detectors are causal: features at bar i use only bars <= i.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _body_wick_codes(df: pd.DataFrame) -> np.ndarray:
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-12)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    bull = c > o
    codes = np.full(len(df), "N", dtype=object)
    body_pct = body / rng
    codes[(body_pct < 0.12) & (upper / rng > 0.35) & (lower / rng > 0.35)] = "D"  # doji-like
    codes[bull & (body_pct >= 0.55) & (lower / rng < 0.2)] = "B"  # strong bull
    codes[(~bull) & (body_pct >= 0.55) & (upper / rng < 0.2)] = "S"  # strong bear
    codes[bull & (lower / rng > 2.0 * np.maximum(body_pct, 0.05)) & (upper / rng < 0.25)] = "H"  # hammer-ish
    codes[(~bull) & (upper / rng > 2.0 * np.maximum(body_pct, 0.05)) & (lower / rng < 0.25)] = "T"  # shooting
    codes[(body_pct >= 0.12) & (body_pct < 0.55) & bull] = "b"
    codes[(body_pct >= 0.12) & (body_pct < 0.55) & (~bull)] = "s"
    return codes


def _sequence_mask(codes: np.ndarray, motif: str) -> np.ndarray:
    n = len(codes)
    m = len(motif)
    out = np.zeros(n, dtype=np.int8)
    if m == 0 or n < m:
        return out
    for i in range(m - 1, n):
        ok = True
        for j, ch in enumerate(motif):
            if codes[i - m + 1 + j] != ch:
                ok = False
                break
        if ok:
            out[i] = 1
    return out


def _accel_features(close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return velocity, acceleration, deceleration flags (causal)."""
    n = len(close)
    vel = np.zeros(n)
    acc = np.zeros(n)
    if n > 1:
        vel[1:] = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0
    if n > 2:
        acc[2:] = vel[2:] - vel[1:-1]
    return vel, acc, (-acc)


def discover_deep_patterns(
    df: pd.DataFrame,
    *,
    max_new: int = 40,
    min_count: int = 8,
    existing_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Mine sequential / momentum / liquidity motifs and name them New1, New2, …

    Returns list of dicts with binary series under ``signal`` plus metadata.
    """
    existing_keys = existing_keys or set()
    n = len(df)
    if n < 30:
        return []

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    codes = _body_wick_codes(df)
    vel, acc, dec = _accel_features(close)

    candidates: list[dict[str, Any]] = []

    # --- sequence motifs (variable windows encoded as fixed motifs) ---
    motifs = [
        ("HB", "Hammer then strong bull", "bullish", "codes[-2:]=H,B"),
        ("TS", "Shooting then strong bear", "bearish", "codes[-2:]=T,S"),
        ("DDB", "Double doji then bull", "bullish", "codes[-3:]=D,D,B"),
        ("DDS", "Double doji then bear", "bearish", "codes[-3:]=D,D,S"),
        ("bbB", "Two weak bulls then thrust", "bullish", "codes[-3:]=b,b,B"),
        ("ssS", "Two weak bears then thrust", "bearish", "codes[-3:]=s,s,S"),
        ("BsB", "Bull-bear-bull continuation", "bullish", "codes[-3:]=B,s,B"),
        ("SbS", "Bear-bull-bear continuation", "bearish", "codes[-3:]=S,b,S"),
        ("HDB", "Hammer-doji-bull reversal", "bullish", "codes[-3:]=H,D,B"),
        ("TDS", "Shooting-doji-bear reversal", "bearish", "codes[-3:]=T,D,S"),
    ]
    for motif, desc, bias, rule in motifs:
        sig = _sequence_mask(codes, motif)
        count = int(sig.sum())
        if count >= min_count:
            candidates.append(
                {
                    "description": desc,
                    "bias": bias,
                    "mathematical_rules": f"candle_code_sequence=='{motif}'",
                    "logical_rules": rule,
                    "appearance_conditions": f"count>={min_count} causal window={len(motif)}",
                    "signal": sig,
                    "category": "discovered",
                    "kind": "sequence",
                }
            )

    # --- velocity / acceleration / deceleration ---
    # Bullish acceleration after slowdown
    sig = np.zeros(n, dtype=np.int8)
    for i in range(5, n):
        if vel[i] > 0 and acc[i] > 0 and vel[i - 1] <= vel[i - 2] and close[i] > close[i - 3]:
            sig[i] = 1
    if int(sig.sum()) >= min_count:
        candidates.append(
            {
                "description": "Momentum re-acceleration up",
                "bias": "bullish",
                "mathematical_rules": "vel_i>0 & acc_i>0 & vel_{i-1}<=vel_{i-2}",
                "logical_rules": "slowdown then bullish acceleration",
                "appearance_conditions": "causal 5-bar lookback",
                "signal": sig.copy(),
                "category": "discovered",
                "kind": "momentum",
            }
        )

    sig = np.zeros(n, dtype=np.int8)
    for i in range(5, n):
        if vel[i] < 0 and acc[i] < 0 and vel[i - 1] >= vel[i - 2] and close[i] < close[i - 3]:
            sig[i] = 1
    if int(sig.sum()) >= min_count:
        candidates.append(
            {
                "description": "Momentum re-acceleration down",
                "bias": "bearish",
                "mathematical_rules": "vel_i<0 & acc_i<0 & vel_{i-1}>=vel_{i-2}",
                "logical_rules": "slowdown then bearish acceleration",
                "appearance_conditions": "causal 5-bar lookback",
                "signal": sig.copy(),
                "category": "discovered",
                "kind": "momentum",
            }
        )

    # --- false breakout / liquidity flip ---
    sig = np.zeros(n, dtype=np.int8)
    for i in range(20, n):
        prior_high = float(np.max(high[i - 20 : i]))
        if high[i] > prior_high and close[i] < prior_high and close[i] < close[i - 1]:
            sig[i] = 1
    if int(sig.sum()) >= min_count:
        candidates.append(
            {
                "description": "False breakout high (liquidity grab)",
                "bias": "bearish",
                "mathematical_rules": "high_i>max(high[i-20:i]) & close_i<that_high",
                "logical_rules": "pierce then reject",
                "appearance_conditions": "20-bar causal window",
                "signal": sig.copy(),
                "category": "discovered",
                "kind": "liquidity",
            }
        )

    sig = np.zeros(n, dtype=np.int8)
    for i in range(20, n):
        prior_low = float(np.min(low[i - 20 : i]))
        if low[i] < prior_low and close[i] > prior_low and close[i] > close[i - 1]:
            sig[i] = 1
    if int(sig.sum()) >= min_count:
        candidates.append(
            {
                "description": "False breakout low (liquidity grab)",
                "bias": "bullish",
                "mathematical_rules": "low_i<min(low[i-20:i]) & close_i>that_low",
                "logical_rules": "pierce then reclaim",
                "appearance_conditions": "20-bar causal window",
                "signal": sig.copy(),
                "category": "discovered",
                "kind": "liquidity",
            }
        )

    # --- volatility compression → expansion ---
    rng = np.maximum(high - low, 1e-12)
    avg20 = pd.Series(rng).rolling(20, min_periods=8).mean().to_numpy()
    sig = np.zeros(n, dtype=np.int8)
    for i in range(25, n):
        if avg20[i - 1] > 0 and rng[i - 1] < 0.55 * avg20[i - 1] and rng[i] > 1.4 * avg20[i - 1] and close[i] > close[i - 1]:
            sig[i] = 1
    if int(sig.sum()) >= min_count:
        candidates.append(
            {
                "description": "Compression then expansion up",
                "bias": "bullish",
                "mathematical_rules": "rng_{i-1}<0.55*ATR20 & rng_i>1.4*ATR20 & close up",
                "logical_rules": "squeeze then bullish expansion",
                "appearance_conditions": "causal ATR20",
                "signal": sig.copy(),
                "category": "discovered",
                "kind": "volatility",
            }
        )

    sig = np.zeros(n, dtype=np.int8)
    for i in range(25, n):
        if avg20[i - 1] > 0 and rng[i - 1] < 0.55 * avg20[i - 1] and rng[i] > 1.4 * avg20[i - 1] and close[i] < close[i - 1]:
            sig[i] = 1
    if int(sig.sum()) >= min_count:
        candidates.append(
            {
                "description": "Compression then expansion down",
                "bias": "bearish",
                "mathematical_rules": "rng_{i-1}<0.55*ATR20 & rng_i>1.4*ATR20 & close down",
                "logical_rules": "squeeze then bearish expansion",
                "appearance_conditions": "causal ATR20",
                "signal": sig.copy(),
                "category": "discovered",
                "kind": "volatility",
            }
        )

    # --- lagged compound: A then B within W bars (not same bar) ---
    if "pat_hammer" in df.columns and "pat_bos_up" in df.columns:
        a = df["pat_hammer"].fillna(0).astype(int).to_numpy()
        b = df["pat_bos_up"].fillna(0).astype(int).to_numpy()
        sig = np.zeros(n, dtype=np.int8)
        last_a = -999
        for i in range(n):
            if a[i] == 1:
                last_a = i
            if b[i] == 1 and 0 < i - last_a <= 5:
                sig[i] = 1
        if int(sig.sum()) >= min_count:
            candidates.append(
                {
                    "description": "Hammer then BOS-up within 5 bars",
                    "bias": "bullish",
                    "mathematical_rules": "exists j in [i-5,i): hammer_j=1 & bos_up_i=1",
                    "logical_rules": "sequential compound hammer→BOS",
                    "appearance_conditions": "lag window=5",
                    "signal": sig.copy(),
                    "category": "discovered",
                    "kind": "sequential_compound",
                }
            )

    if "pat_shooting_star" in df.columns and "pat_bos_down" in df.columns:
        a = df["pat_shooting_star"].fillna(0).astype(int).to_numpy()
        b = df["pat_bos_down"].fillna(0).astype(int).to_numpy()
        sig = np.zeros(n, dtype=np.int8)
        last_a = -999
        for i in range(n):
            if a[i] == 1:
                last_a = i
            if b[i] == 1 and 0 < i - last_a <= 5:
                sig[i] = 1
        if int(sig.sum()) >= min_count:
            candidates.append(
                {
                    "description": "Shooting star then BOS-down within 5 bars",
                    "bias": "bearish",
                    "mathematical_rules": "exists j in [i-5,i): shooting_j=1 & bos_down_i=1",
                    "logical_rules": "sequential compound shooting→BOS",
                    "appearance_conditions": "lag window=5",
                    "signal": sig.copy(),
                    "category": "discovered",
                    "kind": "sequential_compound",
                }
            )

    # --- hierarchical conditional: compression then liquidity grab ---
    if "pat_compression" in df.columns:
        comp = df["pat_compression"].fillna(0).astype(int).to_numpy()
        sig = np.zeros(n, dtype=np.int8)
        for i in range(25, n):
            if comp[i - 1] == 1 and high[i] > float(np.max(high[i - 20 : i])) and close[i] < close[i - 1]:
                sig[i] = 1
        if int(sig.sum()) >= min_count:
            candidates.append(
                {
                    "description": "Compression then false breakout high",
                    "bias": "bearish",
                    "mathematical_rules": "compression_{i-1}=1 & pierce_high_i & close down",
                    "logical_rules": "hierarchical squeeze→trap",
                    "appearance_conditions": "causal compression+20",
                    "signal": sig.copy(),
                    "category": "discovered",
                    "kind": "hierarchical",
                }
            )
        sig = np.zeros(n, dtype=np.int8)
        for i in range(25, n):
            if comp[i - 1] == 1 and low[i] < float(np.min(low[i - 20 : i])) and close[i] > close[i - 1]:
                sig[i] = 1
        if int(sig.sum()) >= min_count:
            candidates.append(
                {
                    "description": "Compression then false breakout low",
                    "bias": "bullish",
                    "mathematical_rules": "compression_{i-1}=1 & pierce_low_i & close up",
                    "logical_rules": "hierarchical squeeze→reclaim",
                    "appearance_conditions": "causal compression+20",
                    "signal": sig.copy(),
                    "category": "discovered",
                    "kind": "hierarchical",
                }
            )

    # --- repeating structure: 3 similar range contractions ---
    sig = np.zeros(n, dtype=np.int8)
    for i in range(30, n):
        r1 = float(np.mean(rng[i - 9 : i - 6]))
        r2 = float(np.mean(rng[i - 6 : i - 3]))
        r3 = float(np.mean(rng[i - 3 : i + 1]))
        if r1 > 0 and r2 < 0.85 * r1 and r3 < 0.85 * r2 and close[i] > close[i - 3]:
            sig[i] = 1
    if int(sig.sum()) >= min_count:
        candidates.append(
            {
                "description": "Triple range contraction then up",
                "bias": "bullish",
                "mathematical_rules": "mean_rng blocks decreasing x3 then close up",
                "logical_rules": "repeating compression motif",
                "appearance_conditions": "3x3 bar blocks",
                "signal": sig.copy(),
                "category": "discovered",
                "kind": "repetition",
            }
        )

    # Rank by rarity * clarity (lower frequency preferred among frequent-enough)
    scored: list[tuple[float, dict[str, Any]]] = []
    for c in candidates:
        cnt = int(c["signal"].sum())
        rarity = 1.0 - min(1.0, cnt / max(n * 0.05, 1))
        scored.append((rarity * np.log1p(cnt), c))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Assign NewN names skipping already-used numbers
    used_nums: set[int] = set()
    for k in existing_keys:
        if k.startswith("New") and k[3:].isdigit():
            used_nums.add(int(k[3:]))
    next_num = 1
    results: list[dict[str, Any]] = []
    for _, c in scored[:max_new]:
        while next_num in used_nums:
            next_num += 1
        name = f"New{next_num}"
        used_nums.add(next_num)
        next_num += 1
        desc = str(c.get("description") or "").strip()
        display_name = f"{name} — {desc}" if desc else name
        results.append(
            {
                "id": name,
                "key": name,
                "name": display_name,
                "description": c["description"],
                "mathematical_rules": c["mathematical_rules"],
                "logical_rules": c["logical_rules"],
                "appearance_conditions": c["appearance_conditions"],
                "bias": c["bias"],
                "category": "discovered",
                "kind": c["kind"],
                "legs": [],
                "signal": c["signal"],
                "occurrences": int(c["signal"].sum()),
                "conditions": c["mathematical_rules"],
            }
        )
    return results


def attach_signals_to_frame(df: pd.DataFrame, discoveries: list[dict[str, Any]]) -> pd.DataFrame:
    """Add NewN columns to a copy of df (single concat to avoid fragmentation)."""
    cols: dict[str, np.ndarray] = {}
    for item in discoveries:
        key = item["key"]
        sig = item.get("signal")
        if sig is None or key in df.columns:
            continue
        cols[key] = np.asarray(sig, dtype=int)
    if not cols:
        return df
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
