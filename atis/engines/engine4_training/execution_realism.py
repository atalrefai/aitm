"""Realistic execution simulation for backtests.

References:
- Almgren & Chriss (2000/2001) — market impact / optimal execution intuition.
- Kissell (2014) — *The Science of Algorithmic Trading* (slippage, latency).
- Institutional practice: scale costs with volatility; delay fills by ≥1 bar
  when signal is known only after close (no look-ahead fill).

ATIS default: signal on bar i → fill at open/close of i+latency_bars with
vol-scaled spread/slippage and optional extra slippage for delay.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def apply_latency_index(i: int, *, latency_bars: int, n: int) -> int | None:
    """Map signal bar → fill bar; None if fill would fall outside series."""
    fill = int(i) + max(0, int(latency_bars))
    if fill >= n:
        return None
    return fill


def execution_cost_fraction(
    close: float,
    *,
    spread_pips: float,
    slippage_pips: float,
    commission_per_lot: float,
    pip_size: float = 0.0001,
    atr_pct: float | None = None,
    vol_slippage_k: float = 1.25,
    latency_bars: int = 0,
    latency_slippage_pips_per_bar: float = 0.5,
    dynamic: bool = True,
) -> dict[str, float]:
    """Round-trip cost as fraction of price + diagnostics."""
    from atis.engines.engine4_training.adaptive_learning import dynamic_execution_costs

    sp, sl = float(spread_pips), float(slippage_pips)
    comm = float(commission_per_lot)
    if dynamic and atr_pct is not None and np.isfinite(atr_pct):
        sp, sl, comm = dynamic_execution_costs(
            close,
            float(atr_pct),
            base_spread_pips=sp,
            base_slippage_pips=sl,
            commission_per_lot=comm,
            pip_size=pip_size,
            vol_slippage_k=vol_slippage_k,
        )
    # Latency adds adverse selection / delayed fill slippage
    lat = max(0, int(latency_bars))
    sl = sl + lat * float(latency_slippage_pips_per_bar)
    spread_cost = (sp + sl) * pip_size / max(abs(close), 1e-12)
    commission_frac = comm / max(abs(close) * 100.0, 1.0)
    total = float(spread_cost + commission_frac)
    return {
        "cost_fraction": total,
        "spread_pips_eff": float(sp),
        "slippage_pips_eff": float(sl),
        "commission_per_lot": float(comm),
        "latency_bars": float(lat),
    }


def simulate_trade_returns(
    close: np.ndarray,
    preds: np.ndarray,
    *,
    hold_bars: int = 1,
    spread_pips: float,
    slippage_pips: float,
    commission_per_lot: float,
    pip_size: float = 0.0001,
    confidences: np.ndarray | None = None,
    min_confidence: float = 0.0,
    non_overlapping: bool = True,
    atr_pct: np.ndarray | None = None,
    dynamic_costs: bool = True,
    vol_slippage_k: float = 1.25,
    latency_bars: int = 0,
    execution_delay_bars: int = 0,
    latency_slippage_pips_per_bar: float = 0.5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Horizon PnL with optional latency, delay, and vol-scaled costs.

    `latency_bars`: bars between signal and fill (e.g. 1 = next bar).
    `execution_delay_bars`: extra bars before entry (broker/API lag).
    Entry is recorded at fill index; return attributed to signal index for
    alignment with prediction arrays.
    """
    close = np.asarray(close, dtype=float)
    preds = np.asarray(preds)
    n = len(preds)
    rets = np.zeros(n, dtype=float)
    hold_bars = max(1, int(hold_bars))
    delay = max(0, int(latency_bars)) + max(0, int(execution_delay_bars))
    next_free = 0
    trades = 0
    skipped_low_conf = 0
    skipped_overlap = 0
    skipped_latency = 0
    cost_sum = 0.0

    atr = None
    if atr_pct is not None:
        atr = np.asarray(atr_pct, dtype=float)
        if len(atr) != n:
            atr = None

    for i in range(n):
        side = int(preds[i])
        if side == 0:
            continue
        if confidences is not None and float(confidences[i]) < float(min_confidence):
            skipped_low_conf += 1
            continue
        if non_overlapping and i < next_free:
            skipped_overlap += 1
            continue
        fill_i = apply_latency_index(i, latency_bars=delay, n=n)
        if fill_i is None:
            skipped_latency += 1
            continue
        exit_i = fill_i + hold_bars
        if exit_i >= n:
            skipped_latency += 1
            continue
        entry = float(close[fill_i])
        exit_px = float(close[exit_i])
        if entry <= 0 or not np.isfinite(entry) or not np.isfinite(exit_px):
            continue
        raw = (exit_px - entry) / entry * side
        atr_i = float(atr[fill_i]) if atr is not None else None
        cost_info = execution_cost_fraction(
            entry,
            spread_pips=spread_pips,
            slippage_pips=slippage_pips,
            commission_per_lot=commission_per_lot,
            pip_size=pip_size,
            atr_pct=atr_i,
            vol_slippage_k=vol_slippage_k,
            latency_bars=delay,
            latency_slippage_pips_per_bar=latency_slippage_pips_per_bar,
            dynamic=dynamic_costs,
        )
        cost = float(cost_info["cost_fraction"])
        rets[i] = raw - cost
        cost_sum += cost
        trades += 1
        next_free = exit_i  # free after position closes

    stats = {
        "trades": float(trades),
        "skipped_low_conf": float(skipped_low_conf),
        "skipped_overlap": float(skipped_overlap),
        "skipped_latency": float(skipped_latency),
        "hold_bars": float(hold_bars),
        "min_confidence": float(min_confidence),
        "latency_bars": float(max(0, int(latency_bars))),
        "execution_delay_bars": float(max(0, int(execution_delay_bars))),
        "mean_cost_fraction": float(cost_sum / max(trades, 1)),
        "dynamic_costs": float(1.0 if dynamic_costs else 0.0),
    }
    return rets, stats


def execution_rationale() -> dict[str, str]:
    return {
        "spread": "Bid-ask round-trip; primary retail FX/metal friction.",
        "slippage": "Adverse fill vs mid; rises with volatility and order size.",
        "commission": "Broker fee mapped to fractional price cost.",
        "latency": "Signal→fill lag; removes optimistic same-bar fills.",
        "execution_delay": "Extra API/broker delay beyond bar latency.",
        "dynamic_costs": "ATR%-scaled spread/slippage (vol regimes).",
    }
