"""Stress testing, session slices, and Monte Carlo path robustness."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


def session_masks_from_frame(df: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(df)
    out: dict[str, np.ndarray] = {}
    if "session" in df.columns:
        s = df["session"].astype(str).str.lower()
        for name, keys in {
            "asian": ("asia", "asian", "tokyo"),
            "london": ("london", "eu", "europe"),
            "new_york": ("newyork", "new_york", "ny", "us", "america"),
        }.items():
            out[name] = s.apply(lambda x, ks=keys: any(k in str(x) for k in ks)).values.astype(bool)
    else:
        # Hour-of-day proxy from timestamp
        ts = None
        for col in ("timestamp", "time", "datetime"):
            if col in df.columns:
                ts = pd.to_datetime(df[col], utc=True, errors="coerce")
                break
        if ts is None and isinstance(df.index, pd.DatetimeIndex):
            ts = pd.Series(df.index, index=df.index)
        if ts is not None:
            h = ts.dt.hour.fillna(12).astype(int).values
            out["asian"] = (h >= 0) & (h < 8)
            out["london"] = (h >= 7) & (h < 16)
            out["new_york"] = (h >= 12) & (h < 21)
        else:
            out = {k: np.ones(n, dtype=bool) for k in ("asian", "london", "new_york")}
    return out


def evaluate_session_slices(
    returns: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    financial_fn: Callable[[np.ndarray], dict[str, float]],
    min_trades: int = 8,
) -> dict[str, Any]:
    out: dict[str, Any] = {"sessions": {}}
    for name, mask in masks.items():
        m = np.asarray(mask, dtype=bool)
        if len(m) != len(returns):
            continue
        sliced = np.where(m, returns, 0.0)
        n_tr = int(np.sum(sliced != 0))
        if n_tr < min_trades:
            out["sessions"][name] = {"skipped": True, "n_trades": n_tr}
            continue
        fin = financial_fn(sliced)
        out["sessions"][name] = {
            "skipped": False,
            "n_trades": n_tr,
            "sharpe": float(fin.get("sharpe", 0) or 0),
            "expectancy": float(fin.get("expectancy", 0) or 0),
            "max_drawdown": float(fin.get("max_drawdown", 0) or 0),
        }
    return out


def stress_scenarios(
    returns: np.ndarray,
    *,
    financial_fn: Callable[[np.ndarray], dict[str, float]],
    spread_shock: float = 1.5,
    noise_sigma: float = 0.0005,
    drop_frac: float = 0.08,
    seed: int = 42,
    latency_extra: int = 0,
    gap_shock: float = 0.0,
) -> dict[str, Any]:
    """Apply adverse execution / data shocks to OOS trade returns."""
    r = np.asarray(returns, dtype=float).copy()
    rng = np.random.default_rng(seed)
    traded = r != 0
    base = financial_fn(r)

    # Higher friction: shrink winning trades / worsen losers
    shock = r.copy()
    shock[traded] = shock[traded] - np.sign(shock[traded] + 1e-12) * abs(float(spread_shock)) * 0.00015
    # Noise
    noisy = r.copy()
    noisy[traded] = noisy[traded] + rng.normal(0, noise_sigma, size=int(traded.sum()))
    # Missing / dropped fills
    dropped = r.copy()
    idx = np.flatnonzero(traded)
    if len(idx):
        kill = rng.choice(idx, size=max(1, int(len(idx) * drop_frac)), replace=False)
        dropped[kill] = 0.0
    # Crash cluster: force consecutive losses on a block of trades
    crash = r.copy()
    if len(idx) >= 20:
        start = int(rng.integers(0, max(1, len(idx) - 15)))
        block = idx[start : start + 12]
        crash[block] = -np.abs(crash[block]) - 0.001

    # Latency extra: shift trades later by dropping early winners (proxy)
    delayed = r.copy()
    if int(latency_extra) > 0 and len(idx) > int(latency_extra) + 5:
        # Zero first latency_extra trades (missed due to delay)
        delayed[idx[: int(latency_extra)]] = 0.0
        delayed[traded] = delayed[traded] - np.sign(delayed[traded] + 1e-12) * 0.00005 * int(latency_extra)

    # Gap / weekend shock: amplify a random subset of losses
    gapped = r.copy()
    if float(gap_shock) > 0 and len(idx) >= 10:
        n_gap = max(1, len(idx) // 12)
        gap_idx = rng.choice(idx, size=n_gap, replace=False)
        gapped[gap_idx] = gapped[gap_idx] - np.sign(gapped[gap_idx] + 1e-12) * abs(float(gap_shock))

    scenarios = {
        "base": base,
        "spread_shock": financial_fn(shock),
        "noise": financial_fn(noisy),
        "missing_fills": financial_fn(dropped),
        "crash_cluster": financial_fn(crash),
        "latency_extra": financial_fn(delayed),
        "gap_shock": financial_fn(gapped),
    }
    sharpes = [float(v.get("sharpe", 0) or 0) for v in scenarios.values()]
    return {
        "scenarios": {
            k: {
                "sharpe": float(v.get("sharpe", 0) or 0),
                "expectancy": float(v.get("expectancy", 0) or 0),
                "max_drawdown": float(v.get("max_drawdown", 0) or 0),
                "n_trades": float(v.get("n_trades", 0) or 0),
            }
            for k, v in scenarios.items()
        },
        "worst_sharpe": float(min(sharpes)) if sharpes else 0.0,
        "median_sharpe": float(np.median(sharpes)) if sharpes else 0.0,
        "robust": bool(min(sharpes) > -0.5) if sharpes else False,
    }


def monte_carlo_trade_paths(
    trade_returns: np.ndarray,
    *,
    n_paths: int = 400,
    seed: int = 42,
    financial_fn: Callable[[np.ndarray], dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Bootstrap reshuffle of trade PnL to estimate ruin / success probabilities."""
    x = np.asarray(trade_returns, dtype=float)
    x = x[np.isfinite(x) & (x != 0)]
    if len(x) < 12:
        return {"enabled": False, "reason": "too_few_trades", "n_paths": 0}
    rng = np.random.default_rng(seed)
    n_paths = int(max(50, min(n_paths, 2000)))
    terminal = np.empty(n_paths, dtype=float)
    max_dd = np.empty(n_paths, dtype=float)
    for i in range(n_paths):
        sample = x[rng.integers(0, len(x), size=len(x))]
        equity = np.cumprod(1.0 + sample)
        peak = np.maximum.accumulate(equity)
        dd = equity / peak - 1.0
        terminal[i] = float(equity[-1] - 1.0)
        max_dd[i] = float(dd.min()) if len(dd) else 0.0
    p_profit = float(np.mean(terminal > 0))
    p_ruin = float(np.mean(max_dd <= -0.25))
    return {
        "enabled": True,
        "n_paths": n_paths,
        "n_trades": int(len(x)),
        "p_profit": round(p_profit, 4),
        "p_dd_gt_25pct": round(p_ruin, 4),
        "terminal_return_p05": float(np.quantile(terminal, 0.05)),
        "terminal_return_p50": float(np.quantile(terminal, 0.50)),
        "terminal_return_p95": float(np.quantile(terminal, 0.95)),
        "max_dd_p05": float(np.quantile(max_dd, 0.05)),
        "max_dd_median": float(np.median(max_dd)),
        "stable": bool(p_profit >= 0.55 and p_ruin <= 0.15),
    }
