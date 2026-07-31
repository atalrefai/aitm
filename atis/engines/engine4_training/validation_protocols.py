"""Professional time-series validation protocols for Engine 4.

Scientific basis (industry + academic):
- Expanding / rolling walk-forward — López de Prado, *Advances in Financial
  Machine Learning* (AFML), Ch. 7; Bailey et al. (2014) on backtest overfitting.
- Purge + embargo — AFML Ch. 7; prevents label-window leakage across folds.
- Combinatorial purged CV (CPCV-lite) — AFML Ch. 12; path diversity without
  random shuffles that destroy temporal dependence.
- Regime-conditional evaluation — Ang & Timmermann (2012) regime-switching;
  hedge-fund practice of stress by trend/vol state (AQR / Two Sigma style reports).

All splitters are chronological. No random K-fold.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

ValidationMode = Literal["expanding", "rolling", "timeseries_cv", "cpcv_lite"]


def expanding_window_splits(
    n: int,
    n_splits: int,
    train_ratio: float,
    *,
    embargo: int = 0,
    purge: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Anchored expanding train → future test (default ATIS walk-forward).

    Inlined (not imported from package __init__) to avoid circular reload issues
    when the web API hot-reloads engine4_training.
    """
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    embargo = max(0, int(embargo))
    purge = max(0, int(purge))
    min_train = max(40, int(n * train_ratio / max(n_splits, 1)))
    min_train = min(min_train, max(40, n // 3))
    fold_size = max(20, (n - min_train) // max(n_splits, 1))
    actual_splits = n_splits
    if n < 200:
        actual_splits = min(n_splits, 3)
        fold_size = max(15, (n - min_train) // max(actual_splits, 1))
    for i in range(actual_splits):
        train_end = min_train + i * fold_size
        test_start = min(train_end + embargo, n)
        test_end = min(train_end + fold_size + embargo, n)
        if train_end >= n or test_end <= test_start:
            break
        train_idx = np.arange(0, train_end)
        if purge > 0 and len(train_idx) > purge + 20:
            train_idx = train_idx[: max(20, len(train_idx) - purge)]
        test_idx = np.arange(test_start, test_end)
        if len(test_idx) < 5 or len(train_idx) < 20:
            continue
        splits.append((train_idx, test_idx))
    return splits


def rolling_window_splits(
    n: int,
    n_splits: int,
    *,
    train_size: int | None = None,
    test_size: int | None = None,
    embargo: int = 0,
    purge: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixed-length rolling train window (adapts faster to regime shifts).

    Prefer rolling when non-stationarity dominates; expanding when scarce history
    needs maximum sample size (classic WF trade-off).
    """
    embargo = max(0, int(embargo))
    purge = max(0, int(purge))
    n_splits = max(1, int(n_splits))
    if train_size is None:
        train_size = max(40, int(n * 0.45))
    if test_size is None:
        test_size = max(15, (n - train_size) // max(n_splits, 1))
    train_size = int(min(max(40, train_size), max(40, n - test_size - embargo - 5)))
    test_size = int(max(10, test_size))

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    # Place windows so last fold ends near n
    span = train_size + embargo + test_size
    if span >= n:
        train_idx = np.arange(0, max(40, n - test_size - embargo))
        if purge > 0 and len(train_idx) > purge + 20:
            train_idx = train_idx[: max(20, len(train_idx) - purge)]
        test_start = min(len(train_idx) + purge + embargo, n - 5) if purge else min(train_idx[-1] + 1 + embargo, n - 5)
        test_idx = np.arange(test_start, n)
        if len(test_idx) >= 5 and len(train_idx) >= 20:
            splits.append((train_idx, test_idx))
        return splits

    step = max(test_size, (n - span) // max(n_splits - 1, 1)) if n_splits > 1 else test_size
    for i in range(n_splits):
        train_end = train_size + i * step
        if train_end + embargo + 5 >= n:
            break
        train_start = max(0, train_end - train_size)
        train_idx = np.arange(train_start, train_end)
        if purge > 0 and len(train_idx) > purge + 20:
            train_idx = train_idx[: max(20, len(train_idx) - purge)]
        test_start = min(train_end + embargo, n)
        test_end = min(test_start + test_size, n)
        test_idx = np.arange(test_start, test_end)
        if len(test_idx) < 5 or len(train_idx) < 20:
            continue
        splits.append((train_idx, test_idx))
    return splits


def timeseries_cv_splits(
    n: int,
    n_splits: int = 5,
    *,
    gap: int = 0,
    test_size: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """sklearn-style TimeSeriesSplit semantics (expanding, optional gap).

    `gap` maps to ATIS embargo; callers should still apply label purge separately
    when using triple-barrier horizons.
    """
    n_splits = max(2, int(n_splits))
    gap = max(0, int(gap))
    if test_size is None:
        test_size = max(15, n // (n_splits + 1))
    test_size = int(max(10, test_size))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        test_end = n - (n_splits - i - 1) * test_size
        test_start = test_end - test_size
        train_end = max(20, test_start - gap)
        if train_end < 20 or test_start <= 0 or test_end <= test_start:
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(test_idx) < 5:
            continue
        splits.append((train_idx, test_idx))
    return splits


def cpcv_lite_paths(
    n: int,
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo: int = 0,
    purge: int = 0,
    max_paths: int = 12,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Combinatorial purged CV lite — subset of group combinations as paths.

    Full CPCV is O(C(n_groups, n_test)); we cap paths for compute budgets while
    still estimating path-wise performance dispersion (PBO input).
    """
    from itertools import combinations

    n_groups = max(4, int(n_groups))
    n_test_groups = max(1, min(int(n_test_groups), n_groups // 2))
    embargo = max(0, int(embargo))
    purge = max(0, int(purge))
    edges = np.linspace(0, n, n_groups + 1, dtype=int)
    groups = [np.arange(edges[i], edges[i + 1]) for i in range(n_groups)]
    groups = [g for g in groups if len(g) >= 5]
    if len(groups) < 4:
        return expanding_window_splits(n, 3, 0.7, embargo=embargo, purge=purge)

    paths: list[tuple[np.ndarray, np.ndarray]] = []
    for test_ids in combinations(range(len(groups)), n_test_groups):
        test_set = set(test_ids)
        # Prefer contiguous-ish test blocks to reduce unrealistic futures→past
        if max(test_ids) - min(test_ids) + 1 > n_test_groups + 1:
            continue
        train_parts = [groups[i] for i in range(len(groups)) if i not in test_set]
        test_parts = [groups[i] for i in test_ids]
        if not train_parts or not test_parts:
            continue
        train_idx = np.concatenate(train_parts)
        test_idx = np.concatenate(test_parts)
        # Keep only train bars strictly before first test bar (causal path)
        t0 = int(test_idx.min())
        train_idx = train_idx[train_idx < t0 - embargo]
        if purge > 0 and len(train_idx) > purge + 20:
            train_idx = train_idx[train_idx < (t0 - embargo - purge)]
        test_idx = np.sort(test_idx)
        train_idx = np.sort(train_idx)
        if len(train_idx) < 40 or len(test_idx) < 10:
            continue
        paths.append((train_idx, test_idx))
        if len(paths) >= max_paths:
            break
    return paths


def build_validation_splits(
    n: int,
    *,
    mode: ValidationMode | str = "expanding",
    n_splits: int = 5,
    train_ratio: float = 0.7,
    embargo: int = 0,
    purge: int = 0,
    rolling_train_size: int | None = None,
    rolling_test_size: int | None = None,
    cpcv_n_groups: int | None = None,
    cpcv_n_test_groups: int | None = None,
    cpcv_max_paths: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Factory for validation protocol selection."""
    mode_l = str(mode or "expanding").lower().strip()
    if mode_l in {"rolling", "rolling_window"}:
        return rolling_window_splits(
            n,
            n_splits,
            train_size=rolling_train_size,
            test_size=rolling_test_size,
            embargo=embargo,
            purge=purge,
        )
    if mode_l in {"timeseries_cv", "time_series_cv", "ts_cv"}:
        return timeseries_cv_splits(n, n_splits, gap=embargo)
    if mode_l in {"cpcv", "cpcv_lite", "combinatorial"}:
        max_paths = int(cpcv_max_paths) if cpcv_max_paths is not None else max(6, n_splits * 2)
        kwargs: dict[str, Any] = {"embargo": embargo, "purge": purge, "max_paths": max_paths}
        if cpcv_n_groups is not None:
            kwargs["n_groups"] = int(cpcv_n_groups)
        if cpcv_n_test_groups is not None:
            kwargs["n_test_groups"] = int(cpcv_n_test_groups)
        return cpcv_lite_paths(n, **kwargs)
    return expanding_window_splits(n, n_splits, train_ratio, embargo=embargo, purge=purge)


def classify_market_regimes(
    close: np.ndarray,
    atr_pct: np.ndarray,
    *,
    trend_window: int = 48,
    vol_low_q: float = 0.30,
    vol_high_q: float = 0.70,
) -> dict[str, np.ndarray]:
    """Boolean masks: trending / ranging / high_vol / low_vol (causal rolling).

    Trend: |SMA slope| high; Range: low |slope| + mid vol; Vol by ATR% quantiles
    computed on the full series for *evaluation slicing only* (not used as
    training features — avoids fitting regimes that leak into model inputs).
    """
    close = np.asarray(close, dtype=float)
    atr_pct = np.asarray(atr_pct, dtype=float)
    n = len(close)
    w = max(10, int(trend_window))
    s = pd.Series(close)
    sma = s.rolling(w, min_periods=max(5, w // 3)).mean()
    slope = (sma - sma.shift(max(5, w // 4))) / np.maximum(sma.shift(max(5, w // 4)).abs(), 1e-12)
    slope_a = slope.fillna(0.0).values
    abs_slope = np.abs(slope_a)
    slope_thr = float(np.nanquantile(abs_slope[np.isfinite(abs_slope)], 0.55)) if n > 50 else 0.002

    atr_clean = atr_pct[np.isfinite(atr_pct)]
    if len(atr_clean) < 20:
        lo = float(np.nanmedian(atr_pct)) if n else 0.0
        hi = lo
    else:
        lo = float(np.quantile(atr_clean, vol_low_q))
        hi = float(np.quantile(atr_clean, vol_high_q))

    trending = abs_slope >= slope_thr
    ranging = abs_slope < slope_thr * 0.65
    high_vol = atr_pct >= hi
    low_vol = atr_pct <= lo
    return {
        "trending": trending,
        "ranging": ranging,
        "high_volatility": high_vol,
        "low_volatility": low_vol,
    }


def evaluate_by_regime(
    returns: np.ndarray,
    regime_masks: dict[str, np.ndarray],
    *,
    financial_fn,
    min_bars: int = 30,
) -> dict[str, Any]:
    """Per-regime financial metrics for generalization / stress diagnostics."""
    rets = np.asarray(returns, dtype=float)
    out: dict[str, Any] = {"regimes": {}, "stable": True, "notes": []}
    sharpes: list[float] = []
    for name, mask in regime_masks.items():
        m = np.asarray(mask, dtype=bool)
        if len(m) != len(rets):
            out["regimes"][name] = {"error": "mask_length_mismatch"}
            continue
        # Keep zeros outside regime so equity path stays aligned; count trades inside.
        sliced = np.where(m, rets, 0.0)
        n_active = int(np.sum(m))
        n_trades = int(np.sum(sliced != 0))
        if n_active < min_bars or n_trades < 3:
            out["regimes"][name] = {
                "n_bars": n_active,
                "n_trades": n_trades,
                "skipped": True,
                "reason": "insufficient_sample",
            }
            continue
        fin = financial_fn(sliced)
        row = {
            "n_bars": n_active,
            "n_trades": n_trades,
            "sharpe": float(fin.get("sharpe", 0.0) or 0.0),
            "sortino": float(fin.get("sortino", 0.0) or 0.0),
            "max_drawdown": float(fin.get("max_drawdown", 0.0) or 0.0),
            "profit_factor": float(fin.get("profit_factor", 0.0) or 0.0),
            "expectancy": float(fin.get("expectancy", 0.0) or 0.0),
            "win_rate": float(fin.get("win_rate", 0.0) or 0.0),
            "skipped": False,
        }
        out["regimes"][name] = row
        sharpes.append(row["sharpe"])

    if len(sharpes) >= 2:
        spread = float(max(sharpes) - min(sharpes))
        out["sharpe_regime_spread"] = spread
        # Unstable if one regime strongly positive and another strongly negative
        if min(sharpes) < -0.5 and max(sharpes) > 1.0 and spread > 2.5:
            out["stable"] = False
            out["notes"].append("performance_collapses_in_at_least_one_regime")
        elif spread > 4.0:
            out["stable"] = False
            out["notes"].append("large_sharpe_dispersion_across_regimes")
    else:
        out["notes"].append("insufficient_regime_coverage_for_stability")
    return out


def protocol_rationale() -> dict[str, str]:
    """Documented why each protocol exists (for reports / UI)."""
    return {
        "expanding": (
            "Maximizes train sample; best when history is scarce. AFML expanding WF."
        ),
        "rolling": (
            "Discards stale regimes; better adaptation under structural breaks."
        ),
        "timeseries_cv": (
            "sklearn TimeSeriesSplit analogue with gap=embargo; honest chronology."
        ),
        "cpcv_lite": (
            "Path diversity for PBO / backtest-overfit diagnostics (AFML Ch.12 lite)."
        ),
        "regime_slices": (
            "Stress OOS by trend/range and high/low vol — not a training feature."
        ),
    }
