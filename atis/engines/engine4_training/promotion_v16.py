"""Fold stability, crisis/recent holdouts, and promotion helpers (v16)."""

from __future__ import annotations

from typing import Any

import numpy as np


def fold_stability_report(
    fold_metrics: list[dict[str, Any]],
    *,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require consistent Val Sharpe across liquid folds (not one lucky fold)."""
    cfg = cfg or {}
    sharpes: list[float] = []
    rates: list[float] = []
    liquid = 0
    min_val = float(cfg.get("fold_stability_min_val_trades", 10))
    for row in fold_metrics:
        n_vt = float(row.get("n_val_trades", (row.get("policy") or {}).get("val_trades", 0)) or 0)
        sh = float((row.get("policy") or {}).get("val_sharpe", row.get("val_sharpe", 0)) or 0)
        rate = float(row.get("trade_rate", 0) or 0)
        if n_vt >= min_val:
            liquid += 1
            sharpes.append(sh)
            rates.append(rate)
    if len(sharpes) < 2:
        return {
            "enabled": True,
            "stable": False,
            "reason": "insufficient_liquid_folds",
            "n_liquid": liquid,
            "gate_pass": not bool(cfg.get("fail_on_fold_unstable", True)),
        }
    arr = np.asarray(sharpes, dtype=float)
    med = float(np.median(arr))
    iqr = float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25))
    frac_positive = float(np.mean(arr > 0))
    # Unstable if IQR huge vs median or few positive folds
    max_iqr = float(cfg.get("fold_stability_max_iqr", 4.0))
    min_pos_frac = float(cfg.get("fold_stability_min_pos_frac", 0.6))
    unstable = (iqr > max_iqr and med < 3.0) or (frac_positive < min_pos_frac)
    # Also flag if trade rates all pegged to max (policy saturation)
    rate_pegged = bool(rates) and float(np.mean(np.asarray(rates) >= 0.20)) >= 0.8
    return {
        "enabled": True,
        "stable": not unstable,
        "n_liquid": liquid,
        "median_val_sharpe": round(med, 4),
        "iqr_val_sharpe": round(iqr, 4),
        "frac_positive_folds": round(frac_positive, 4),
        "trade_rate_pegged": rate_pegged,
        "gate_pass": (not unstable) if bool(cfg.get("fail_on_fold_unstable", True)) else True,
        "summary_ar": (
            f"ثبات الطيات: {'مستقر' if not unstable else 'غير مستقر'} · "
            f"median={med:.2f} · IQR={iqr:.2f} · +folds={frac_positive:.0%}"
        ),
    }


def crisis_recent_holdout_slices(
    n: int,
    *,
    recent_frac: float = 0.12,
    crisis_frac: float = 0.15,
) -> dict[str, np.ndarray]:
    """Index slices for recent-window and mid-sample 'crisis-like' stress holdout.

    Crisis proxy: middle band of the series (often contains regime transitions in
    multi-year gold samples). Recent: last fraction of bars.
    """
    n = int(n)
    recent_n = max(30, int(n * float(recent_frac)))
    crisis_n = max(30, int(n * float(crisis_frac)))
    recent = np.arange(max(0, n - recent_n), n)
    mid = n // 2
    half = crisis_n // 2
    lo = max(0, mid - half)
    hi = min(n, lo + crisis_n)
    crisis = np.arange(lo, hi)
    return {"recent": recent, "crisis": crisis}


def evaluate_holdout_slice(
    returns: np.ndarray,
    idx: np.ndarray,
    *,
    financial_fn,
    name: str,
) -> dict[str, Any]:
    if len(idx) < 10:
        return {"name": name, "skipped": True, "reason": "too_short"}
    sub = np.asarray(returns)[idx]
    fin = financial_fn(sub)
    return {
        "name": name,
        "skipped": False,
        "n_bars": int(len(idx)),
        "sharpe": fin.get("sharpe"),
        "expectancy": fin.get("expectancy"),
        "max_drawdown": fin.get("max_drawdown"),
        "n_trades": fin.get("n_trades"),
        "total_return": fin.get("total_return"),
    }


def confidence_position_size(
    confidence: float,
    *,
    atr_pct: float,
    base_size: float = 1.0,
    max_size: float = 1.5,
    min_size: float = 0.25,
    target_atr: float = 0.002,
) -> float:
    """Vol-targeted size scaled by model confidence (Kelly-lite, capped)."""
    conf = float(np.clip(confidence, 0.0, 1.0))
    vol_scale = float(target_atr) / max(float(atr_pct), 1e-6)
    vol_scale = float(np.clip(vol_scale, 0.4, 1.6))
    # Confidence above 0.55 scales up gently
    conf_scale = 0.6 + 0.8 * max(0.0, conf - 0.50) / 0.50
    size = float(base_size) * vol_scale * conf_scale
    return float(np.clip(size, min_size, max_size))
