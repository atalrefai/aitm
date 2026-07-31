"""Barrier sensitivity sweep + label-noise cleaning (v16)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _directional_clarity(y: np.ndarray, w: np.ndarray) -> float:
    mask = y != 0
    if int(mask.sum()) < 30:
        return 0.0
    return float(np.mean(w[mask]))


def _label_balance_score(y: np.ndarray) -> float:
    pos = int((y == 1).sum())
    neg = int((y == -1).sum())
    flat = int((y == 0).sum())
    n = max(len(y), 1)
    dir_share = (pos + neg) / n
    if pos + neg == 0:
        return 0.0
    imb = max(pos, neg) / max(min(pos, neg), 1)
    # Prefer 20–55% directional, imbalance < 2.0
    share_score = 1.0 - abs(dir_share - 0.35) / 0.35
    imb_score = max(0.0, 1.0 - (imb - 1.0) / 2.0)
    flat_pen = 0.15 if flat / n > 0.85 else 0.0
    return float(max(0.0, 0.55 * share_score + 0.45 * imb_score - flat_pen))


def sweep_barrier_params(
    df: pd.DataFrame,
    *,
    timeframe: str,
    cfg: dict[str, Any] | None = None,
    label_fn: Any = None,
) -> dict[str, Any]:
    """Grid-search barrier ATR multiplier × horizon on train-only quality score.

    Does not look at future model OOS — only label health (clarity, balance, noise proxy).
    """
    cfg = cfg or {}
    if not bool(cfg.get("barrier_sweep_enabled", True)):
        return {"enabled": False, "reason": "disabled"}

    from atis.engines.engine4_training import (
        horizon_for_timeframe,
        triple_barrier_labels_and_weights,
    )

    label_fn = label_fn or triple_barrier_labels_and_weights
    base_h = horizon_for_timeframe(timeframe, cfg)
    base_atr = float(cfg.get("barrier_atr_multiplier", 1.5))
    by_tf = cfg.get("barrier_atr_multiplier_by_tf") or {}
    if timeframe in by_tf:
        base_atr = float(by_tf[timeframe])

    atr_grid = cfg.get("barrier_sweep_atr_grid") or [
        round(base_atr * m, 3) for m in (0.85, 1.0, 1.15, 1.35)
    ]
    hor_grid = cfg.get("barrier_sweep_horizon_grid") or [
        max(2, int(base_h + d)) for d in (-2, -1, 0, 1)
    ]
    # Deduplicate
    atr_grid = sorted({float(a) for a in atr_grid if float(a) > 0.3})
    hor_grid = sorted({int(h) for h in hor_grid if int(h) >= 2})

    # Use only first 70% of bars (no future leakage into sweep choice)
    n = len(df)
    cut = max(80, int(n * 0.70))
    sub = df.iloc[:cut]

    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for atr in atr_grid:
        for hor in hor_grid:
            try:
                labels, weights = label_fn(sub, horizon=int(hor), atr_mult=float(atr))
                y = np.asarray(labels, dtype=float)
                w = np.asarray(weights, dtype=float)
                bal = _label_balance_score(y)
                clar = _directional_clarity(y, w)
                dir_share = float(np.mean(y != 0))
                score = 0.55 * bal + 0.35 * clar + 0.10 * min(dir_share / 0.35, 1.0)
                # Prefer staying near base unless clear gain
                score -= 0.02 * abs(float(atr) - base_atr)
                score -= 0.01 * abs(int(hor) - base_h)
                row = {
                    "atr_mult": float(atr),
                    "horizon": int(hor),
                    "score": round(score, 5),
                    "balance": round(bal, 4),
                    "clarity": round(clar, 4),
                    "directional_share": round(dir_share, 4),
                }
                trials.append(row)
                if best is None or score > float(best["score"]):
                    best = row
            except Exception as exc:
                trials.append({"atr_mult": atr, "horizon": hor, "error": str(exc)})

    if best is None:
        return {
            "enabled": True,
            "applied": False,
            "reason": "all_trials_failed",
            "base_atr": base_atr,
            "base_horizon": base_h,
            "trials": trials[:20],
        }

    # Apply only if meaningfully better than base trial
    base_trial = next(
        (t for t in trials if t.get("atr_mult") == base_atr and t.get("horizon") == base_h),
        None,
    )
    base_score = float(base_trial["score"]) if base_trial and "score" in base_trial else -1.0
    apply = float(best["score"]) >= base_score + float(cfg.get("barrier_sweep_min_gain", 0.02))
    return {
        "enabled": True,
        "applied": apply,
        "base_atr": base_atr,
        "base_horizon": base_h,
        "chosen_atr": best["atr_mult"] if apply else base_atr,
        "chosen_horizon": best["horizon"] if apply else base_h,
        "best": best,
        "base_score": round(base_score, 5),
        "gain": round(float(best["score"]) - base_score, 5),
        "n_trials": len(trials),
        "trials": sorted(
            [t for t in trials if "score" in t],
            key=lambda x: -float(x["score"]),
        )[:12],
        "summary_ar": (
            f"مسح الحواجز: {'طُبّق' if apply else 'أُبقي الأساس'} · "
            f"ATR={best['atr_mult'] if apply else base_atr} · H={best['horizon'] if apply else base_h}"
        ),
    }


def clean_label_weights(
    y: pd.Series | np.ndarray,
    label_weights: np.ndarray,
    X: pd.DataFrame | None = None,
    *,
    cfg: dict[str, Any] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Down-weight likely noisy directional labels via weak-model disagreement."""
    cfg = cfg or {}
    w = np.asarray(label_weights, dtype=float).copy()
    y_arr = np.asarray(y, dtype=float)
    meta: dict[str, Any] = {"enabled": bool(cfg.get("label_cleaning_enabled", True))}
    if not meta["enabled"] or X is None or len(y_arr) < 120:
        meta["reason"] = "disabled_or_too_small"
        return w, meta

    from atis.engines.engine4_training.label_quality import estimate_label_noise

    noise = estimate_label_noise(
        X,
        y_arr,
        seed=seed,
        n_splits=int(cfg.get("label_noise_cv_splits", 3)),
        max_rows=int(cfg.get("label_noise_max_rows", 6000)),
    )
    meta["noise"] = noise
    noise_score = float(noise.get("noise_score") or 0.0) if noise.get("enabled") else 0.0
    # Soft shrink weights for low-clarity directional samples when noise elevated
    dir_mask = y_arr != 0
    low_clarity = dir_mask & (w < float(cfg.get("label_clean_clarity_floor", 0.55)))
    shrink = float(cfg.get("label_clean_shrink", 0.65))
    if noise_score >= float(cfg.get("label_clean_noise_trigger", 0.35)) and low_clarity.any():
        w[low_clarity] *= shrink
        meta["n_downweighted"] = int(low_clarity.sum())
        meta["shrink"] = shrink
    else:
        meta["n_downweighted"] = 0
    # Mild time emphasis already exists separately; here boost clear directional hits
    clear = dir_mask & (w >= 0.85)
    if clear.any():
        w[clear] = np.minimum(w[clear] * 1.05, float(cfg.get("label_weight_cap", 1.35)))
        meta["n_boosted_clear"] = int(clear.sum())
    meta["mean_weight"] = round(float(np.mean(w)), 4)
    return w, meta
