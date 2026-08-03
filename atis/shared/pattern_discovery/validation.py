"""Statistical validation gates for pattern promotion (Engine4-style reject-weak).

Causal: all folds / OOS / Monte Carlo use only forward returns already computed
from past bars — no future labels leak into detection.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Calibrated gates: strict enough to reject noise, loose enough to promote real edges
DEFAULT_GATES: dict[str, float] = {
    "min_evaluated": 25,
    "min_success_rate": 0.54,
    "min_precision": 0.54,
    "min_recall": 0.18,
    "min_f1": 0.28,
    "min_expectancy": 0.0,
    "min_profit_factor": 1.08,
    "min_sharpe": 0.20,
    "max_drawdown": 0.40,
    "min_win_rate": 0.54,
    "min_stability": 0.40,
    "min_ci_low": 0.48,
    "monte_carlo_p": 0.20,
    "min_quality_score": 0.42,
    # Soft promotion (Engine4 recommend) without full hard approve
    "soft_min_quality": 0.50,
    "soft_min_success": 0.55,
    "soft_min_evaluated": 25,
    "soft_min_expectancy": 0.0,
    # Rankings / Engine4: auto-reject ultra-rare (e.g. Tasuki on XAU)
    "min_occurrences_rank": 20,
}

# Per-timeframe overrides (merged on top of DEFAULT_GATES)
TF_GATE_OVERRIDES: dict[str, dict[str, float]] = {
    "M1": {
        "min_evaluated": 45,
        "soft_min_evaluated": 40,
        "min_occurrences_rank": 40,
        "min_success_rate": 0.53,
        "min_profit_factor": 1.05,
    },
    "M5": {
        "min_evaluated": 35,
        "soft_min_evaluated": 32,
        "min_occurrences_rank": 30,
    },
    "M15": {
        "min_evaluated": 30,
        "soft_min_evaluated": 28,
        "min_occurrences_rank": 25,
    },
    "M30": {
        "min_evaluated": 28,
        "soft_min_evaluated": 26,
        "min_occurrences_rank": 22,
    },
    "H1": {
        "min_evaluated": 28,
        "soft_min_evaluated": 25,
        "min_occurrences_rank": 22,
        "min_success_rate": 0.56,
        "min_profit_factor": 1.12,
        "soft_min_success": 0.56,
        "min_quality_score": 0.45,
        "soft_min_quality": 0.52,
    },
    "H4": {
        "min_evaluated": 22,
        "soft_min_evaluated": 20,
        "min_occurrences_rank": 18,
        "min_success_rate": 0.58,
        "min_profit_factor": 1.15,
        "soft_min_success": 0.57,
        "min_quality_score": 0.48,
        "soft_min_quality": 0.54,
    },
    "D1": {
        "min_evaluated": 15,
        "soft_min_evaluated": 12,
        "min_occurrences_rank": 12,
        "min_success_rate": 0.58,
        "min_profit_factor": 1.15,
    },
}


def gates_for_timeframe(timeframe: str | None = None) -> dict[str, float]:
    """Return DEFAULT_GATES merged with TF-specific overrides."""
    g = dict(DEFAULT_GATES)
    if timeframe:
        g.update(TF_GATE_OVERRIDES.get(str(timeframe).upper(), {}))
    return g


def success_mask(rets: np.ndarray, bias: str) -> np.ndarray:
    """Directional wins; neutral uses magnitude above median absolute move."""
    rets = np.asarray(rets, dtype=float)
    if bias == "bullish":
        return rets > 0
    if bias == "bearish":
        return rets < 0
    # Neutral: meaningful move vs sample median |r| (avoids success_rate≈1.0)
    abs_r = np.abs(rets)
    if len(abs_r) == 0:
        return np.array([], dtype=bool)
    thr = float(np.median(abs_r)) if len(abs_r) else 0.0
    thr = max(thr, 1e-8)
    return abs_r >= thr


def signed_returns(rets: np.ndarray, bias: str) -> np.ndarray:
    rets = np.asarray(rets, dtype=float)
    if bias == "bearish":
        return -rets
    if bias == "bullish":
        return rets
    # Neutral edge: signed by whether move exceeded median threshold
    wins = success_mask(rets, "neutral")
    out = np.zeros_like(rets)
    out[wins] = np.abs(rets[wins])
    out[~wins] = -np.abs(rets[~wins])
    return out


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return (float(max(0.0, lo)), float(min(1.0, hi)))


def _max_drawdown(rets: np.ndarray) -> float:
    if len(rets) == 0:
        return 0.0
    equity = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1e-12)
    return float(np.max(dd)) if len(dd) else 0.0


def _walk_forward_stability(
    rets: np.ndarray,
    successes_mask: np.ndarray,
    *,
    n_folds: int = 5,
) -> dict[str, Any]:
    n = len(rets)
    if n < n_folds * 2:
        rate = float(successes_mask.mean()) if n else 0.0
        return {"fold_rates": [rate], "stability": 1.0 if rate > 0.5 else 0.0, "oos_rate": rate}
    fold_size = n // n_folds
    rates: list[float] = []
    for f in range(n_folds):
        start = f * fold_size
        end = n if f == n_folds - 1 else (f + 1) * fold_size
        chunk = successes_mask[start:end]
        if len(chunk) == 0:
            continue
        rates.append(float(chunk.mean()))
    if not rates:
        return {"fold_rates": [], "stability": 0.0, "oos_rate": 0.0}
    oos = rates[-1]
    stability = float(sum(1 for r in rates if r > 0.5) / len(rates))
    return {"fold_rates": rates, "stability": stability, "oos_rate": float(oos)}


def _monte_carlo_pvalue(
    rets: np.ndarray,
    *,
    bias: str,
    n_sims: int = 200,
    rng: np.random.Generator | None = None,
) -> float:
    """Sign-flip / phase-randomization null — plain shuffle preserves mean and is useless."""
    if len(rets) < 8:
        return 1.0
    gen = rng or np.random.default_rng(42)
    signed = signed_returns(rets, bias)
    obs = float(np.mean(signed))
    null = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        # Randomize signs of absolute moves (destroys directional edge)
        abs_r = np.abs(rets)
        flipped = abs_r * gen.choice([-1.0, 1.0], size=len(rets))
        null[i] = float(np.mean(signed_returns(flipped, bias)))
    return float(np.mean(null >= obs))


def evaluate_validation_metrics(
    forward_returns: np.ndarray,
    *,
    bias: str = "neutral",
    gates: dict[str, float] | None = None,
    timeframe: str | None = None,
) -> dict[str, Any]:
    """Compute full validation battery and approve/reject decision."""
    g = {**gates_for_timeframe(timeframe), **(gates or {})}
    rets = np.asarray(forward_returns, dtype=float)
    rets = rets[~np.isnan(rets)]
    n = int(len(rets))
    empty = {
        "evaluated": 0,
        "approved": False,
        "soft_promoted": False,
        "reject_reasons": ["insufficient_samples"],
        "metrics": {},
        "quality_score": 0.0,
        "gates": g,
        "rare_rejected": True,
    }
    if n == 0:
        return empty

    wins = success_mask(rets, bias)
    successes = int(wins.sum())
    win_rate = successes / n
    precision = win_rate
    recall = float(min(1.0, successes / max(n * 0.5, 1)))
    f1 = (
        2 * precision * recall / max(precision + recall, 1e-12)
        if (precision + recall) > 0
        else 0.0
    )

    signed = signed_returns(rets, bias)
    gross_win = float(signed[signed > 0].sum()) if np.any(signed > 0) else 0.0
    gross_loss = float(-signed[signed < 0].sum()) if np.any(signed < 0) else 0.0
    profit_factor = gross_win / max(gross_loss, 1e-12) if gross_loss > 0 else (2.0 if gross_win > 0 else 0.0)
    expectancy = float(np.mean(signed))
    std = float(np.std(signed)) if n > 1 else 0.0
    sharpe = float(expectancy / std * np.sqrt(min(n, 252))) if std > 1e-12 else 0.0
    mdd = _max_drawdown(signed)
    ci_lo, ci_hi = _wilson_ci(successes, n)
    wf = _walk_forward_stability(rets, wins.astype(float))
    mc_p = _monte_carlo_pvalue(rets, bias=bias)

    fold_rates = wf["fold_rates"]
    fold_std = float(np.std(fold_rates)) if len(fold_rates) > 1 else 0.0
    robustness = float(max(0.0, 1.0 - fold_std * 2.0))

    metrics = {
        "evaluated": n,
        "successes": successes,
        "success_rate": float(win_rate),
        "win_rate": float(win_rate),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "expectancy": expectancy,
        "profit_factor": float(profit_factor),
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "std_dev": std,
        "avg_move_after": float(np.mean(rets)),
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
        "walk_forward": wf,
        "cross_validation_stability": wf["stability"],
        "oos_success_rate": wf["oos_rate"],
        "monte_carlo_pvalue": mc_p,
        "robustness": robustness,
        "stability": float(wf["stability"]),
    }

    reasons: list[str] = []
    checks = [
        (n < g["min_evaluated"], "min_evaluated"),
        (win_rate < g["min_success_rate"], "min_success_rate"),
        (precision < g["min_precision"], "min_precision"),
        (recall < g["min_recall"], "min_recall"),
        (f1 < g["min_f1"], "min_f1"),
        (expectancy < g["min_expectancy"], "min_expectancy"),
        (profit_factor < g["min_profit_factor"], "min_profit_factor"),
        (sharpe < g["min_sharpe"], "min_sharpe"),
        (mdd > g["max_drawdown"], "max_drawdown"),
        (win_rate < g["min_win_rate"], "min_win_rate"),
        (wf["stability"] < g["min_stability"], "walk_forward_stability"),
        (ci_lo < g["min_ci_low"], "confidence_interval"),
        (mc_p > g["monte_carlo_p"], "monte_carlo"),
    ]
    for failed, name in checks:
        if failed:
            reasons.append(name)

    quality = (
        0.25 * min(1.0, max(0.0, (win_rate - 0.5) * 4.0))
        + 0.20 * min(1.0, n / 80.0)
        + 0.20 * wf["stability"]
        + 0.15 * min(1.0, max(0.0, sharpe) / 1.5)
        + 0.10 * min(1.0, profit_factor / 2.0)
        + 0.10 * robustness
    )
    quality = float(max(0.0, min(1.0, quality)))
    if quality < g["min_quality_score"]:
        reasons.append("min_quality_score")

    rare_rejected = n < int(g.get("min_occurrences_rank", g["min_evaluated"]))
    if rare_rejected:
        reasons.append("rare_pattern")

    approved = len(reasons) == 0
    soft_promoted = (not rare_rejected) and (
        approved
        or (
            n >= g["soft_min_evaluated"]
            and win_rate >= g["soft_min_success"]
            and quality >= g["soft_min_quality"]
            and expectancy >= g["soft_min_expectancy"]
            and profit_factor >= 1.0
            and mc_p <= 0.35
        )
    )

    return {
        "evaluated": n,
        "approved": approved,
        "soft_promoted": soft_promoted,
        "reject_reasons": reasons,
        "metrics": metrics,
        "quality_score": quality,
        "gates": g,
        "rare_rejected": rare_rejected,
    }


def gate_pattern(
    forward_returns: np.ndarray,
    *,
    bias: str = "neutral",
    gates: dict[str, float] | None = None,
    timeframe: str | None = None,
) -> dict[str, Any]:
    """Alias used by discovery pipeline."""
    return evaluate_validation_metrics(
        forward_returns, bias=bias, gates=gates, timeframe=timeframe
    )


def confirm_htf_bias(
    *,
    bias: str,
    htf_bias_values: np.ndarray | None,
    htf_chart_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    """Require directional agreement with higher-TF bias when HTF context exists.

    Returns ``{confirmed: bool|None, mean_htf_bias, mean_chart_score}``.
    ``confirmed is None`` means HTF context unavailable (do not block).
    """
    if bias not in {"bullish", "bearish"}:
        return {"confirmed": None, "mean_htf_bias": None, "mean_chart_score": None, "skipped": True}

    mean_bias = None
    mean_chart = None
    if htf_bias_values is not None and len(htf_bias_values):
        vals = np.asarray(htf_bias_values, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals):
            mean_bias = float(np.mean(vals))
    if htf_chart_scores is not None and len(htf_chart_scores):
        vals = np.asarray(htf_chart_scores, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals):
            mean_chart = float(np.mean(vals))

    if mean_bias is None and mean_chart is None:
        return {"confirmed": None, "mean_htf_bias": None, "mean_chart_score": None, "skipped": True}

    # Prefer pat_bias; fall back to chart_pattern_score sign
    signal = mean_bias if mean_bias is not None else mean_chart
    if bias == "bullish":
        ok = signal is not None and signal >= 0
    else:
        ok = signal is not None and signal <= 0
    return {
        "confirmed": bool(ok),
        "mean_htf_bias": mean_bias,
        "mean_chart_score": mean_chart,
        "skipped": False,
    }
