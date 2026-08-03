"""Financial-objective model selection: Nested HP vs Model Zoo resolution (v16+).

Phase A (NA-QL): quality_compound_score ranks by WR + F1 + expectancy with
explicit saturation / inflation penalties — anti-cosmetic Acc.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _macro_f1_directional(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Simple macro-F1 over {-1,0,1} without sklearn (HPO inner loop)."""
    labels = (-1, 0, 1)
    f1s: list[float] = []
    for lab in labels:
        tp = float(np.sum((y_pred == lab) & (y_true == lab)))
        fp = float(np.sum((y_pred == lab) & (y_true != lab)))
        fn = float(np.sum((y_pred != lab) & (y_true == lab)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec <= 0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def financial_proxy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    min_directional: int = 8,
    unit_cost: float = 0.00025,
    target_trade_rate: float = 0.10,
    max_trade_rate: float | None = None,
) -> float:
    """Leak-safe proxy when trade returns are unavailable.

    Rewards directional accuracy / win-rate edge / F1 and balanced activity;
    subtracts a friction proxy per trade so HPO prefers expectancy that can cover cost.
    Quality-first: strong penalty when trade_rate pegs near/above the target band.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 20:
        return -1.0
    # Map to directional-only evaluation when both have direction
    mask = (y_true != 0) | (y_pred != 0)
    if int(mask.sum()) < min_directional:
        return -0.5
    yt, yp = y_true[mask], y_pred[mask]
    traded = yp != 0
    n_tr = int(traded.sum())
    if n_tr < min_directional:
        return -0.45
    # +1 correct direction, -1 wrong (on traded bars only)
    signed = np.where(yt[traded] == yp[traded], 1.0, -1.0)
    gross = float(np.mean(signed))
    win_rate = float(np.mean(yt[traded] == yp[traded]))
    f1 = _macro_f1_directional(y_true, y_pred)
    # Convert to rough return units and subtract round-trip friction
    expectancy = 0.0015 * gross - float(unit_cost)
    trade_rate = float(np.mean(y_pred != 0))
    # Prefer slightly below target (headroom) rather than matching the hard ceiling.
    preferred = max(0.02, float(target_trade_rate) * 0.85)
    activity = 0.04 - abs(trade_rate - preferred) * 0.30
    if trade_rate > 0.14:
        activity -= 0.08
    # Penalize fill-to-cap / pegged rates (saturation smell in HPO objective)
    if trade_rate >= max(0.08, float(target_trade_rate) * 1.15):
        activity -= 0.10
    if trade_rate >= max(0.10, float(target_trade_rate) * 1.35):
        activity -= 0.08
    rate_cap = float(max_trade_rate) if max_trade_rate and max_trade_rate > 0 else 0.0
    if rate_cap > 0 and trade_rate >= rate_cap * 0.90:
        activity -= 0.12
    if len(np.unique(yp[traded])) < 2:
        expectancy -= 0.08
    # Explicit win-rate + F1 lift for Acc/F1/WinRate health panel goals.
    quality = 0.06 * (win_rate - 0.50) + 0.04 * (f1 - 0.33)
    return float(expectancy + activity + quality)


def quality_compound_score(
    *,
    win_rate: float,
    f1: float,
    expectancy: float,
    trade_rate: float = 0.0,
    trade_sharpe_raw: float | None = None,
    target_trade_rate: float = 0.08,
    max_trade_rate: float = 0.12,
    inflated: bool = False,
    trade_rate_pegged: bool = False,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Rank models/policies by quality compound (Phase A NA-QL).

    Primary terms: win_rate + F1 + expectancy (+ optional trade-level Sharpe).
    Penalties: trade_rate peg/saturation and metric inflation.
    Does **not** reward filling the trade-rate ceiling.
    """
    w = {
        "win_rate": 1.0,
        "f1": 0.85,
        "expectancy": 1.15,
        "trade_sharpe": 0.35,
    }
    if weights:
        w.update({k: float(v) for k, v in weights.items()})

    wr = float(win_rate or 0.0)
    f1v = float(f1 or 0.0)
    exp = float(expectancy or 0.0)
    rate = float(trade_rate or 0.0)
    tsr = float(trade_sharpe_raw) if trade_sharpe_raw is not None else None

    raw = (
        w["win_rate"] * (wr - 0.50)
        + w["f1"] * (f1v - 0.33)
        + w["expectancy"] * (exp * 100.0)  # scale ~bps-ish edge into score units
    )
    if tsr is not None:
        raw += w["trade_sharpe"] * max(-0.5, min(tsr, 1.5))

    penalties: list[str] = []
    penalty = 0.0
    soft_cap = max(0.05, float(target_trade_rate) * 1.20)
    soft_cap = min(soft_cap, float(max_trade_rate) * 0.85) if max_trade_rate > 0 else soft_cap
    if rate > soft_cap:
        penalty += 0.35 * (rate / max(soft_cap, 1e-6))
        penalties.append("trade_rate_above_soft_cap")
    if max_trade_rate > 0 and rate >= float(max_trade_rate) * 0.90:
        penalty += 0.55
        penalties.append("trade_rate_near_hard_cap")
    if trade_rate_pegged:
        penalty += 0.70
        penalties.append("trade_rate_pegged")
    if inflated:
        penalty += 0.80
        penalties.append("metric_inflation")
    if exp < 0:
        penalty += 0.40
        penalties.append("negative_expectancy")
    if wr < 0.48 and rate > 0.02:
        penalty += 0.25
        penalties.append("sub_coin_win_rate")

    score = float(raw - penalty)
    return {
        "score": round(score, 6),
        "raw": round(float(raw), 6),
        "penalty": round(float(penalty), 6),
        "penalties": penalties,
        "components": {
            "win_rate": round(wr, 4),
            "f1": round(f1v, 4),
            "expectancy": round(exp, 8),
            "trade_rate": round(rate, 4),
            "trade_sharpe_raw": round(tsr, 6) if tsr is not None else None,
        },
        "weights": w,
    }


def score_model_on_inner_val(
    model: Any,
    X_va: np.ndarray,
    y_va: np.ndarray,
    *,
    target_trade_rate: float = 0.10,
    max_trade_rate: float | None = None,
) -> float:
    try:
        pred = model.predict(X_va)
        return financial_proxy_score(
            y_va,
            pred,
            target_trade_rate=target_trade_rate,
            max_trade_rate=max_trade_rate,
        )
    except Exception:
        return -1.0


_FAMILY_COMPLEXITY: dict[str, int] = {
    "logistic": 0,
    "lightgbm": 1,
    "lgbm": 1,
    "lgb": 1,
    "hist_gbm": 1,
    "xgboost": 2,
    "catboost": 2,
    "random_forest": 2,
    "rf": 2,
    "extra_trees": 2,
    "soft_vote": 3,
    "ensemble": 3,
}


def _family_complexity(name: str) -> int:
    return int(_FAMILY_COMPLEXITY.get(str(name or "").lower(), 1))


def _zoo_nested_scores_comparable(nested_score: float, zoo_score: float) -> bool:
    """Nested expectancy_cost (~negative) vs zoo acc-edge (~[−0.5,0.5]) are not comparable."""
    if nested_score <= -0.15 and zoo_score >= 0.0:
        return False
    if nested_score * zoo_score < 0 and abs(nested_score - zoo_score) > 0.35:
        return False
    return True


def resolve_zoo_vs_nested(
    *,
    nested_meta: dict[str, Any],
    zoo_meta: dict[str, Any],
    current_model_name: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick deploy family when Zoo and Nested disagree.

    Preference order:
      1) Soft-vote / ensemble if ``prefer_ensemble_on_conflict``
      2) Under regularize / prefer-simpler: nested when zoo is higher complexity
         (avoids M1 closed-loop: LGB knobs applied but Zoo RF still wins)
      3) Zoo winner only when scores are on a comparable scale and zoo ≥ nested − tol
      4) Else nested family mapped to baseline
    """
    cfg = cfg or {}
    nested_fam = str((nested_meta or {}).get("best_family") or "")
    zoo_win = str((zoo_meta or {}).get("winner") or "")
    nested_score = float((nested_meta or {}).get("best_score") or -1e9)
    zoo_rank = list((zoo_meta or {}).get("ranking") or [])
    zoo_score = float(zoo_rank[0].get("score", -1e9)) if zoo_rank else -1e9

    conflict = bool(nested_fam and zoo_win and nested_fam != zoo_win)
    decision = {
        "conflict": conflict,
        "nested_family": nested_fam or None,
        "zoo_winner": zoo_win or None,
        "nested_score": round(nested_score, 6) if nested_meta else None,
        "zoo_score": round(zoo_score, 6) if zoo_meta else None,
        "selected_family": current_model_name,
        "selected_baseline": current_model_name,
        "reason": "no_conflict",
        "scores_comparable": _zoo_nested_scores_comparable(nested_score, zoo_score)
        if conflict
        else None,
    }
    if not conflict:
        return decision

    tol = float(cfg.get("family_conflict_score_tol", 0.01))
    # Default OFF: pick a single family by financial proxy instead of always ensembling.
    prefer_ens = bool(cfg.get("prefer_ensemble_on_conflict", False))
    allow_switch = bool(cfg.get("nested_hp_allow_family_switch", True))
    prefer_simpler = bool(
        cfg.get("prefer_simpler_within_epsilon", False)
        or cfg.get("force_regularize_hp", False)
        or cfg.get("regularize_capacity", False)
        or cfg.get("prefer_nested_on_capacity_conflict", False)
    )

    from atis.engines.engine4_training.model_zoo import map_winner_to_baseline

    if prefer_ens and conflict:
        decision["selected_family"] = "soft_vote"
        decision["selected_baseline"] = "ensemble"
        decision["reason"] = "ensemble_on_zoo_nested_conflict"
        return decision

    nest_c = _family_complexity(map_winner_to_baseline(nested_fam) if nested_fam else nested_fam)
    # Also rank by raw zoo family name (random_forest vs mapped rf).
    zoo_c = max(_family_complexity(zoo_win), _family_complexity(map_winner_to_baseline(zoo_win)))
    if prefer_simpler and zoo_c > nest_c and nested_fam:
        decision["selected_family"] = nested_fam
        decision["selected_baseline"] = map_winner_to_baseline(nested_fam)
        decision["reason"] = "prefer_nested_simpler_under_regularize"
        return decision

    comparable = bool(decision.get("scores_comparable"))
    if comparable and zoo_score >= nested_score - tol:
        decision["selected_family"] = zoo_win
        decision["selected_baseline"] = map_winner_to_baseline(zoo_win)
        decision["reason"] = "prefer_zoo_financial_proxy"
    elif allow_switch and nested_fam:
        decision["selected_family"] = nested_fam
        decision["selected_baseline"] = map_winner_to_baseline(nested_fam)
        decision["reason"] = (
            "prefer_nested_incomparable_scores" if not comparable else "prefer_nested_financial_proxy"
        )
    else:
        decision["reason"] = "keep_current_on_conflict"
    return decision


def trade_level_sharpe(trade_returns: np.ndarray) -> dict[str, float]:
    """Unannualized trade Sharpe + simple diagnostics (anti-inflation view)."""
    x = np.asarray(trade_returns, dtype=float)
    x = x[np.isfinite(x) & (x != 0)]
    if len(x) < 3:
        return {
            "trade_sharpe_raw": 0.0,
            "mean_trade": 0.0,
            "std_trade": 0.0,
            "n_trades": float(len(x)),
            "pct_positive": 0.0,
        }
    mu = float(np.mean(x))
    sig = float(np.std(x, ddof=0))
    raw = mu / sig if sig > 1e-12 else 0.0
    return {
        "trade_sharpe_raw": round(raw, 6),
        "mean_trade": round(mu, 8),
        "std_trade": round(sig, 8),
        "n_trades": float(len(x)),
        "pct_positive": round(float(np.mean(x > 0)), 4),
    }


def expectancy_covers_cost(
    expectancy: float,
    *,
    spread_pips: float,
    slippage_pips: float,
    pip_size: float,
    close_price: float,
    cost_multiple: float = 1.0,
) -> tuple[bool, dict[str, float]]:
    """Gate: mean trade expectancy must exceed estimated round-trip friction × k."""
    rt_cost = (float(spread_pips) + 2.0 * float(slippage_pips)) * float(pip_size)
    # Convert absolute price cost to approximate return at reference price
    ref = max(float(close_price), 1e-9)
    cost_ret = rt_cost / ref
    need = float(cost_multiple) * cost_ret
    ok = float(expectancy) > need
    return ok, {
        "expectancy": float(expectancy),
        "est_cost_return": round(cost_ret, 8),
        "required": round(need, 8),
        "cost_multiple": float(cost_multiple),
        "covers": ok,
    }
