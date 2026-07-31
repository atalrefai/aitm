"""Financial-objective model selection: Nested HP vs Model Zoo resolution (v16)."""

from __future__ import annotations

from typing import Any

import numpy as np


def financial_proxy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    min_directional: int = 8,
) -> float:
    """Leak-safe proxy when trade returns are unavailable.

    Rewards directional accuracy edge and trade balance; penalizes near-chance
    and extreme class collapse.
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
    acc = float(np.mean(yt == yp)) if len(yt) else 0.0
    edge = acc - 0.50
    # Prefer some activity but not spam
    trade_rate = float(np.mean(yp != 0))
    activity = 0.05 - abs(trade_rate - 0.12) * 0.2
    # Class collapse penalty
    if len(np.unique(yp)) < 2:
        edge -= 0.08
    return float(edge + activity)


def score_model_on_inner_val(
    model: Any,
    X_va: np.ndarray,
    y_va: np.ndarray,
) -> float:
    try:
        pred = model.predict(X_va)
        return financial_proxy_score(y_va, pred)
    except Exception:
        return -1.0


def resolve_zoo_vs_nested(
    *,
    nested_meta: dict[str, Any],
    zoo_meta: dict[str, Any],
    current_model_name: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick deploy family when Zoo and Nested disagree.

    Preference order when ``resolve_on_financial_proxy``:
      1) Zoo winner if its score ≥ nested score − tolerance
      2) Else nested family mapped to baseline
      3) Soft-vote / ensemble if ``prefer_ensemble_on_conflict``
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
    }
    if not conflict:
        return decision

    tol = float(cfg.get("family_conflict_score_tol", 0.01))
    prefer_ens = bool(cfg.get("prefer_ensemble_on_conflict", True))
    allow_switch = bool(cfg.get("nested_hp_allow_family_switch", True))

    from atis.engines.engine4_training.model_zoo import map_winner_to_baseline

    if prefer_ens and conflict:
        decision["selected_family"] = "soft_vote"
        decision["selected_baseline"] = "ensemble"
        decision["reason"] = "ensemble_on_zoo_nested_conflict"
        return decision

    if zoo_score >= nested_score - tol:
        decision["selected_family"] = zoo_win
        decision["selected_baseline"] = map_winner_to_baseline(zoo_win)
        decision["reason"] = "prefer_zoo_financial_proxy"
    elif allow_switch and nested_fam:
        decision["selected_family"] = nested_fam
        decision["selected_baseline"] = map_winner_to_baseline(nested_fam)
        decision["reason"] = "prefer_nested_financial_proxy"
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
