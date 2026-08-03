"""Nested HP search, fold eligibility, policy consensus, iterative retrain helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


def fold_eligible_for_selection(
    *,
    n_val_trades: float,
    val_sharpe: float,
    timeframe: str,
    cfg: dict[str, Any] | None = None,
    n_val_bars: int = 0,
) -> bool:
    """Starved Val folds must not drive best-fold / deploy window selection."""
    from atis.engines.engine4_training.data_quality_gate import fold_has_min_val_liquidity

    if not fold_has_min_val_liquidity(
        n_val_trades, timeframe=timeframe, cfg=cfg, n_val_bars=n_val_bars
    ):
        return False
    # Extremely negative Val Sharpe with no liquidity signal still excluded above;
    # allow mild negative if liquid enough for diagnostics.
    _ = val_sharpe
    return True


def select_best_liquid_fold(
    fold_metrics: list[dict[str, Any]],
    *,
    timeframe: str,
    cfg: dict[str, Any] | None = None,
) -> tuple[int, float, dict[str, Any] | None]:
    """Return (fold_i, score, fold_row) among liquid folds only; (-1, -inf, None) if none."""
    best_i = -1
    best_score = -1e18
    best_row: dict[str, Any] | None = None
    for row in fold_metrics:
        n_vt = float(row.get("n_val_trades", 0.0) or 0.0)
        n_vb = int(row.get("n_validation", 0) or 0)
        score = float(row.get("val_sharpe", 0.0) or 0.0)
        if not fold_eligible_for_selection(
            n_val_trades=n_vt,
            val_sharpe=score,
            timeframe=timeframe,
            cfg=cfg,
            n_val_bars=n_vb,
        ):
            continue
        if score > best_score:
            best_score = score
            best_i = int(row.get("fold", -1))
            best_row = row
    return best_i, best_score, best_row


def policy_consensus_ok(
    fold_metrics: list[dict[str, Any]],
    *,
    timeframe: str,
    cfg: dict[str, Any] | None = None,
    min_agree_folds: int | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Freeze trade policy only if ≥K liquid folds share similar thresholds."""
    cfg = cfg or {}
    need = int(min_agree_folds if min_agree_folds is not None else cfg.get("policy_min_agree_folds", 3))
    liquid = []
    for row in fold_metrics:
        n_vt = float(row.get("n_val_trades", 0.0) or 0.0)
        n_vb = int(row.get("n_validation", 0) or 0)
        if fold_eligible_for_selection(
            n_val_trades=n_vt,
            val_sharpe=float(row.get("val_sharpe", 0.0) or 0.0),
            timeframe=timeframe,
            cfg=cfg,
            n_val_bars=n_vb,
        ):
            liquid.append(row)
    meta = {
        "n_liquid_folds": len(liquid),
        "n_folds": len(fold_metrics),
        "min_agree_folds": need,
        "consensus": False,
    }
    if len(liquid) < need:
        meta["reason"] = "insufficient_liquid_folds"
        return False, meta

    thr = np.array([float((r.get("policy") or {}).get("decision_threshold", 0.55)) for r in liquid])
    edge = np.array([float((r.get("policy") or {}).get("directional_edge", 0.15)) for r in liquid])
    # Count folds within tolerance of median
    med_t, med_e = float(np.median(thr)), float(np.median(edge))
    tol_t = float(cfg.get("policy_consensus_thr_tol", 0.04))
    tol_e = float(cfg.get("policy_consensus_edge_tol", 0.05))
    agree = int(np.sum((np.abs(thr - med_t) <= tol_t) & (np.abs(edge - med_e) <= tol_e)))
    meta["agree_folds"] = agree
    meta["median_threshold"] = med_t
    meta["median_edge"] = med_e
    meta["consensus"] = agree >= need
    if not meta["consensus"]:
        meta["reason"] = "policy_dispersion"
    return bool(meta["consensus"]), meta


def _sample_lgb_params(rng: np.random.Generator, base: dict[str, Any], *, regularize: bool) -> dict[str, Any]:
    depth_lo, depth_hi = (2, 4) if regularize else (3, 5)
    leaves_lo, leaves_hi = (8, 16) if regularize else (12, 31)
    child_lo, child_hi = (120, 280) if regularize else (60, 200)
    # Honour self-opt / user capacity floors so nested HP cannot undo regularization.
    try:
        cap_depth = int(base.get("lgb_max_depth", depth_hi) or depth_hi)
        depth_hi = min(depth_hi, max(2, cap_depth))
        depth_lo = min(depth_lo, depth_hi)
    except (TypeError, ValueError):
        pass
    try:
        floor_child = int(base.get("lgb_min_child_samples", child_lo) or child_lo)
        child_lo = max(child_lo, floor_child)
        child_hi = max(child_hi, child_lo)
    except (TypeError, ValueError):
        pass
    lam_choices = [2.0, 3.5, 5.0, 6.5, 8.0] if regularize else [1.5, 2.5, 3.5, 5.0]
    try:
        floor_lam = float(base.get("lgb_reg_lambda", 0) or 0)
        if floor_lam > 0:
            lam_choices = [x for x in lam_choices if x + 1e-9 >= floor_lam] or [floor_lam]
    except (TypeError, ValueError):
        pass
    col_choices = [0.4, 0.48, 0.55, 0.65]
    try:
        cap_col = float(base.get("lgb_colsample", 0) or 0)
        if 0 < cap_col < 1:
            col_choices = [x for x in col_choices if x <= cap_col + 1e-9] or [cap_col]
    except (TypeError, ValueError):
        pass
    return {
        "lgb_estimators": int(rng.choice([120, 160, 200, 240, 280])),
        "lgb_learning_rate": float(rng.choice([0.018, 0.022, 0.028, 0.035])),
        "lgb_max_depth": int(rng.integers(depth_lo, depth_hi + 1)),
        "lgb_num_leaves": int(rng.integers(leaves_lo, leaves_hi + 1)),
        "lgb_min_child_samples": int(rng.integers(child_lo, child_hi + 1)),
        "lgb_subsample": float(rng.choice([0.65, 0.7, 0.75, 0.8])),
        "lgb_colsample": float(rng.choice(col_choices)),
        "lgb_reg_alpha": float(rng.choice([0.3, 0.5, 0.85, 1.2])),
        "lgb_reg_lambda": float(rng.choice(lam_choices)),
        "lgb_early_stopping": bool(base.get("lgb_early_stopping", True)),
        "lgb_early_stopping_rounds": int(
            max(30, int(base.get("lgb_early_stopping_rounds", 50)) + (20 if regularize else 0))
        ),
    }


def nested_hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
    *,
    base_cfg: dict[str, Any],
    timeframe: str,
    seed: int = 42,
    n_trials: int | None = None,
    build_model_fn: Callable[..., Any] | None = None,
    fit_fn: Callable[..., Any] | None = None,
    score_fn: Callable[[Any, np.ndarray, np.ndarray], float] | None = None,
    objective: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Search HP on train-only chronological inner split (no Test leakage).

    Objectives:
      - ``accuracy`` (default): inner-val accuracy − 0.50
      - ``financial_proxy``: accuracy edge + mild class-balance reward − depth penalty
    Financial Sharpe on true trade returns is applied by the caller when available.
    """
    from atis.engines.engine4_training import build_model, fit_classifier

    build_model_fn = build_model_fn or build_model
    fit_fn = fit_fn or fit_classifier
    cfg = dict(base_cfg)
    n = len(X_train)
    trials = int(n_trials if n_trials is not None else cfg.get("nested_hp_trials", 8))
    trials = max(1, min(trials, 24))
    if n < 80 or not bool(cfg.get("nested_hp_search", True)):
        return cfg, {"enabled": False, "reason": "disabled_or_too_small", "trials": 0}

    cut = max(40, int(n * 0.82))
    if cut >= n - 15:
        return cfg, {"enabled": False, "reason": "no_inner_val", "trials": 0}

    X_tr, X_va = X_train[:cut], X_train[cut:]
    y_tr, y_va = y_train[:cut], y_train[cut:]
    w_tr = sample_weight[:cut] if sample_weight is not None else None
    rng = np.random.default_rng(seed + hash(str(timeframe)) % 10007)
    regularize = (
        str(timeframe).upper() in {"H1", "H4", "M1", "M5"}
        or bool(cfg.get("force_regularize_hp", False))
        or bool(cfg.get("nested_hp_train_val_gap_penalty", False))
        or bool(cfg.get("regularize_capacity", False))
    )
    prefer_simpler = bool(cfg.get("prefer_simpler_within_epsilon", False)) or regularize
    simpler_eps = float(cfg.get("simpler_within_epsilon", 0.004) or 0.004)
    obj = str(objective or cfg.get("nested_hp_objective", "financial_proxy")).lower()

    def _default_score(model: Any, Xv: np.ndarray, yv: np.ndarray) -> float:
        try:
            pred = model.predict(Xv)
            if obj == "accuracy":
                acc = float(np.mean(pred == yv)) if len(yv) else 0.0
                return acc - 0.50
            # financial_proxy / expectancy_cost / quality_compound: cost-aware + F1/WR
            from atis.engines.engine4_training.financial_hpo import financial_proxy_score

            return financial_proxy_score(
                yv,
                pred,
                unit_cost=float(cfg.get("nested_hp_unit_cost", 0.00025)),
                target_trade_rate=float(cfg.get("target_trade_rate", 0.08) or 0.08),
                max_trade_rate=float(cfg.get("max_fold_trade_rate", 0.12) or 0.12),
            )
        except Exception:
            return -1.0

    score_fn = score_fn or _default_score
    candidates = [dict(cfg)]  # always evaluate baseline
    for _ in range(trials - 1):
        cand = dict(cfg)
        cand.update(_sample_lgb_params(rng, cfg, regularize=regularize))
        candidates.append(cand)

    # Optional linear / HistGBM baseline comparison on same inner split
    families = ["lightgbm"]
    if bool(cfg.get("nested_hp_compare_baselines", True)):
        families.extend(["logistic"])
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: F401

            families.append("hist_gbm")
        except Exception:
            pass

    best_cfg = dict(cfg)
    best_score = -1e18
    best_complexity = 1e18
    history: list[dict[str, Any]] = []

    def _complexity(cand: dict[str, Any], fam: str) -> float:
        if fam == "logistic":
            return 0.5
        return (
            float(cand.get("lgb_max_depth", 4))
            + 0.01 * max(0.0, 8.0 - float(cand.get("lgb_reg_lambda", 1.0)))
            + 0.001 * max(0.0, 300 - float(cand.get("lgb_min_child_samples", 80)))
        )

    for fam in families:
        for i, cand in enumerate(candidates if fam == "lightgbm" else candidates[:1]):
            try:
                if fam == "hist_gbm":
                    from sklearn.ensemble import HistGradientBoostingClassifier

                    model = HistGradientBoostingClassifier(
                        max_depth=int(cand.get("lgb_max_depth", 4)),
                        learning_rate=float(cand.get("lgb_learning_rate", 0.05)),
                        max_iter=min(200, int(cand.get("lgb_estimators", 200))),
                        min_samples_leaf=int(cand.get("lgb_min_child_samples", 80)),
                        l2_regularization=float(cand.get("lgb_reg_lambda", 1.0)),
                        random_state=seed,
                    )
                    if w_tr is not None:
                        model.fit(X_tr, y_tr, sample_weight=w_tr)
                    else:
                        model.fit(X_tr, y_tr)
                    model_name = "hist_gbm"
                elif fam == "logistic":
                    model = build_model_fn("logistic", seed, cand)
                    model = fit_fn(model, X_tr, y_tr, w_tr, cfg=cand)
                    model_name = "logistic"
                else:
                    model = build_model_fn("lightgbm", seed, cand)
                    model = fit_fn(model, X_tr, y_tr, w_tr, cfg=cand)
                    model_name = "lightgbm"
                sc = float(score_fn(model, X_va, y_va))
                # Prefer stronger regularization when scores are close
                if regularize and fam == "lightgbm":
                    sc -= 0.002 * float(cand.get("lgb_max_depth", 4))
                cx = _complexity(cand, model_name)
                history.append({"family": model_name, "trial": i, "score": round(sc, 6)})
                better = sc > best_score + 1e-12
                tie_simpler = (
                    prefer_simpler
                    and abs(sc - best_score) <= simpler_eps
                    and cx < best_complexity
                )
                if better or tie_simpler:
                    if better:
                        best_score = sc
                    best_cfg = dict(cand)
                    best_cfg["_nested_model_family"] = model_name
                    best_complexity = cx
            except Exception as exc:  # pragma: no cover - defensive
                history.append({"family": fam, "trial": i, "error": str(exc)})

    meta = {
        "enabled": True,
        "trials": len(history),
        "best_score": round(best_score, 6),
        "best_family": best_cfg.get("_nested_model_family", "lightgbm"),
        "regularize": regularize,
        "prefer_simpler_within_epsilon": prefer_simpler,
        "objective": obj,
        "history": history[:30],
        "inner_train": int(cut),
        "inner_val": int(n - cut),
    }
    return best_cfg, meta


def nested_hp_across_outer_folds(
    fold_train_payloads: list[dict[str, Any]],
    *,
    base_cfg: dict[str, Any],
    timeframe: str,
    seed: int = 42,
    n_trials: int = 6,
    max_folds: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """True nested-style HP: re-search on up to ``max_folds`` outer train windows.

    Each payload: ``{"X": ndarray, "y": ndarray, "w": ndarray|None, "fold": int}``.
    Winner = median-best family/cfg across folds (stability over peak score).
    """
    if not fold_train_payloads or not bool(base_cfg.get("nested_hp_search", True)):
        return dict(base_cfg), {"enabled": False, "reason": "no_payloads"}

    selected = fold_train_payloads[: max(1, int(max_folds))]
    fold_winners: list[dict[str, Any]] = []
    cfg_votes: dict[str, list[float]] = {}
    best_cfgs: dict[str, dict[str, Any]] = {}

    for payload in selected:
        X = payload["X"]
        y = payload["y"]
        w = payload.get("w")
        fold_i = int(payload.get("fold", -1))
        cfg_i, meta_i = nested_hyperparameter_search(
            X,
            y,
            w,
            base_cfg=base_cfg,
            timeframe=timeframe,
            seed=seed + max(0, fold_i) * 17,
            n_trials=n_trials,
        )
        if not meta_i.get("enabled"):
            continue
        fam = str(meta_i.get("best_family", "lightgbm"))
        sc = float(meta_i.get("best_score", 0.0) or 0.0)
        fold_winners.append({"fold": fold_i, "family": fam, "score": sc})
        cfg_votes.setdefault(fam, []).append(sc)
        # Keep cfg with best score per family
        prev = best_cfgs.get(fam)
        if prev is None or sc > float(prev.get("_score", -1e18)):
            kept = dict(cfg_i)
            kept["_score"] = sc
            best_cfgs[fam] = kept

    if not fold_winners:
        return dict(base_cfg), {"enabled": False, "reason": "all_inner_failed", "folds_tried": len(selected)}

    # Prefer family with best median score across folds
    fam_rank = sorted(
        ((fam, float(np.median(scores)), float(np.mean(scores))) for fam, scores in cfg_votes.items()),
        key=lambda t: (t[1], t[2]),
        reverse=True,
    )
    winner_fam = fam_rank[0][0]
    best_cfg = dict(best_cfgs.get(winner_fam) or base_cfg)
    best_cfg.pop("_score", None)
    best_cfg["_nested_model_family"] = winner_fam

    meta = {
        "enabled": True,
        "mode": "nested_across_outer_folds",
        "folds_tried": len(selected),
        "fold_winners": fold_winners,
        "family_median": {f: round(m, 6) for f, m, _ in fam_rank},
        "best_family": winner_fam,
        "best_score": round(fam_rank[0][1], 6),
        "objective": str(base_cfg.get("nested_hp_objective", "financial_proxy")),
    }
    return best_cfg, meta


def dynamic_execution_costs(
    close: float,
    atr_pct: float,
    *,
    base_spread_pips: float,
    base_slippage_pips: float,
    commission_per_lot: float,
    pip_size: float,
    vol_slippage_k: float = 1.25,
) -> tuple[float, float, float]:
    """Scale spread/slippage with local volatility (ATR%)."""
    # atr_pct ~ 0.001–0.01 for gold; map to pip inflation
    vol_mult = 1.0 + float(vol_slippage_k) * max(0.0, float(atr_pct) / 0.002 - 1.0)
    vol_mult = float(np.clip(vol_mult, 0.85, 2.5))
    spread = float(base_spread_pips) * vol_mult
    slip = float(base_slippage_pips) * vol_mult
    return spread, slip, float(commission_per_lot)


def should_trigger_retrain(
    *,
    last_train_utc: str | None,
    retrain_interval_days: float,
    drift_score: float = 0.0,
    drift_threshold: float = 0.25,
    deploy_sharpe: float | None = None,
    min_deploy_sharpe: float = 0.0,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """retrain_interval_days + drift / deploy-collapse triggers."""
    now = now or datetime.now(timezone.utc)
    if drift_score >= float(drift_threshold):
        return True, "concept_drift"
    if deploy_sharpe is not None and float(deploy_sharpe) < float(min_deploy_sharpe):
        return True, "deploy_collapse"
    if not last_train_utc or retrain_interval_days <= 0:
        return False, "no_schedule"
    try:
        last = datetime.fromisoformat(str(last_train_utc).replace("Z", "+00:00"))
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        age_days = (now - last).total_seconds() / 86400.0
        if age_days >= float(retrain_interval_days):
            return True, "retrain_interval"
    except Exception:
        return True, "invalid_last_train_ts"
    return False, "ok"


def iterative_stop_decision(
    history: list[dict[str, Any]],
    *,
    kpi_ci_low: float = 1.5,
    delta: float = 0.15,
    patience: int = 2,
    max_experiments: int = 5,
) -> tuple[bool, str]:
    """Stop when CI_low stable, KPI hit, or budget exhausted."""
    if len(history) >= max_experiments:
        return True, "budget_exhausted"
    if not history:
        return False, "continue"
    last = history[-1]
    ci = float(last.get("sharpe_ci_low", last.get("ci_low", -999)) or -999)
    if ci >= float(kpi_ci_low) and str(last.get("fit_status", "")) == "balanced":
        return True, "kpi_reached"
    if len(history) >= patience + 1:
        recent = [float(h.get("sharpe_ci_low", h.get("ci_low", 0)) or 0) for h in history[-(patience + 1) :]]
        if max(recent) - min(recent) <= float(delta) and recent[-1] > 0:
            return True, "ci_stable"
    return False, "continue"


def load_experiment_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("experiments") or data if isinstance(data, list) else [])
    except Exception:
        return []


def append_experiment_log(path: Path, experiment: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_experiment_log(path)
    rows.append(experiment)
    path.write_text(json.dumps({"experiments": rows, "updated_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
