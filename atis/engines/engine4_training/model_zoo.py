"""Enterprise model zoo — multi-family compare on chronological inner split."""

from __future__ import annotations

from typing import Any

import numpy as np


def _safe_acc(model: Any, X: np.ndarray, y: np.ndarray) -> float:
    try:
        pred = model.predict(X)
        return float(np.mean(pred == y)) if len(y) else 0.0
    except Exception:
        return 0.0


def build_zoo_candidates(seed: int = 42, cfg: dict[str, Any] | None = None) -> list[tuple[str, Any]]:
    """Practical institutional subset (heavy DL optional / separate path)."""
    cfg = cfg or {}
    out: list[tuple[str, Any]] = []
    # LightGBM
    try:
        from lightgbm import LGBMClassifier

        out.append(
            (
                "lightgbm",
                LGBMClassifier(
                    n_estimators=int(cfg.get("lgb_estimators", 180)),
                    learning_rate=float(cfg.get("lgb_learning_rate", 0.028)),
                    max_depth=int(cfg.get("lgb_max_depth", 4)),
                    num_leaves=int(cfg.get("lgb_num_leaves", 15)),
                    min_child_samples=int(cfg.get("lgb_min_child_samples", 120)),
                    subsample=float(cfg.get("lgb_subsample", 0.75)),
                    colsample_bytree=float(cfg.get("lgb_colsample", 0.55)),
                    reg_alpha=float(cfg.get("lgb_reg_alpha", 0.5)),
                    reg_lambda=float(cfg.get("lgb_reg_lambda", 3.0)),
                    class_weight="balanced",
                    random_state=seed,
                    verbosity=-1,
                ),
            )
        )
    except Exception:
        pass
    # XGBoost (optional)
    try:
        from xgboost import XGBClassifier

        out.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=min(200, int(cfg.get("lgb_estimators", 180))),
                    learning_rate=float(cfg.get("lgb_learning_rate", 0.05)),
                    max_depth=int(cfg.get("lgb_max_depth", 4)),
                    subsample=0.75,
                    colsample_bytree=0.55,
                    reg_lambda=float(cfg.get("lgb_reg_lambda", 3.0)),
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=seed,
                    n_jobs=2,
                    verbosity=0,
                ),
            )
        )
    except Exception:
        pass
    # CatBoost (optional)
    try:
        from catboost import CatBoostClassifier

        out.append(
            (
                "catboost",
                CatBoostClassifier(
                    iterations=min(200, int(cfg.get("lgb_estimators", 180))),
                    depth=min(6, int(cfg.get("lgb_max_depth", 4)) + 1),
                    learning_rate=float(cfg.get("lgb_learning_rate", 0.05)),
                    loss_function="MultiClass",
                    verbose=False,
                    random_seed=seed,
                    allow_writing_files=False,
                ),
            )
        )
    except Exception:
        pass

    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression

    out.append(
        (
            "hist_gbm",
            HistGradientBoostingClassifier(
                max_depth=int(cfg.get("lgb_max_depth", 4)),
                learning_rate=float(cfg.get("lgb_learning_rate", 0.05)),
                max_iter=min(200, int(cfg.get("lgb_estimators", 200))),
                min_samples_leaf=int(cfg.get("lgb_min_child_samples", 80)),
                l2_regularization=float(cfg.get("lgb_reg_lambda", 1.0)),
                random_state=seed,
            ),
        )
    )
    regularize = bool(cfg.get("force_regularize_hp") or cfg.get("regularize_capacity"))
    rf_depth = int(cfg.get("rf_max_depth", int(cfg.get("lgb_max_depth", 6)) + (0 if regularize else 2)))
    if regularize:
        rf_depth = min(rf_depth, int(cfg.get("rf_max_depth", 4) or 4))
    rf_leaf = int(
        cfg.get(
            "rf_min_samples_leaf",
            max(8, int(cfg.get("lgb_min_child_samples", 40) // (4 if regularize else 3))),
        )
    )
    if regularize:
        rf_leaf = max(rf_leaf, 25)
    out.append(
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=int(cfg.get("rf_estimators", 180)),
                max_depth=max(2, rf_depth),
                min_samples_leaf=max(4, rf_leaf),
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
        )
    )
    out.append(
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=int(cfg.get("rf_estimators", 200)),
                max_depth=max(2, rf_depth),
                min_samples_leaf=max(4, rf_leaf),
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        )
    )
    out.append(
        (
            "logistic",
            LogisticRegression(max_iter=600, class_weight="balanced", random_state=seed),
        )
    )
    # Soft voting hybrid if ≥2 tree/boost families present
    boosters = [p for p in out if p[0] in {"lightgbm", "xgboost", "catboost", "hist_gbm"}]
    if len(boosters) >= 2:
        from sklearn.ensemble import VotingClassifier

        estimators = [(n, m) for n, m in boosters[:3]]
        out.append(("soft_vote", VotingClassifier(estimators=estimators, voting="soft")))
    return out


def compare_model_zoo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
    *,
    seed: int = 42,
    cfg: dict[str, Any] | None = None,
    max_models: int = 8,
) -> dict[str, Any]:
    """Chronological inner-split bake-off; returns ranking + winner family."""
    cfg = cfg or {}
    n = len(X_train)
    if n < 120 or not bool(cfg.get("model_zoo_enabled", True)):
        return {"enabled": False, "reason": "disabled_or_too_small", "ranking": []}

    cut = max(50, int(n * 0.82))
    if cut >= n - 20:
        return {"enabled": False, "reason": "no_inner_val", "ranking": []}

    X_tr, X_va = X_train[:cut], X_train[cut:]
    y_tr, y_va = y_train[:cut], y_train[cut:]
    w_tr = sample_weight[:cut] if sample_weight is not None else None

    ranking: list[dict[str, Any]] = []
    for name, model in build_zoo_candidates(seed=seed, cfg=cfg)[: max(3, int(max_models))]:
        try:
            # XGBoost needs 0..K labels sometimes
            y_fit = y_tr
            if name == "xgboost":
                classes = sorted(set(np.unique(y_tr).tolist()) | set(np.unique(y_va).tolist()))
                mapping = {c: i for i, c in enumerate(classes)}
                inv = {i: c for c, i in mapping.items()}
                y_mapped = np.array([mapping[int(v)] for v in y_tr])
                model.fit(X_tr, y_mapped, sample_weight=w_tr) if w_tr is not None else model.fit(X_tr, y_mapped)
                pred_i = model.predict(X_va)
                pred = np.array([inv[int(i)] for i in pred_i])
                acc = float(np.mean(pred == y_va))
            else:
                if w_tr is not None:
                    try:
                        model.fit(X_tr, y_tr, sample_weight=w_tr)
                    except TypeError:
                        model.fit(X_tr, y_tr)
                else:
                    model.fit(X_tr, y_tr)
                acc = _safe_acc(model, X_va, y_va)
            # Mild complexity penalty (stronger under regularize so logistic/LGB win close races)
            regularize = bool(cfg.get("force_regularize_hp") or cfg.get("regularize_capacity") or cfg.get("prefer_simpler_within_epsilon"))
            penalty = 0.0
            if name in {"random_forest", "extra_trees", "soft_vote"}:
                penalty = 0.02 if regularize else 0.005
            if name == "logistic":
                penalty = -0.008 if regularize else -0.002  # prefer simple if close
            ranking.append({"family": name, "inner_val_acc": round(acc, 6), "score": round(acc - penalty, 6)})
        except Exception as exc:
            ranking.append({"family": name, "error": str(exc), "score": -1.0})

    # Prefer financial proxy when enabled (v16)
    if bool(cfg.get("model_zoo_financial_proxy", True)):
        try:
            from atis.engines.engine4_training.financial_hpo import financial_proxy_score

            for row in ranking:
                if "error" in row:
                    continue
                # Re-score using stored acc as fallback; true proxy needs preds —
                # approximate: map acc edge to financial_proxy style
                acc = float(row.get("inner_val_acc") or 0)
                row["score"] = round((acc - 0.50) + (0.002 if row["family"] == "logistic" else 0.0), 6)
                row["financial_proxy"] = row["score"]
        except Exception:
            pass

    ranking = sorted(ranking, key=lambda r: float(r.get("score", -1)), reverse=True)
    winner = ranking[0]["family"] if ranking and ranking[0].get("score", -1) > 0 else "lightgbm"
    return {
        "enabled": True,
        "inner_train": int(cut),
        "inner_val": int(n - cut),
        "winner": winner,
        "ranking": ranking,
        "n_models_tried": len(ranking),
    }


def map_winner_to_baseline(winner: str) -> str:
    w = str(winner).lower()
    if w in {"lightgbm", "lgbm", "lgb"}:
        return "lightgbm"
    if w in {"random_forest", "rf", "extra_trees"}:
        return "rf"
    if w in {"logistic"}:
        return "logistic"
    if w in {"soft_vote", "ensemble", "hist_gbm", "xgboost", "catboost"}:
        return "ensemble" if w in {"soft_vote", "ensemble"} else "lightgbm"
    return "lightgbm"
