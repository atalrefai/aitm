"""Ensemble learning helpers for Engine 4.

References:
- Dietterich (2000) — ensemble methods survey.
- Soft voting / stacking — sklearn Meta-estimators; quant practice of blending
  heterogeneous learners to reduce variance (AQR-style model committees).
- Nested stacking only on train chronologically — no Test leakage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression


def build_soft_voting_ensemble(
    *,
    seed: int = 42,
    cfg: dict[str, Any] | None = None,
) -> Any:
    """LightGBM (if available) + HistGBM/RF + Logistic soft vote."""
    cfg = cfg or {}
    estimators: list[tuple[str, Any]] = []
    try:
        from lightgbm import LGBMClassifier

        estimators.append(
            (
                "lgbm",
                LGBMClassifier(
                    n_estimators=int(cfg.get("lgb_estimators", 160)),
                    learning_rate=float(cfg.get("lgb_learning_rate", 0.028)),
                    max_depth=int(cfg.get("lgb_max_depth", 4)),
                    num_leaves=int(cfg.get("lgb_num_leaves", 15)),
                    min_child_samples=int(cfg.get("lgb_min_child_samples", 120)),
                    subsample=float(cfg.get("lgb_subsample", 0.75)),
                    colsample_bytree=float(cfg.get("lgb_colsample", 0.55)),
                    reg_alpha=float(cfg.get("lgb_reg_alpha", 0.5)),
                    reg_lambda=float(cfg.get("lgb_reg_lambda", 3.0)),
                    random_state=seed,
                    verbosity=-1,
                ),
            )
        )
    except Exception:
        pass
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        estimators.append(
            (
                "hgb",
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
    except Exception:
        estimators.append(
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=int(cfg.get("lgb_max_depth", 5)),
                    min_samples_leaf=int(cfg.get("lgb_min_child_samples", 40)),
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
        )
    estimators.append(
        (
            "logit",
            LogisticRegression(
                max_iter=400,
                C=float(1.0 / max(cfg.get("lgb_reg_lambda", 1.0), 0.1)),
                random_state=seed,
            ),
        )
    )
    if len(estimators) < 2:
        return estimators[0][1] if estimators else LogisticRegression(max_iter=400, random_state=seed)
    return VotingClassifier(estimators=estimators, voting="soft", weights=None)


def blend_probas(
    proba_list: list[np.ndarray],
    *,
    weights: list[float] | None = None,
) -> np.ndarray:
    """Weighted average of probability matrices (same shape)."""
    if not proba_list:
        raise ValueError("empty proba_list")
    mats = [np.asarray(p, dtype=float) for p in proba_list]
    shape0 = mats[0].shape
    for m in mats:
        if m.shape != shape0:
            raise ValueError("proba shape mismatch")
    if weights is None:
        w = np.ones(len(mats), dtype=float) / len(mats)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / max(float(w.sum()), 1e-12)
    out = np.zeros_like(mats[0])
    for wi, mi in zip(w, mats):
        out += wi * mi
    return out


def ensemble_rationale() -> dict[str, str]:
    return {
        "soft_voting": (
            "Averages calibrated class probabilities across diverse learners; "
            "reduces variance vs single tree (Dietterich 2000)."
        ),
        "nested_only": (
            "Ensemble fit only on chronological train; Val tunes policy; Test untouched."
        ),
    }
