"""Explainable AI for Engine 4: SHAP, permutation importance, feature stability."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def _tree_feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    imp = getattr(model, "feature_importances_", None)
    if imp is None and hasattr(model, "named_steps"):
        # Pipeline last step
        for step in reversed(list(model.named_steps.values())):
            imp = getattr(step, "feature_importances_", None)
            if imp is not None:
                model = step
                break
    if imp is None:
        return {}
    imp = np.asarray(imp, dtype=float)
    if len(imp) != len(feature_names):
        return {}
    total = float(np.sum(np.abs(imp))) or 1.0
    return {c: float(v / total) for c, v in zip(feature_names, imp)}


def compute_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    seed: int = 42,
    n_repeats: int = 5,
    max_rows: int = 4000,
) -> dict[str, Any]:
    n = len(X)
    if n < 40 or len(feature_names) == 0:
        return {"enabled": False, "reason": "too_small"}
    idx = np.arange(n)
    if n > max_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_rows, replace=False))
    try:
        result = permutation_importance(
            model,
            X[idx],
            y[idx],
            n_repeats=max(2, int(n_repeats)),
            random_state=seed,
            scoring="accuracy",
            n_jobs=1,
        )
    except Exception as exc:
        return {"enabled": False, "error": str(exc)}

    means = {c: float(v) for c, v in zip(feature_names, result.importances_mean)}
    stds = {c: float(v) for c, v in zip(feature_names, result.importances_std)}
    ranked = sorted(means.items(), key=lambda kv: -kv[1])
    return {
        "enabled": True,
        "top": [{"feature": c, "importance": round(v, 6), "std": round(stds.get(c, 0.0), 6)} for c, v in ranked[:20]],
        "n_rows": int(len(idx)),
        "n_repeats": int(n_repeats),
    }


def compute_shap_importance(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    *,
    seed: int = 42,
    max_rows: int = 800,
) -> dict[str, Any]:
    """Optional SHAP TreeExplainer; falls back gracefully if shap is unavailable."""
    try:
        import shap  # type: ignore
    except Exception:
        return {"enabled": False, "reason": "shap_not_installed"}

    n = len(X)
    if n < 30 or len(feature_names) == 0:
        return {"enabled": False, "reason": "too_small"}
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if n > max_rows:
        idx = np.sort(rng.choice(n, size=max_rows, replace=False))
    Xs = X[idx]

    estimator = model
    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]

    try:
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(Xs)
        if isinstance(values, list):
            # Multi-class: mean abs across classes
            arr = np.mean([np.abs(v) for v in values], axis=0)
        else:
            arr = np.abs(values)
        if arr.ndim == 3:
            arr = np.mean(arr, axis=0)
        mean_abs = np.mean(arr, axis=0)
        if len(mean_abs) != len(feature_names):
            return {"enabled": False, "reason": "shape_mismatch"}
        total = float(np.sum(mean_abs)) or 1.0
        ranked = sorted(
            ((c, float(v / total)) for c, v in zip(feature_names, mean_abs)),
            key=lambda kv: -kv[1],
        )
        return {
            "enabled": True,
            "method": "TreeExplainer",
            "n_rows": int(len(idx)),
            "top": [{"feature": c, "shap_share": round(v, 6)} for c, v in ranked[:20]],
        }
    except Exception as exc:
        # KernelExplainer is too slow for production WF — skip
        return {"enabled": False, "reason": "tree_explainer_failed", "error": str(exc)[:200]}


def feature_stability_across_folds(
    fold_top_features: list[list[str]],
    *,
    min_frac: float = 0.5,
) -> dict[str, Any]:
    """Jaccard + vote frequency across fold-level top feature lists."""
    if not fold_top_features:
        return {"enabled": False, "reason": "no_folds"}
    votes: dict[str, int] = {}
    for feats in fold_top_features:
        for f in set(feats):
            votes[f] = votes.get(f, 0) + 1
    n = len(fold_top_features)
    stable = sorted(
        [f for f, c in votes.items() if (c / n) >= float(min_frac)],
        key=lambda f: (-votes[f], f),
    )
    unstable = sorted(
        [f for f, c in votes.items() if (c / n) < float(min_frac)],
        key=lambda f: (votes[f], f),
    )
    # Pairwise Jaccard among fold sets
    jaccards: list[float] = []
    sets = [set(f) for f in fold_top_features]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            if not u:
                continue
            jaccards.append(len(sets[i] & sets[j]) / len(u))
    mean_j = float(np.mean(jaccards)) if jaccards else 0.0
    return {
        "enabled": True,
        "n_folds": n,
        "mean_jaccard": round(mean_j, 4),
        "stable_features": stable[:40],
        "unstable_features": unstable[:40],
        "vote_counts": {k: votes[k] for k in stable[:20]},
        "overfit_risk_features": unstable[:15],
        "summary_ar": (
            f"استقرار الميزات Jaccard={mean_j:.2f} · مستقرة={len(stable)} · غير مستقرة={len(unstable)}"
        ),
    }


def build_explainability_report(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    fold_top_features: list[list[str]] | None = None,
    seed: int = 42,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Unified explainability payload for metrics + UI."""
    cfg = cfg or {}
    tree_imp = _tree_feature_importance(model, feature_names)
    tree_top = sorted(tree_imp.items(), key=lambda kv: -kv[1])[:20]

    perm = {"enabled": False}
    if bool(cfg.get("permutation_importance_enabled", True)):
        perm = compute_permutation_importance(
            model,
            X,
            y,
            feature_names,
            seed=seed,
            n_repeats=int(cfg.get("permutation_n_repeats", 4)),
            max_rows=int(cfg.get("permutation_max_rows", 3500)),
        )

    shap_rep = {"enabled": False}
    if bool(cfg.get("shap_enabled", True)):
        shap_rep = compute_shap_importance(
            model,
            X,
            feature_names,
            seed=seed,
            max_rows=int(cfg.get("shap_max_rows", 600)),
        )

    stability = feature_stability_across_folds(
        fold_top_features or [],
        min_frac=float(cfg.get("feature_stability_min_frac", 0.5)),
    )

    # Consensus top features across methods
    scores: dict[str, float] = {}
    for c, v in tree_top:
        scores[c] = scores.get(c, 0.0) + 0.35 * float(v)
    if perm.get("enabled"):
        for row in perm.get("top") or []:
            scores[row["feature"]] = scores.get(row["feature"], 0.0) + 0.35 * max(0.0, float(row["importance"]))
    if shap_rep.get("enabled"):
        for row in shap_rep.get("top") or []:
            scores[row["feature"]] = scores.get(row["feature"], 0.0) + 0.30 * float(row["shap_share"])
    consensus = sorted(scores.items(), key=lambda kv: -kv[1])[:20]

    unstable = set(stability.get("overfit_risk_features") or [])
    warnings: list[str] = []
    if stability.get("enabled") and float(stability.get("mean_jaccard") or 0) < 0.35:
        warnings.append("low_feature_stability_across_folds")
    risky_top = [c for c, _ in consensus[:10] if c in unstable]
    if risky_top:
        warnings.append(f"top_features_unstable:{','.join(risky_top[:5])}")

    return {
        "enabled": True,
        "tree_importance_top": [{"feature": c, "importance": round(v, 6)} for c, v in tree_top],
        "permutation": perm,
        "shap": shap_rep,
        "stability": stability,
        "consensus_top": [{"feature": c, "score": round(v, 6)} for c, v in consensus],
        "warnings": warnings,
        "summary_ar": (
            f"تفسير الميزات · SHAP={'نعم' if shap_rep.get('enabled') else 'لا'} · "
            f"Permutation={'نعم' if perm.get('enabled') else 'لا'} · "
            f"{stability.get('summary_ar') or ''}"
        ),
    }
