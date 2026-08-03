"""Explainable AI for Engine 4: SHAP, permutation importance, feature stability."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def _tree_feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    imp = getattr(model, "feature_importances_", None)
    if imp is None and hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            imp = getattr(step, "feature_importances_", None)
            if imp is not None:
                break
    if imp is None and hasattr(model, "estimators_"):
        # VotingClassifier / Bagging: average tree importances when available
        parts: list[np.ndarray] = []
        ests = model.estimators_
        if isinstance(ests, dict):
            ests = list(ests.values())
        for est in ests:
            ei = getattr(est, "feature_importances_", None)
            if ei is not None and len(ei) == len(feature_names):
                parts.append(np.asarray(ei, dtype=float))
        if parts:
            imp = np.mean(np.vstack(parts), axis=0)
    if imp is None and hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 2:
            imp = np.mean(np.abs(coef), axis=0)
        else:
            imp = np.abs(coef)
    if imp is None:
        return {}
    imp = np.asarray(imp, dtype=float)
    if len(imp) != len(feature_names):
        return {}
    total = float(np.sum(np.abs(imp))) or 1.0
    return {c: float(v / total) for c, v in zip(feature_names, imp)}


def _unwrap_estimator(model: Any) -> Any:
    est = model
    if hasattr(est, "named_steps"):
        est = list(est.named_steps.values())[-1]
    if hasattr(est, "calibrated_classifiers_"):
        # CalibratedClassifierCV — use first underlying
        try:
            est = est.calibrated_classifiers_[0].estimator
        except Exception:
            pass
    if hasattr(est, "base_estimator") and est.__class__.__name__.startswith("Calibrated"):
        est = getattr(est, "estimator", est)
    return est


def _rank_shap_array(
    mean_abs: np.ndarray, feature_names: list[str], *, method: str, n_rows: int
) -> dict[str, Any]:
    if len(mean_abs) != len(feature_names):
        return {"enabled": False, "reason": "shape_mismatch"}
    total = float(np.sum(mean_abs)) or 1.0
    ranked = sorted(
        ((c, float(v / total)) for c, v in zip(feature_names, mean_abs)),
        key=lambda kv: -kv[1],
    )
    return {
        "enabled": True,
        "method": method,
        "n_rows": int(n_rows),
        "top": [{"feature": c, "shap_share": round(v, 6)} for c, v in ranked[:20]],
    }


def _mean_abs_shap(values: Any) -> np.ndarray:
    if isinstance(values, list):
        arr = np.mean([np.abs(v) for v in values], axis=0)
    else:
        arr = np.abs(values)
    if getattr(arr, "ndim", 1) == 3:
        arr = np.mean(arr, axis=0)
    return np.mean(arr, axis=0)


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
        "top": [
            {"feature": c, "importance": round(v, 6), "std": round(stds.get(c, 0.0), 6)}
            for c, v in ranked[:20]
        ],
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
    """SHAP with Tree / Linear / auto Explainer fallbacks (Voting unwrap)."""
    try:
        import shap  # type: ignore
    except Exception:
        # Coefficient / tree importance pseudo-SHAP so reports are never empty.
        pseudo = _tree_feature_importance(model, feature_names)
        if not pseudo:
            return {"enabled": False, "reason": "shap_not_installed"}
        ranked = sorted(pseudo.items(), key=lambda kv: -kv[1])
        return {
            "enabled": True,
            "method": "pseudo_importance_no_shap_pkg",
            "n_rows": int(min(len(X), max_rows)),
            "top": [{"feature": c, "shap_share": round(v, 6)} for c, v in ranked[:20]],
        }

    n = len(X)
    if n < 30 or len(feature_names) == 0:
        return {"enabled": False, "reason": "too_small"}
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    # Keep SHAP fast on large TF samples
    use_rows = min(int(max_rows), 400, n)
    if n > use_rows:
        idx = np.sort(rng.choice(n, size=use_rows, replace=False))
    Xs = X[idx]
    errors: list[str] = []

    estimator = _unwrap_estimator(model)
    candidates: list[Any] = [estimator]
    # VotingClassifier: try each named estimator
    if hasattr(estimator, "estimators_"):
        ests = estimator.estimators_
        if isinstance(ests, (list, tuple)):
            candidates.extend(list(ests))
        elif isinstance(ests, dict):
            candidates.extend(list(ests.values()))
    if hasattr(estimator, "named_estimators_"):
        try:
            candidates.extend(list(estimator.named_estimators_.values()))
        except Exception:
            pass

    # 1) TreeExplainer on each candidate
    for cand in candidates:
        try:
            explainer = shap.TreeExplainer(cand)
            values = explainer.shap_values(Xs)
            out = _rank_shap_array(
                _mean_abs_shap(values),
                feature_names,
                method=f"TreeExplainer:{cand.__class__.__name__}",
                n_rows=len(idx),
            )
            if out.get("enabled"):
                return out
        except Exception as exc:
            errors.append(f"tree:{cand.__class__.__name__}:{str(exc)[:80]}")

    # 2) LinearExplainer for linear models
    for cand in candidates:
        if not hasattr(cand, "coef_"):
            continue
        try:
            # Background: small sample
            bg_n = min(80, len(Xs))
            bg = Xs[:bg_n]
            explainer = shap.LinearExplainer(cand, bg)
            values = explainer.shap_values(Xs)
            out = _rank_shap_array(
                _mean_abs_shap(values),
                feature_names,
                method=f"LinearExplainer:{cand.__class__.__name__}",
                n_rows=len(idx),
            )
            if out.get("enabled"):
                return out
        except Exception as exc:
            errors.append(f"linear:{cand.__class__.__name__}:{str(exc)[:80]}")

    # 3) shap.Explainer auto (small sample)
    try:
        sample = Xs[: min(120, len(Xs))]
        explainer = shap.Explainer(estimator.predict if hasattr(estimator, "predict") else model.predict, sample)
        values = explainer(sample)
        raw = getattr(values, "values", values)
        out = _rank_shap_array(
            _mean_abs_shap(raw),
            feature_names,
            method="Explainer_auto",
            n_rows=len(sample),
        )
        if out.get("enabled"):
            return out
    except Exception as exc:
        errors.append(f"auto:{str(exc)[:100]}")

    # 4) Pseudo-SHAP from native importances / coefficients
    pseudo = _tree_feature_importance(model, feature_names)
    if not pseudo:
        pseudo = _tree_feature_importance(estimator, feature_names)
    if pseudo:
        ranked = sorted(pseudo.items(), key=lambda kv: -kv[1])
        return {
            "enabled": True,
            "method": "pseudo_importance_fallback",
            "n_rows": int(len(idx)),
            "top": [{"feature": c, "shap_share": round(v, 6)} for c, v in ranked[:20]],
            "fallback_errors": errors[:6],
        }

    return {
        "enabled": False,
        "reason": "all_shap_methods_failed",
        "error": "; ".join(errors[:4])[:300],
    }


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
    }


def build_explainability_report(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    fold_top_features: list[list[str]] | None = None,
    cfg: dict[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Combined tree/permutation/SHAP + fold stability consensus."""
    cfg = cfg or {}
    tree_map = _tree_feature_importance(model, feature_names)
    tree_top = sorted(tree_map.items(), key=lambda kv: -kv[1])[:20]

    perm: dict[str, Any] = {"enabled": False}
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

    shap_rep: dict[str, Any] = {"enabled": False}
    warnings: list[str] = []
    if bool(cfg.get("shap_enabled", True)):
        shap_rep = compute_shap_importance(
            model,
            X,
            feature_names,
            seed=seed,
            max_rows=int(cfg.get("shap_max_rows", 600)),
        )
        if bool(cfg.get("shap_required_for_report", False)) and not shap_rep.get("enabled"):
            warnings.append(f"shap_required_missing:{shap_rep.get('reason')}")

    stability = feature_stability_across_folds(
        fold_top_features or [],
        min_frac=float(cfg.get("feature_stability_min_frac", 0.5)),
    )

    scores: dict[str, float] = {}
    for c, v in tree_top:
        scores[c] = scores.get(c, 0.0) + 0.35 * float(v)
    if perm.get("enabled"):
        for row in perm.get("top") or []:
            scores[row["feature"]] = scores.get(row["feature"], 0.0) + 0.35 * max(
                0.0, float(row["importance"])
            )
    if shap_rep.get("enabled"):
        for row in shap_rep.get("top") or []:
            scores[row["feature"]] = scores.get(row["feature"], 0.0) + 0.30 * float(row["shap_share"])
    consensus = sorted(scores.items(), key=lambda kv: -kv[1])[:20]

    unstable = set(stability.get("overfit_risk_features") or [])
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
            f"تفسير الميزات · SHAP={'نعم' if shap_rep.get('enabled') else 'لا'}"
            f" ({shap_rep.get('method', '-')}) · "
            f"Permutation={'نعم' if perm.get('enabled') else 'لا'} · "
            + (
                f"استقرار الميزات Jaccard={stability.get('mean_jaccard')} · "
                f"مستقرة={len(stability.get('stable_features') or [])} · "
                f"غير مستقرة={len(stability.get('unstable_features') or [])}"
                if stability.get("enabled")
                else ""
            )
        ),
    }
