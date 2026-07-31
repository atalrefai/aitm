"""Feature intelligence: MI, correlation pruning, importance ranking."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif


def _corr_prune(X: pd.DataFrame, *, threshold: float = 0.92) -> tuple[list[str], list[tuple[str, str, float]]]:
    cols = list(X.columns)
    if len(cols) < 3:
        return cols, []
    corr = X.corr(numeric_only=True).abs()
    drop: set[str] = set()
    pairs: list[tuple[str, str, float]] = []
    for i, a in enumerate(cols):
        if a in drop or a not in corr.columns:
            continue
        for b in cols[i + 1 :]:
            if b in drop or b not in corr.columns:
                continue
            v = float(corr.loc[a, b]) if a in corr.index and b in corr.columns else 0.0
            if v >= threshold:
                drop.add(b)
                pairs.append((a, b, v))
    keep = [c for c in cols if c not in drop]
    return keep, pairs[:40]


def analyze_and_select_features(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    max_features: int = 56,
    seed: int = 42,
    corr_threshold: float = 0.92,
    sample_weight: np.ndarray | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Return selected feature names + intelligence report."""
    y_arr = np.asarray(y)
    feats = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    if not feats:
        return [], {"enabled": False, "reason": "no_numeric"}

    Xn = X[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Subsample for MI speed
    n = len(Xn)
    idx = np.arange(n)
    if n > 12000:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=12000, replace=False))
    Xi = Xn.iloc[idx]
    yi = y_arr[idx]

    mi = mutual_info_classif(Xi.values, yi, discrete_features=False, random_state=seed)
    mi_map = {c: float(v) for c, v in zip(feats, mi)}

    # Quick RF importance on same subsample
    rf = RandomForestClassifier(
        n_estimators=80,
        max_depth=5,
        min_samples_leaf=40,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    sw = sample_weight[idx] if sample_weight is not None else None
    try:
        if sw is not None:
            rf.fit(Xi.values, yi, sample_weight=sw)
        else:
            rf.fit(Xi.values, yi)
        imp = {c: float(v) for c, v in zip(feats, rf.feature_importances_)}
    except Exception:
        imp = {c: 0.0 for c in feats}

    # Combined score
    mi_max = max(mi_map.values()) if mi_map else 1.0
    imp_max = max(imp.values()) if imp else 1.0
    score = {
        c: 0.55 * (mi_map.get(c, 0.0) / max(mi_max, 1e-12))
        + 0.45 * (imp.get(c, 0.0) / max(imp_max, 1e-12))
        for c in feats
    }
    ranked = sorted(feats, key=lambda c: score[c], reverse=True)

    # Drop near-zero MI+imp
    weak = [c for c in ranked if score[c] < 0.02]
    candidates = [c for c in ranked if c not in set(weak)] or ranked
    pruned, corr_pairs = _corr_prune(Xn[candidates], threshold=corr_threshold)
    # Preserve rank order
    order = {c: i for i, c in enumerate(ranked)}
    pruned_sorted = sorted(pruned, key=lambda c: order.get(c, 10**9))
    selected = pruned_sorted[: max(8, int(max_features))]

    report = {
        "enabled": True,
        "n_input": len(feats),
        "n_selected": len(selected),
        "n_weak_dropped": len(weak),
        "n_corr_dropped": len(corr_pairs),
        "top_mutual_info": sorted(mi_map.items(), key=lambda x: -x[1])[:15],
        "top_importance": sorted(imp.items(), key=lambda x: -x[1])[:15],
        "top_combined": [(c, round(score[c], 4)) for c in selected[:15]],
        "corr_pairs_dropped": [
            {"keep": a, "drop": b, "corr": round(v, 3)} for a, b, v in corr_pairs[:15]
        ],
        "weak_features_sample": weak[:15],
    }
    return selected, report
