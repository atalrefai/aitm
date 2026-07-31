"""Label quality, noise estimation, and barrier sensitivity for Engine 4."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2 or len(y) < 20:
        return 0.5
    try:
        return float(roc_auc_score(y, scores))
    except Exception:
        return 0.5


def estimate_label_noise(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    seed: int = 42,
    n_splits: int = 3,
    max_rows: int = 8000,
) -> dict[str, Any]:
    """Proxy label-noise via disagreement of a weak linear model under purged-ish CV.

    High disagreement on directional labels suggests noisy / inconsistent barriers.
    """
    y_arr = np.asarray(y).astype(float)
    if isinstance(X, pd.DataFrame):
        Xn = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_arr = Xn.values
    else:
        X_arr = np.asarray(X, dtype=float)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Directional-only noise estimate
    dir_mask = y_arr != 0
    if int(dir_mask.sum()) < 80:
        return {
            "enabled": False,
            "reason": "insufficient_directional",
            "noise_rate": None,
            "n_directional": int(dir_mask.sum()),
        }

    Xd = X_arr[dir_mask]
    yd = (y_arr[dir_mask] > 0).astype(int)
    n = len(yd)
    if n > max_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_rows, replace=False))
        Xd, yd = Xd[idx], yd[idx]
        n = len(yd)

    splits = max(2, min(int(n_splits), max(2, n // 200)))
    tscv = TimeSeriesSplit(n_splits=splits)
    disagree = 0
    total = 0
    fold_aucs: list[float] = []
    for tr, te in tscv.split(Xd):
        if len(tr) < 40 or len(te) < 20:
            continue
        try:
            clf = LogisticRegression(
                max_iter=200,
                C=0.5,
                class_weight="balanced",
                random_state=seed,
                solver="lbfgs",
            )
            clf.fit(Xd[tr], yd[tr])
            pred = clf.predict(Xd[te])
            proba = clf.predict_proba(Xd[te])[:, 1]
            disagree += int(np.sum(pred != yd[te]))
            total += len(te)
            fold_aucs.append(_safe_auc(yd[te], proba))
        except Exception:
            continue

    noise_rate = float(disagree / max(total, 1)) if total else None
    # Chance-level disagreement ≈ 0.5; map excess toward "noise score"
    noise_score = None
    if noise_rate is not None:
        noise_score = float(np.clip((noise_rate - 0.35) / 0.30, 0.0, 1.0))

    return {
        "enabled": True,
        "noise_rate": round(noise_rate, 4) if noise_rate is not None else None,
        "noise_score": round(noise_score, 4) if noise_score is not None else None,
        "weak_model_auc_mean": round(float(np.mean(fold_aucs)), 4) if fold_aucs else None,
        "n_directional": int(dir_mask.sum()),
        "n_evaluated": int(total),
        "n_cv_folds": len(fold_aucs),
    }


def barrier_hit_profile(y: pd.Series | np.ndarray, label_weights: np.ndarray | None = None) -> dict[str, Any]:
    y_arr = np.asarray(y).astype(float)
    n = max(len(y_arr), 1)
    pos = int((y_arr == 1).sum())
    neg = int((y_arr == -1).sum())
    flat = int((y_arr == 0).sum())
    directional = pos + neg
    w = np.asarray(label_weights, dtype=float) if label_weights is not None else None
    mean_w = float(np.mean(w[y_arr != 0])) if w is not None and directional else None
    low_clarity = None
    if w is not None and directional:
        low_clarity = float(np.mean(w[y_arr != 0] < 0.55))
    return {
        "n": int(len(y_arr)),
        "up": pos,
        "down": neg,
        "flat": flat,
        "directional_share": round(directional / n, 4),
        "flat_share": round(flat / n, 4),
        "imbalance_ratio": round(float(max(pos, neg) / max(min(pos, neg), 1)), 4) if directional else None,
        "mean_directional_weight": round(mean_w, 4) if mean_w is not None else None,
        "low_clarity_frac": round(low_clarity, 4) if low_clarity is not None else None,
    }


def analyze_label_quality(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    label_weights: np.ndarray | None = None,
    timeframe: str = "H1",
    cfg: dict[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Full label-quality dossier used by DQ gate + enterprise report."""
    cfg = cfg or {}
    profile = barrier_hit_profile(y, label_weights)
    noise = estimate_label_noise(
        X,
        y,
        seed=seed,
        n_splits=int(cfg.get("label_noise_cv_splits", 3)),
        max_rows=int(cfg.get("label_noise_max_rows", 8000)),
    )

    flags: dict[str, bool] = {
        "sparse_directional": float(profile.get("directional_share") or 0) < float(
            cfg.get("label_min_directional_share", 0.12)
        ),
        "high_flat": float(profile.get("flat_share") or 0) > float(cfg.get("label_max_flat_share", 0.88)),
        "high_imbalance": float(profile.get("imbalance_ratio") or 1) >= float(
            cfg.get("label_max_imbalance", 2.2)
        ),
        "low_clarity": float(profile.get("low_clarity_frac") or 0) >= float(
            cfg.get("label_max_low_clarity", 0.45)
        ),
        "high_noise": bool(
            noise.get("enabled")
            and noise.get("noise_score") is not None
            and float(noise["noise_score"]) >= float(cfg.get("label_max_noise_score", 0.65))
        ),
        "weak_separability": bool(
            noise.get("enabled")
            and noise.get("weak_model_auc_mean") is not None
            and float(noise["weak_model_auc_mean"]) < float(cfg.get("label_min_weak_auc", 0.52))
        ),
    }

    # Score 0–100
    score = 100.0
    if flags["sparse_directional"]:
        score -= 25.0
    if flags["high_flat"]:
        score -= 15.0
    if flags["high_imbalance"]:
        score -= 12.0
    if flags["low_clarity"]:
        score -= 10.0
    if flags["high_noise"]:
        score -= 18.0
    if flags["weak_separability"]:
        score -= 12.0
    score = float(max(0.0, min(100.0, round(score, 2))))

    recommendations: list[str] = []
    if flags["sparse_directional"] or flags["high_flat"]:
        recommendations.append("Widen barriers slightly or reduce horizon_bars to increase directional hits.")
    if flags["high_imbalance"]:
        recommendations.append("Enable class_weight / meta-labeling; review barrier ATR multiplier asymmetry.")
    if flags["low_clarity"]:
        recommendations.append("Increase barrier_atr_multiplier so vertical barrier hits are clearer.")
    if flags["high_noise"] or flags["weak_separability"]:
        recommendations.append(
            "Labels appear noisy vs features — tighten purge/embargo, prefer meta-labeling, or retune barriers."
        )
    if not recommendations:
        recommendations.append("Label profile looks healthy; keep current barrier settings.")

    hard_min = float(cfg.get("label_quality_min_score", 45.0))
    gate_pass = score >= hard_min and not (flags["sparse_directional"] and flags["high_noise"])
    if not bool(cfg.get("label_quality_hard", False)):
        # Advisory unless explicitly hard
        fail_hard = False
    else:
        fail_hard = not gate_pass

    return {
        "enabled": True,
        "timeframe": str(timeframe).upper(),
        "score": score,
        "gate_pass": not fail_hard,
        "fail_hard": fail_hard,
        "profile": profile,
        "noise": noise,
        "flags": flags,
        "recommendations": recommendations,
        "summary_ar": _summary_ar(timeframe, score, flags, noise),
    }


def _summary_ar(timeframe: str, score: float, flags: dict[str, bool], noise: dict[str, Any]) -> str:
    active = [k for k, v in flags.items() if v]
    noise_txt = ""
    if noise.get("noise_rate") is not None:
        noise_txt = f" · ضوضاء≈{float(noise['noise_rate']):.0%}"
    if active:
        return f"{timeframe}: جودة Labels {score:.0f}/100 — تحذيرات: {', '.join(active)}{noise_txt}"
    return f"{timeframe}: جودة Labels {score:.0f}/100 — سليمة{noise_txt}"


def merge_label_quality_into_dq(dq: dict[str, Any], label_q: dict[str, Any]) -> dict[str, Any]:
    """Attach label quality into the data-quality report and optionally harden the gate."""
    out = dict(dq)
    out["label_quality"] = label_q
    flags = dict(out.get("quality_flags") or {})
    for k, v in (label_q.get("flags") or {}).items():
        flags[f"label_{k}"] = bool(v)
    out["quality_flags"] = flags
    # Soft score blend
    if label_q.get("score") is not None and out.get("score") is not None:
        blended = 0.7 * float(out["score"]) + 0.3 * float(label_q["score"])
        out["score_with_labels"] = round(blended, 2)
    if label_q.get("fail_hard"):
        out["gate_pass"] = False
        out["skip_reason"] = out.get("skip_reason") or "label_quality_gate"
        reasons = list(out.get("reasons") or [])
        reasons.append("label_quality_hard_fail")
        out["reasons"] = reasons
    return out
