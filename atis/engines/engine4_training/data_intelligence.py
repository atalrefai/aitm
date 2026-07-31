"""Pre-train data intelligence report (enterprise readiness).

Produces a structured readiness dossier before model fitting:
completeness, outliers, class balance, regime diversity, temporal drift.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def analyze_training_frame(
    df: pd.DataFrame,
    y: pd.Series | np.ndarray,
    feature_cols: list[str],
    *,
    timeframe: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = cfg or {}
    n = len(df)
    feats = [c for c in feature_cols if c in df.columns]
    X = df[feats] if feats else pd.DataFrame(index=df.index)
    y_arr = np.asarray(y)

    missing_frac = float(X.isna().mean().mean()) if len(feats) else 0.0
    const_cols = [c for c in feats if X[c].nunique(dropna=True) <= 1]
    outlier_frac = 0.0
    if "is_outlier" in df.columns:
        outlier_frac = float(pd.Series(df["is_outlier"]).fillna(False).mean())
    elif "close" in df.columns and n >= 50:
        ret = pd.to_numeric(df["close"], errors="coerce").pct_change()
        med = float(ret.median(skipna=True) or 0.0)
        mad = float((ret - med).abs().median(skipna=True) or 0.0) or 1e-8
        z = (ret - med).abs() / (mad * 1.4826)
        outlier_frac = float((z > 6.0).fillna(False).mean())

    # Class balance
    vals, counts = np.unique(y_arr[np.isfinite(y_arr.astype(float))], return_counts=True)
    share = {str(int(v) if float(v).is_integer() else v): float(c / max(n, 1)) for v, c in zip(vals, counts)}
    dir_counts = [counts[i] for i, v in enumerate(vals) if float(v) != 0]
    imbalance = float(max(dir_counts) / max(min(dir_counts), 1)) if len(dir_counts) >= 2 else 1.0

    # Temporal split diversity (early vs late return vol)
    diversity = {"early_vol": 0.0, "late_vol": 0.0, "vol_shift_ratio": 1.0}
    if "close" in df.columns and n >= 100:
        r = pd.to_numeric(df["close"], errors="coerce").pct_change().fillna(0.0).values
        mid = n // 2
        e_vol = float(np.std(r[:mid]))
        l_vol = float(np.std(r[mid:]))
        diversity = {
            "early_vol": e_vol,
            "late_vol": l_vol,
            "vol_shift_ratio": float(l_vol / max(e_vol, 1e-12)),
        }

    # Session coverage if available
    session_share: dict[str, float] = {}
    if "session" in df.columns:
        vc = df["session"].astype(str).value_counts(normalize=True)
        session_share = {str(k): float(v) for k, v in vc.items()}

    score = 100.0
    flags: list[str] = []
    if n < int((cfg.get("dq_min_rows_by_tf") or {}).get(str(timeframe).upper(), 400)):
        score -= 25
        flags.append("thin_sample")
    if missing_frac > 0.05:
        score -= 15
        flags.append("missing_values")
    if outlier_frac > 0.05:
        score -= 10
        flags.append("high_outliers")
    if imbalance > 2.5:
        score -= 12
        flags.append("class_imbalance")
    if len(const_cols) > max(3, len(feats) // 10):
        score -= 8
        flags.append("many_constant_features")
    if abs(diversity["vol_shift_ratio"] - 1.0) > 1.8:
        score -= 6
        flags.append("regime_vol_shift")
    score = float(np.clip(score, 0, 100))

    ready = score >= float(cfg.get("data_intel_min_score", 55)) and "thin_sample" not in flags
    return {
        "timeframe": timeframe,
        "n_rows": int(n),
        "n_features": len(feats),
        "missing_frac": round(missing_frac, 5),
        "outlier_frac": round(outlier_frac, 5),
        "constant_features": const_cols[:20],
        "label_share": share,
        "imbalance_ratio": round(imbalance, 4),
        "market_diversity": diversity,
        "session_share": session_share,
        "score": round(score, 1),
        "ready": bool(ready),
        "flags": flags,
        "summary_ar": (
            f"جاهزية البيانات {score:.0f}/100 · صفوف={n} · ميزات={len(feats)} · "
            f"اختلال={imbalance:.2f} · أعلام={flags or ['نظيف']}"
        ),
    }
