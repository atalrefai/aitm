"""Hard Data-Quality Gate with automatic remediation for Engine 4."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# Absolute floors: slow TFs need fewer bars but still a usable WF sample.
_MIN_ROWS_BY_TF: dict[str, int] = {
    "M1": 8000,
    "M5": 4000,
    "M15": 2500,
    "M30": 1800,
    "H1": 1200,
    "H4": 600,
    "D1": 300,
}


def min_rows_for_timeframe(timeframe: str, cfg: dict[str, Any] | None = None) -> int:
    cfg = cfg or {}
    by_tf = cfg.get("dq_min_rows_by_tf") or {}
    tf = str(timeframe).upper()
    if tf in by_tf:
        return int(by_tf[tf])
    return int(by_tf.get(tf) or _MIN_ROWS_BY_TF.get(tf, int(cfg.get("min_rows", 80))))


def min_val_trades_for_timeframe(timeframe: str, cfg: dict[str, Any] | None = None) -> int:
    """Minimum Val trades per fold for best-fold / policy freeze eligibility."""
    cfg = cfg or {}
    by_tf = cfg.get("min_val_trades_by_tf") or {}
    tf = str(timeframe).upper()
    if tf in by_tf:
        return int(by_tf[tf])
    defaults = {"M1": 20, "M5": 20, "M15": 18, "M30": 15, "H1": 12, "H4": 8, "D1": 6}
    return int(defaults.get(tf, int(cfg.get("min_val_trades_per_fold", 12))))


def _timestamp_series(df: pd.DataFrame) -> pd.Series | None:
    for col in ("timestamp", "time", "datetime"):
        if col in df.columns:
            ts = pd.to_datetime(df[col], utc=True, errors="coerce")
            if ts.notna().any():
                return ts
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(pd.to_datetime(df.index, utc=True), index=df.index)
    return None


def _expected_bar_delta(timeframe: str) -> pd.Timedelta | None:
    mapping = {
        "M1": "1min",
        "M5": "5min",
        "M15": "15min",
        "M30": "30min",
        "H1": "1h",
        "H4": "4h",
        "D1": "1D",
        "W1": "7D",
    }
    key = str(timeframe).upper()
    if key not in mapping:
        return None
    return pd.Timedelta(mapping[key])


def remediate_frame(
    df: pd.DataFrame,
    *,
    timeframe: str,
    feature_cols: list[str] | None = None,
    max_gap_bars: int = 3,
    jump_z: float = 8.0,
    drop_duplicate_rows: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Flag gaps/jumps, drop dead columns, optionally drop duplicate timestamps.

    Default keeps row count stable so X/y/label_weights stay aligned in train.
    """
    actions: list[str] = []
    out = df.copy()
    n0 = len(out)

    ts = _timestamp_series(out)
    n_dup = 0
    if ts is not None:
        dup_mask = ts.duplicated(keep="last")
        n_dup = int(dup_mask.sum())
        if n_dup and drop_duplicate_rows:
            out = out.loc[~dup_mask].copy()
            ts = _timestamp_series(out)
            actions.append(f"drop_duplicate_timestamps:{n_dup}")
        elif n_dup:
            actions.append(f"detect_duplicate_timestamps:{n_dup}")

    gap_frac = 0.0
    n_gaps = 0
    if ts is not None and len(out) >= 3:
        delta = _expected_bar_delta(timeframe)
        if delta is not None:
            diffs = ts.diff()
            # Gaps larger than max_gap_bars * expected spacing
            gap_mask = diffs > (delta * max(1, int(max_gap_bars)))
            n_gaps = int(gap_mask.fillna(False).sum())
            gap_frac = float(n_gaps / max(len(out) - 1, 1))
            if n_gaps:
                actions.append(f"detect_candle_gaps:{n_gaps}")

    jump_frac = 0.0
    if "close" in out.columns and len(out) >= 30:
        close = pd.to_numeric(out["close"], errors="coerce")
        ret = close.pct_change()
        med = float(ret.median(skipna=True) or 0.0)
        mad = float((ret - med).abs().median(skipna=True) or 0.0)
        scale = max(mad * 1.4826, 1e-8)
        z = (ret - med).abs() / scale
        jump_mask = z > float(jump_z)
        jump_frac = float(jump_mask.fillna(False).mean())
        if jump_mask.fillna(False).any():
            # Soft clip returns via close reconstruction is risky; mark outliers only.
            if "is_outlier" not in out.columns:
                out["is_outlier"] = False
            out.loc[jump_mask.fillna(False), "is_outlier"] = True
            actions.append(f"flag_price_jumps:{int(jump_mask.fillna(False).sum())}")

    dropped_const: list[str] = []
    cols = list(feature_cols or [])
    if cols:
        keep: list[str] = []
        for c in cols:
            if c not in out.columns:
                continue
            s = out[c]
            if s.nunique(dropna=True) <= 1:
                dropped_const.append(c)
                continue
            keep.append(c)
        if dropped_const:
            actions.append(f"drop_constant_features:{len(dropped_const)}")
            feature_cols = keep

    # Fill sparse NaNs in features with causal ffill then 0 (no bfill).
    filled = 0
    for c in feature_cols or []:
        if c not in out.columns:
            continue
        if out[c].isna().any():
            before = int(out[c].isna().sum())
            out[c] = out[c].ffill()
            out[c] = out[c].fillna(0.0)
            filled += before
    if filled:
        actions.append(f"ffill_missing_feature_cells:{filled}")

    return out, {
        "n_rows_before": n0,
        "n_rows_after": int(len(out)),
        "n_duplicates_dropped": n_dup,
        "n_candle_gaps": n_gaps,
        "gap_frac": round(gap_frac, 4),
        "jump_frac": round(jump_frac, 4),
        "dropped_constant_features": dropped_const,
        "feature_cols": list(feature_cols or []),
        "actions": actions,
    }


def compute_data_quality_score(
    df: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    *,
    timeframe: str,
    remediation: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score 0–100 plus hard-gate decision and Awareness payload."""
    cfg = cfg or {}
    rem = remediation or {}
    n = int(len(y))
    min_rows = min_rows_for_timeframe(timeframe, cfg)
    score = 100.0
    flags: dict[str, bool] = {}
    reasons: list[str] = []

    # Sample size
    if n < min_rows:
        deficit = (min_rows - n) / max(min_rows, 1)
        score -= min(40.0, 25.0 + 40.0 * deficit)
        flags["insufficient_rows"] = True
        reasons.append(f"rows={n} < min_rows={min_rows} for {timeframe}")
    else:
        flags["insufficient_rows"] = False

    # Missing features
    missing = {}
    for c in feature_cols[:100]:
        if c not in df.columns:
            continue
        frac = float(pd.Series(df[c]).isna().mean()) if len(df) else 0.0
        if frac > 0:
            missing[c] = round(frac, 4)
    miss_pen = min(20.0, 2.0 * len(missing) + 40.0 * max(missing.values() or [0.0]))
    score -= miss_pen
    flags["many_missing_cols"] = len(missing) >= 10

    # Label balance / directional mass
    pos = int((y == 1).sum())
    neg = int((y == -1).sum())
    flat = int((y == 0).sum())
    total = max(n, 1)
    directional = pos + neg
    imbalance = float(max(pos, neg) / max(min(pos, neg), 1)) if directional > 0 else 99.0
    dir_share = directional / total
    if imbalance >= 2.2:
        score -= 12.0
        flags["high_imbalance"] = True
        reasons.append(f"directional_imbalance={imbalance:.2f}")
    else:
        flags["high_imbalance"] = False
    if dir_share < 0.12:
        score -= 15.0
        flags["sparse_directional_labels"] = True
        reasons.append(f"directional_share={dir_share:.3f}")
    else:
        flags["sparse_directional_labels"] = False

    outlier_frac = float(df["is_outlier"].mean()) if "is_outlier" in df.columns else float(
        rem.get("jump_frac", 0.0) or 0.0
    )
    if outlier_frac >= 0.05:
        score -= min(15.0, 5.0 + 100.0 * outlier_frac)
        flags["high_outliers"] = True
    else:
        flags["high_outliers"] = False

    gap_frac = float(rem.get("gap_frac", 0.0) or 0.0)
    if gap_frac >= 0.02:
        score -= min(15.0, 50.0 * gap_frac)
        flags["excessive_gaps"] = True
        reasons.append(f"gap_frac={gap_frac:.3f}")
    else:
        flags["excessive_gaps"] = False

    # Fold liquidity foresight: estimate Val bars after WF split
    n_splits = max(1, int(cfg.get("walk_forward_splits", 5)))
    fold_val_ratio = float(cfg.get("fold_validation_ratio", 0.25))
    approx_fold = max(1, (n - n // 3) // n_splits)
    approx_val = max(0, int(approx_fold * fold_val_ratio))
    min_val = min_val_trades_for_timeframe(timeframe, cfg)
    # Need enough Val bars to possibly realize min_val trades (≈2 bars/trade heuristic)
    if approx_val < max(6, int(min_val * 1.5)):
        score -= 20.0
        flags["insufficient_fold_liquidity"] = True
        reasons.append(
            f"approx_val_bars={approx_val} too small for min_val_trades={min_val} ({timeframe})"
        )
    else:
        flags["insufficient_fold_liquidity"] = False

    score = float(max(0.0, min(100.0, round(score, 2))))
    hard_min = float(cfg.get("dq_gate_min_score", 55.0))
    # Early reject: tiny sample / empty folds / dead labels. Score-only fails when severe.
    fail_hard = bool(cfg.get("dq_gate_enabled", True)) and (
        flags.get("insufficient_rows")
        or flags.get("insufficient_fold_liquidity")
        or flags.get("sparse_directional_labels")
        or score < hard_min
    )
    # Allow override: advisory-only mode
    if not bool(cfg.get("dq_gate_hard", True)):
        fail_hard = False

    skip_reason = None
    if fail_hard:
        if flags.get("insufficient_rows"):
            skip_reason = "dq_insufficient_rows"
        elif flags.get("insufficient_fold_liquidity"):
            skip_reason = "dq_insufficient_fold_liquidity"
        elif flags.get("sparse_directional_labels"):
            skip_reason = "dq_sparse_labels"
        else:
            skip_reason = "dq_score_below_threshold"

    return {
        "score": score,
        "n_rows": n,
        "n_features": len(feature_cols),
        "min_rows_required": min_rows,
        "approx_val_bars_per_fold": approx_val,
        "min_val_trades_required": min_val,
        "label_share": {
            "up": round(pos / total, 4),
            "down": round(neg / total, 4),
            "flat": round(flat / total, 4),
        },
        "directional_imbalance_ratio": round(imbalance, 4),
        "missing_frac_top": dict(sorted(missing.items(), key=lambda kv: -kv[1])[:15]),
        "outlier_frac": round(outlier_frac, 4),
        "gap_frac": round(gap_frac, 4),
        "quality_flags": flags,
        "gate_pass": not fail_hard,
        "skip_reason": skip_reason,
        "reasons": reasons,
        "remediation": rem,
        "awareness": {
            "data_quality_score": score,
            "risk_flags": [k for k, v in flags.items() if v],
            "explanation_ar": _explain_ar(timeframe, score, flags, reasons, skip_reason),
        },
    }


def _explain_ar(
    timeframe: str,
    score: float,
    flags: dict[str, bool],
    reasons: list[str],
    skip_reason: str | None,
) -> str:
    if skip_reason:
        detail = "؛ ".join(reasons[:3]) if reasons else skip_reason
        return f"رفض مبكر لـ {timeframe}: درجة الجودة {score:.0f}/100 — {detail}"
    if any(flags.values()):
        active = [k for k, v in flags.items() if v]
        return f"{timeframe}: درجة {score:.0f}/100 مع تحذيرات: {', '.join(active)}"
    return f"{timeframe}: درجة جودة البيانات {score:.0f}/100 — جاهز للتدريب"


def run_data_quality_gate(
    df: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    *,
    timeframe: str,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, Any]]:
    """Remediate then score. Caller must abort training when gate_pass is False."""
    cfg = cfg or {}
    remediated, rem = remediate_frame(
        df,
        timeframe=timeframe,
        feature_cols=list(feature_cols),
        max_gap_bars=int(cfg.get("dq_max_gap_bars", 3)),
        jump_z=float(cfg.get("dq_jump_z", 8.0)),
        drop_duplicate_rows=bool(cfg.get("dq_drop_duplicate_rows", False)),
    )
    cols = list(rem.get("feature_cols") or feature_cols)
    if len(remediated) == len(y):
        y_aligned = y
    elif hasattr(y, "loc") and remediated.index.equals(getattr(y, "index", pd.Index([]))):
        y_aligned = y.loc[remediated.index]
    else:
        y_aligned = pd.Series(np.asarray(y)[: len(remediated)], index=remediated.index)

    report = compute_data_quality_score(
        remediated,
        y_aligned,
        cols,
        timeframe=timeframe,
        remediation=rem,
        cfg=cfg,
    )
    report["n_rows"] = int(len(y_aligned))
    return remediated, y_aligned, cols, report


def fold_has_min_val_liquidity(
    n_val_trades: float,
    *,
    timeframe: str,
    cfg: dict[str, Any] | None = None,
    n_val_bars: int = 0,
) -> bool:
    """True when a fold may participate in best-fold / deploy-policy selection."""
    need = min_val_trades_for_timeframe(timeframe, cfg)
    if float(n_val_trades) < float(need):
        return False
    # Also reject near-zero rate on long Val windows
    if n_val_bars >= 40 and float(n_val_trades) / float(n_val_bars) < 0.01:
        return False
    return True
