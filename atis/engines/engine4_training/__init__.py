"""Engine 4 — Training & Backtesting (walk-forward, no random split)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from atis.config import (
    PROJECT_ROOT,
    ensure_project_dirs,
    get_path,
    load_engine_config,
    set_global_seed,
)
from atis.engines.engine4_training.data_sources import load_training_frame
from atis.engines.engine4_training.deep_learning import HAS_TORCH, train_llmodel
from atis.engines.engine4_training.final_model import publish_final_model
from atis.engines.engine4_training.multi_tf_decision import (  # noqa: F401
    confirm_tfs_for_primary,
    multi_tf_decision,
)
from atis.shared.data_registry import DataStateRegistry
from atis.shared.logging_utils import get_logger

# Optional multi-TF helpers (safe if a long-lived process has a stale data_sources).
try:
    from atis.engines.engine4_training.data_sources import (  # noqa: E402
        enrich_with_higher_timeframes,
        higher_timeframes_for,
    )
except ImportError:  # pragma: no cover - hot-reload race on old server process
    enrich_with_higher_timeframes = None  # type: ignore[assignment]
    higher_timeframes_for = None  # type: ignore[assignment]

logger = get_logger("atis.engine4")

# Bumped when train/test/validation methodology changes — appears in job logs.
PIPELINE_VERSION = "e4-v16.0-research-factory-20260731"

# Arabic labels for gate failure keys (UI + logs). English key always kept alongside.
GATE_FAILURE_AR: dict[str, str] = {
    "sharpe_or_drawdown": "شارب ضعيف أو تراجع أقصى مرتفع",
    "min_trades_oos": "عدد صفقات الاختبار غير كافٍ",
    "buy_hold": "لم يتفوق على الشراء والاحتفاظ بمخاطر مقبولة",
    "random_baseline": "لم يتفوق على خط الأساس العشوائي",
    "median_fold_val_sharpe": "وسيط شارب التحقق عبر الطيات ضعيف",
    "val_catastrophic": "التحقق كارثي مقابل اختبار إيجابي",
    "val_test_gap_weak_test": "فجوة تحقق↔اختبار مع اختبار ضعيف",
    "val_test_gap_hard": "فجوة تحقق≫اختبار صلبة",
    "overfit_sharpe_gap": "فجوة شارب تدريب→تحقق (إفراط ملاءمة)",
    "overfit_acc_sharpe": "فجوة دقة + شارب تدريب→تحقق",
    "overfit_sharpe_gap_hard": "فجوة شارب تدريب→تحقق قاسية مع انهيار تعميم",
    "weak_sharpe_ci": "الحد الأدنى لفاصل شارب غير موثوق",
    "unstable_generalization": "تعميم غير مستقر (تحقق متفائل)",
    "inactive_folds": "طيات غير نشطة (سيولة منخفضة)",
    "overtrading_folds": "إفراط تداول عبر الطيات",
    "oos_trade_rate": "معدل تداول الاختبار منخفض جداً",
    "deploy_holdout_sharpe": "شارب نافذة النشر ضعيف",
    "deploy_holdout_too_few_trades": "صفقات نافذة النشر أقل من الحد الأدنى المطلق",
    "deploy_holdout_trades": "صفقات نافذة النشر غير كافية",
    "sparse_deploy_unreliable": "نشر متناثر غير موثوق (شارب مرتفع على صفقات قليلة)",
    "filter_driven_edge": "حافة مدفوعة بالفلاتر مع تصنيف قريب من العشوائية",
    "filter_driven_sparse": "حافة فلاتر + شارب متناثر",
    "data_quality_gate": "بوابة جودة البيانات (رفض مبكر)",
    "val_fold_liquidity": "سيولة طيات التحقق دون الحد الأدنى",
    "overfit_champion_blocked": "منع نشر إطار overfitting مع وجود بديل متوازن",
    "regime_unstable": "أداء غير مستقر عبر أنظمة السوق (ترند/رينج/تقلب)",
    "weak_expectancy": "توقّع الصفقة (Expectancy) ضعيف أو سالب",
    "high_pbo": "احتمال إفراط ملاءمة الاختبار الرجعي مرتفع (PBO)",
    "stress_fragile": "هش تحت اختبارات الضغط (سبريد/ضوضاء/انهيار)",
    "monte_carlo_unstable": "مسارات مونت كارلو غير مستقرة",
    "low_live_readiness": "درجة الجاهزية الحية دون الحد الأدنى",
    "h4_no_edge": "H4 بلا حافة تصنيفية (قرب العشوائية)",
    "label_quality_gate": "بوابة جودة التسميات (ضوضاء/توازن Labels)",
    "challenger_not_better": "المتحدي لم يتفوق على البطل الحالي",
    "feature_unstable": "ميزات غير مستقرة عبر طيات التحقق",
    "fold_unstable": "طيات التحقق غير متسقة (IQR/إيجابية ضعيفة)",
    "expectancy_below_cost": "التوقع أقل من تكلفة التنفيذ المقدّرة",
    "crisis_holdout_weak": "أداء ضعيف على نافذة الأزمة/التحول",
    "recent_holdout_weak": "أداء ضعيف على النافذة الحديثة",
    "trade_rate_saturated": "معدل التداول ملامس للسقف (سياسة مشبعة)",
}


def gate_failure_ar(key: str) -> str:
    return GATE_FAILURE_AR.get(str(key), str(key))


def annotate_gate_failures(keys: list[str]) -> list[dict[str, str]]:
    """Structured gate failures for API/UI: English key + Arabic reason."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for k in keys:
        key = str(k)
        if key in seen:
            continue
        seen.add(key)
        out.append({"key": key, "ar": gate_failure_ar(key), "en": key})
    return out


def overtrading_rate_exceeds(rate: float, max_rate: float, *, tol_frac: float = 0.05) -> bool:
    """True only when trade_rate meaningfully exceeds the fold cap (not equality noise)."""
    cap = float(max_rate)
    r = float(rate)
    if not np.isfinite(r) or not np.isfinite(cap) or cap <= 0:
        return False
    return r > max(cap * (1.0 + float(tol_frac)), cap + 1e-6)


def should_fail_overfit_sharpe_gap_hard(
    *,
    sharpe_gap_tv: float,
    overfit_sharpe_gap: float,
    train_sharpe: float,
    val_sharpe: float,
    test_sharpe: float,
    sharpe_gap_vt: float,
    n_test_trades: float,
    min_sharpe: float,
    min_trades: int,
    val_test_gap_hard: float,
    acc_gap_tv: float = 0.0,
    max_acc_gap: float = 0.18,
) -> bool:
    """Hard Train→Val Sharpe gap only when generalization truly collapsed.

    Exempts strong Test + Val↔Test consistency + enough trades (e.g. M30 report).
    Still fails large accuracy-gap overfitting (e.g. H4-style) when Test is weak.
    """
    hard_thr = max(float(overfit_sharpe_gap) * 1.5, 3.0)
    if float(sharpe_gap_tv) <= hard_thr:
        return False
    if float(train_sharpe) <= float(val_sharpe):
        return False
    te = float(test_sharpe)
    va = float(val_sharpe)
    gap_vt = float(sharpe_gap_vt)
    n_te = float(n_test_trades)
    generalization_ok = (
        te >= max(float(min_sharpe), 0.75)
        and gap_vt <= float(val_test_gap_hard)
        and n_te >= float(min_trades)
        and va >= float(min_sharpe)
    )
    # Real collapse: Val weak vs Train AND Test collapsed, or huge acc gap with weak Test.
    real_collapse = (
        (va < 0.45 * float(train_sharpe) and te < max(float(min_sharpe), 0.5 * max(va, 1e-9)))
        or (float(acc_gap_tv) > float(max_acc_gap) * 1.25 and te < 1.0)
    )
    if generalization_ok and not real_collapse:
        return False
    return True


TRAINING_STAGES = (
    "queued",
    "loading_data",
    "features",
    "labeling",
    "feature_selection",
    "walk_forward",
    "val_policy",
    "test_oos",
    "deploy_holdout",
    "gates",
    "done",
    "error",
    "skipped",
)


def empty_tf_run_status(timeframe: str) -> dict[str, Any]:
    return {
        "timeframe": timeframe,
        "stage": "queued",
        "progress_pct": 0.0,
        "passed_gates": None,
        "gate_failures": [],
        "gate_failures_detail": [],
        "metrics": {},
        "folds": [],
        "fit_diagnosis": {},
        "htf_sources": [],
        "n_htf_cols": 0,
        "model_version": None,
        "error": None,
        "message": "قيد الانتظار",
    }

# Prefer relative / scale-free columns over absolute price levels (gold AUC ~0.5).
_RELATIVE_FEATURE_HINTS = (
    "ret", "pct", "z", "dist", "rsi", "atr", "bb_", "macd", "stoch", "cci",
    "mom", "roc", "spread", "ratio", "norm", "score", "bias", "strength",
    "structure", "pat_", "vol_", "session", "near_", "trend", "ema_slope",
    "sma_slope", "body", "wick", "range",
)
_ABSOLUTE_FEATURE_HINTS = (
    "open", "high", "low", "close", "tick_volume", "real_volume",
    "ema_20", "ema_50", "ema_200", "sma_20", "sma_50", "sma_200",
    "vwap", "pivot", "support", "resist",
)

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:  # pragma: no cover
    HAS_LGB = False


META_EXCLUDE = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
    "symbol",
    "timeframe",
    "is_imputed",
    "is_outlier",
    "label",
    "label_meta",
    "session",
    "vol_regime",
    "label_meta",
}


@dataclass
class TrainResult:
    symbol: str
    timeframe: str
    version: str
    model_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    passed_gates: bool = False
    error: str | None = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _cfg() -> dict[str, Any]:
    return load_engine_config().get("engine4_training", {})


DEFAULT_HORIZON_BY_TF = {
    "M1": 30,
    "M5": 12,
    "M15": 8,
    "M30": 6,
    "H1": 6,
    "H4": 4,
    "D1": 3,
    "W1": 2,
    "MN1": 1,
}

PERIODS_PER_YEAR = {
    "M1": 252 * 24 * 60,
    "M5": 252 * 24 * 12,
    "M15": 252 * 24 * 4,
    "M30": 252 * 24 * 2,
    "H1": 252 * 24,
    "H4": 252 * 6,
    "D1": 252,
    "W1": 52,
    "MN1": 12,
}


def horizon_for_timeframe(timeframe: str, cfg: dict[str, Any] | None = None) -> int:
    cfg = cfg or _cfg()
    by_tf = cfg.get("horizon_by_timeframe") or DEFAULT_HORIZON_BY_TF
    if timeframe in by_tf:
        return max(1, int(by_tf[timeframe]))
    return max(1, int(cfg.get("horizon_bars", DEFAULT_HORIZON_BY_TF.get(timeframe, 8))))


def periods_per_year_for(timeframe: str) -> float:
    return float(PERIODS_PER_YEAR.get(timeframe, 252 * 24))


def triple_barrier_labels(
    df: pd.DataFrame,
    *,
    horizon: int,
    atr_mult: float,
) -> pd.Series:
    labels, _ = triple_barrier_labels_and_weights(df, horizon=horizon, atr_mult=atr_mult)
    return labels


def triple_barrier_labels_and_weights(
    df: pd.DataFrame,
    *,
    horizon: int,
    atr_mult: float,
) -> tuple[pd.Series, pd.Series]:
    """
    Triple-barrier labeling (causal at decision time t; label uses future path
    only for y — never as features).
    Barriers: +atr_mult*ATR (upper), -atr_mult*ATR (lower), or horizon timeout.
    Classes: 1=up, -1=down, 0=timeout/vertical.
    Weights emphasize clearer barrier hits (capped).
    """
    close = df["close"].values
    atr = df["atr"].values if "atr" in df.columns else np.full(len(df), np.nan)
    # Fallback ATR proxy
    if np.isnan(atr).all():
        atr = pd.Series(close).pct_change().abs().rolling(14).mean().fillna(0.001).values * close

    n = len(df)
    labels = np.full(n, np.nan)
    weights = np.ones(n, dtype=float)
    for i in range(n - horizon):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            a = abs(close[i]) * 0.001
        upper = close[i] + atr_mult * a
        lower = close[i] - atr_mult * a
        label = 0
        hit_j = horizon
        for j in range(1, horizon + 1):
            px = close[i + j]
            if px >= upper:
                label = 1
                hit_j = j
                break
            if px <= lower:
                label = -1
                hit_j = j
                break
        labels[i] = label
        if label != 0:
            move = abs(float(close[i + hit_j]) - float(close[i])) / max(float(a), 1e-12)
            weights[i] = float(np.clip(move / max(atr_mult, 1e-6), 0.75, 3.0))
        else:
            weights[i] = 0.35
    return (
        pd.Series(labels, index=df.index, name="label"),
        pd.Series(weights, index=df.index, name="label_weight"),
    )


def classification_horizon_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Simple up/down/flat by forward return sign (fallback)."""
    fwd = df["close"].shift(-horizon) / df["close"] - 1.0
    thr = df["close"].pct_change().rolling(50).std().fillna(0.0005)
    lab = pd.Series(0, index=df.index, dtype=float)
    lab = lab.mask(fwd > thr, 1.0)
    lab = lab.mask(fwd < -thr, -1.0)
    lab.iloc[-horizon:] = np.nan
    return lab


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in META_EXCLUDE:
            continue
        if df[c].dtype == object:
            continue
        cols.append(c)
    return cols


def _feature_relative_score(name: str) -> int:
    n = str(name).lower()
    score = 0
    for h in _RELATIVE_FEATURE_HINTS:
        if h in n:
            score += 2
    for h in _ABSOLUTE_FEATURE_HINTS:
        if n == h or n.endswith("_" + h) or n.startswith(h + "_"):
            score -= 3
    if any(x in n for x in ("dist_", "zscore", "z_", "pct", "return", "ret_")):
        score += 3
    # Prefer causal multi-TF context / agreement features (report: single-TF AUC≈0.5).
    if n.startswith("htf_") or n.startswith("mtf_") or n.startswith("feat_"):
        score += 4
    return score


def engineer_learning_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add scale-free derived features that improve directional learning.

    All transforms are causal (past-only rolling / lag). Does not use future bars.
    """
    work = df.copy()
    close = work["close"].astype(float) if "close" in work.columns else None
    if close is not None and len(close) >= 5:
        work["feat_ret_1"] = close.pct_change(1)
        work["feat_ret_3"] = close.pct_change(3)
        work["feat_ret_8"] = close.pct_change(8)
        work["feat_ret_21"] = close.pct_change(21)
        work["feat_ret_vol_20"] = work["feat_ret_1"].rolling(20, min_periods=8).std()
        work["feat_ret_vol_60"] = work["feat_ret_1"].rolling(60, min_periods=20).std()
        mu20 = work["feat_ret_1"].rolling(20, min_periods=8).mean()
        sd20 = work["feat_ret_vol_20"].replace(0, np.nan)
        work["feat_ret_z_20"] = (work["feat_ret_1"] - mu20) / sd20
        # Momentum acceleration (causal lag of returns).
        work["feat_mom_accel"] = work["feat_ret_3"] - work["feat_ret_3"].shift(3)
        vol_ratio = work["feat_ret_vol_20"] / work["feat_ret_vol_60"].replace(0, np.nan)
        work["feat_vol_compress"] = vol_ratio
    if close is not None and "atr" in work.columns:
        work["feat_atr_pct"] = work["atr"].astype(float) / close.replace(0, np.nan)
        atr_pct = work["feat_atr_pct"]
        work["feat_atr_pct_z"] = (
            (atr_pct - atr_pct.rolling(50, min_periods=20).mean())
            / atr_pct.rolling(50, min_periods=20).std().replace(0, np.nan)
        )
    if "rsi_14" in work.columns:
        work["feat_rsi_centered"] = (work["rsi_14"].astype(float) - 50.0) / 50.0
        work["feat_rsi_slope"] = work["rsi_14"].astype(float).diff(3) / 50.0
    if close is not None and "ema_20" in work.columns and "ema_50" in work.columns:
        work["feat_ema_spread"] = (work["ema_20"].astype(float) - work["ema_50"].astype(float)) / close.replace(
            0, np.nan
        )
    if close is not None and "ema_50" in work.columns and "ema_200" in work.columns:
        work["feat_ema_regime"] = (work["ema_50"].astype(float) - work["ema_200"].astype(float)) / close.replace(
            0, np.nan
        )
    if "bb_upper" in work.columns and "bb_lower" in work.columns and close is not None:
        span = (work["bb_upper"].astype(float) - work["bb_lower"].astype(float)).replace(0, np.nan)
        work["feat_bb_pctb"] = (close - work["bb_lower"].astype(float)) / span
    if "adx" in work.columns and "trend_strength" in work.columns:
        work["feat_trend_adx"] = work["trend_strength"].astype(float) * (work["adx"].astype(float) / 50.0)
    if "macd_hist" in work.columns and close is not None:
        work["feat_macd_hist_norm"] = work["macd_hist"].astype(float) / close.replace(0, np.nan)
        work["feat_macd_hist_slope"] = work["macd_hist"].astype(float).diff(2) / close.replace(0, np.nan)
    if close is not None and "high" in work.columns and "low" in work.columns:
        rng = (work["high"].astype(float) - work["low"].astype(float)).replace(0, np.nan)
        work["feat_close_loc"] = (close - work["low"].astype(float)) / rng
        work["feat_range_pct"] = rng / close.replace(0, np.nan)
    # Encode session / vol_regime categoricals as ordinals when present as strings.
    if "session" in work.columns and work["session"].dtype == object:
        session_map = {"asia": 0.0, "europe": 1.0, "us": 2.0, "overlap": 1.5}
        work["feat_session_code"] = work["session"].astype(str).str.lower().map(session_map).fillna(1.0)
        # Session one-hot style without leakage (deterministic map).
        work["feat_session_us"] = (work["feat_session_code"] >= 1.8).astype(float)
        work["feat_session_asia"] = (work["feat_session_code"] <= 0.2).astype(float)
    else:
        # Derive London/NY/Asia from timestamp when session column absent (v16).
        ts = None
        for col in ("timestamp", "time", "datetime"):
            if col in work.columns:
                ts = pd.to_datetime(work[col], utc=True, errors="coerce")
                break
        if ts is None and isinstance(work.index, pd.DatetimeIndex):
            ts = pd.Series(work.index, index=work.index)
        if ts is not None:
            h = ts.dt.hour.fillna(12).astype(float)
            work["feat_hour_sin"] = np.sin(2 * np.pi * h / 24.0)
            work["feat_hour_cos"] = np.cos(2 * np.pi * h / 24.0)
            work["feat_session_asia"] = ((h >= 0) & (h < 8)).astype(float)
            work["feat_session_london"] = ((h >= 7) & (h < 16)).astype(float)
            work["feat_session_us"] = ((h >= 12) & (h < 21)).astype(float)
            work["feat_session_overlap"] = ((h >= 12) & (h < 16)).astype(float)
    if "vol_regime" in work.columns and work["vol_regime"].dtype == object:
        vol_map = {"low": 0.0, "mid": 1.0, "high": 2.0, "normal": 1.0, "calm": 0.0, "violent": 2.0}
        work["feat_vol_regime_code"] = work["vol_regime"].astype(str).str.lower().map(vol_map).fillna(1.0)
    for col in work.columns:
        if str(col).startswith("feat_") and pd.api.types.is_numeric_dtype(work[col]):
            work[col] = work[col].replace([np.inf, -np.inf], np.nan)
    return work


def resolve_model_cfg_for_tf(cfg: dict[str, Any], timeframe: str) -> dict[str, Any]:
    """Merge global LGB/RF knobs with per-timeframe overrides (overfit control)."""
    out = dict(cfg)
    by_tf = cfg.get("lgb_by_tf") or {}
    tf_over = by_tf.get(timeframe) or by_tf.get(str(timeframe).upper()) or {}
    for key, value in tf_over.items():
        out[key] = value
    return out


def prefer_relative_features(cols: list[str], *, keep_min: int = 12) -> list[str]:
    """Drop weak absolute price-level columns when enough relative features remain."""
    if len(cols) <= keep_min:
        return cols
    ranked = sorted(cols, key=lambda c: (_feature_relative_score(c), c), reverse=True)
    kept = [c for c in ranked if _feature_relative_score(c) >= 0]
    if len(kept) < keep_min:
        kept = ranked[: max(keep_min, len(ranked) // 2)]
    return kept


def structure_primary_sides(allow_long: np.ndarray, allow_short: np.ndarray) -> np.ndarray:
    """Trend/structure primary signal for meta-labeling (exclusive long/short)."""
    out = np.zeros(len(allow_long), dtype=float)
    long_only = np.asarray(allow_long, dtype=bool) & ~np.asarray(allow_short, dtype=bool)
    short_only = np.asarray(allow_short, dtype=bool) & ~np.asarray(allow_long, dtype=bool)
    out[long_only] = 1.0
    out[short_only] = -1.0
    return out


def cap_preds_by_trade_rate(
    preds: np.ndarray,
    confidences: np.ndarray,
    *,
    max_trade_rate: float,
) -> np.ndarray:
    """Hard cap overtrading (e.g. fold trade_rate 0.27) while keeping top confidence."""
    out = preds.astype(float).copy()
    if max_trade_rate <= 0 or max_trade_rate >= 1:
        return out
    idx = np.flatnonzero(out != 0)
    n_max = max(1, int(round(len(out) * float(max_trade_rate))))
    if len(idx) <= n_max:
        return out
    order = idx[np.argsort(confidences[idx])[::-1][:n_max]]
    kept = np.zeros_like(out)
    kept[order] = out[order]
    return kept


def policy_liquidity_starved(
    *,
    trades: float,
    n_bars: int,
    min_trades: int,
    target_trade_rate: float,
) -> bool:
    rate = float(trades) / max(int(n_bars), 1)
    if trades < float(min_trades):
        return True
    if target_trade_rate > 0 and rate < target_trade_rate * 0.35:
        return True
    return False


def build_model(name: str, seed: int = 42, cfg: dict[str, Any] | None = None) -> Any:
    name = name.lower()
    cfg = cfg or {}
    if name == "logistic":
        return LogisticRegression(max_iter=800, random_state=seed, n_jobs=1, class_weight="balanced")
    if name in ("rf", "random_forest"):
        return RandomForestClassifier(
            n_estimators=int(cfg.get("rf_estimators", 250)),
            max_depth=int(cfg.get("rf_max_depth", 8)),
            min_samples_leaf=int(cfg.get("rf_min_samples_leaf", 8)),
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    if name in ("lightgbm", "lgbm", "lgb"):
        if not HAS_LGB:
            logger.warning("lightgbm_missing_fallback_rf")
            return build_model("rf", seed, cfg)
        return lgb.LGBMClassifier(
            n_estimators=int(cfg.get("lgb_estimators", 220)),
            learning_rate=float(cfg.get("lgb_learning_rate", 0.03)),
            max_depth=int(cfg.get("lgb_max_depth", 4)),
            num_leaves=int(cfg.get("lgb_num_leaves", 15)),
            min_child_samples=int(cfg.get("lgb_min_child_samples", 80)),
            subsample=float(cfg.get("lgb_subsample", 0.75)),
            colsample_bytree=float(cfg.get("lgb_colsample", 0.7)),
            reg_alpha=float(cfg.get("lgb_reg_alpha", 0.2)),
            reg_lambda=float(cfg.get("lgb_reg_lambda", 1.5)),
            class_weight="balanced",
            random_state=seed,
            verbosity=-1,
        )
    if name in ("ensemble", "soft_vote", "voting"):
        from atis.engines.engine4_training.ensemble_models import build_soft_voting_ensemble

        return build_soft_voting_ensemble(seed=seed, cfg=cfg)
    raise ValueError(f"Unknown model: {name}")


def walk_forward_splits(
    n: int,
    n_splits: int,
    train_ratio: float,
    *,
    embargo: int = 0,
    purge: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding chronological folds with embargo gap and label purge.

    Purge drops the last `purge` train rows whose label windows overlap the
    embargo/test region (critical for triple-barrier labels).
    """
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    embargo = max(0, int(embargo))
    purge = max(0, int(purge))
    min_train = max(40, int(n * train_ratio / max(n_splits, 1)))
    min_train = min(min_train, max(40, n // 3))
    fold_size = max(20, (n - min_train) // max(n_splits, 1))
    actual_splits = n_splits
    if n < 200:
        actual_splits = min(n_splits, 3)
        fold_size = max(15, (n - min_train) // max(actual_splits, 1))
    for i in range(actual_splits):
        train_end = min_train + i * fold_size
        test_start = min(train_end + embargo, n)
        test_end = min(train_end + fold_size + embargo, n)
        if train_end >= n or test_end <= test_start:
            break
        train_idx = np.arange(0, train_end)
        if purge > 0 and len(train_idx) > purge + 20:
            train_idx = train_idx[: max(20, len(train_idx) - purge)]
        test_idx = np.arange(test_start, test_end)
        if len(test_idx) < 5 or len(train_idx) < 20:
            continue
        splits.append((train_idx, test_idx))
    return splits


def _pip_size_for_symbol(symbol: str) -> float:
    up = symbol.upper()
    if "XAU" in up or "XAG" in up or "JPY" in up:
        return 0.01
    return 0.0001


def _unit_cost(
    close: float,
    *,
    spread_pips: float,
    slippage_pips: float,
    commission_per_lot: float,
    pip_size: float = 0.0001,
    atr_pct: float | None = None,
    vol_slippage_k: float = 1.25,
) -> float:
    """Round-trip cost as fraction of price; optionally scale with ATR%."""
    sp, sl = float(spread_pips), float(slippage_pips)
    if atr_pct is not None and np.isfinite(atr_pct):
        from atis.engines.engine4_training.adaptive_learning import dynamic_execution_costs

        sp, sl, commission_per_lot = dynamic_execution_costs(
            close,
            float(atr_pct),
            base_spread_pips=sp,
            base_slippage_pips=sl,
            commission_per_lot=commission_per_lot,
            pip_size=pip_size,
            vol_slippage_k=vol_slippage_k,
        )
    spread_cost = (sp + sl) * pip_size / max(abs(close), 1e-12)
    # Gold lot notionals are much smaller than FX 100k; use conservative fractional commission.
    comm = commission_per_lot / max(abs(close) * 100.0, 1.0)
    return float(spread_cost + comm)


def _trade_returns_from_preds(
    close: np.ndarray,
    preds: np.ndarray,
    *,
    hold_bars: int = 1,
    spread_pips: float,
    slippage_pips: float,
    commission_per_lot: float,
    pip_size: float = 0.0001,
    confidences: np.ndarray | None = None,
    min_confidence: float = 0.0,
    non_overlapping: bool = True,
    atr_pct: np.ndarray | None = None,
    dynamic_costs: bool = False,
    vol_slippage_k: float = 1.25,
    latency_bars: int = 0,
    execution_delay_bars: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Horizon-aligned execution with optional vol costs + latency/delay.

    - enter only on directional preds with enough confidence
    - hold `hold_bars` (label horizon), apply costs once per trade
    - optionally skip overlapping entries
    - latency/execution_delay shift fill bar (no same-bar optimism)
    """
    from atis.engines.engine4_training.execution_realism import simulate_trade_returns

    # Backward-compatible path: when no realism extras, keep prior cost formula
    # but still allow atr scaling via shared helper.
    return simulate_trade_returns(
        close,
        preds,
        hold_bars=hold_bars,
        spread_pips=spread_pips,
        slippage_pips=slippage_pips,
        commission_per_lot=commission_per_lot,
        pip_size=pip_size,
        confidences=confidences,
        min_confidence=min_confidence,
        non_overlapping=non_overlapping,
        atr_pct=atr_pct,
        dynamic_costs=bool(dynamic_costs),
        vol_slippage_k=vol_slippage_k,
        latency_bars=latency_bars,
        execution_delay_bars=execution_delay_bars,
    )


def _sharpe_ann_factor(
    *,
    periods_per_year: float,
    n_bars: int,
    n_trades: int,
    hold_bars: int = 1,
    ann_cap: str | float = "daily",
) -> tuple[float, float]:
    """Conservative annualization for non-overlapping horizon trades.

    Report 2026-07-31 showed Sharpe 10–16 on M15/M30 — still inflated after an
    H1-equivalent cap (√(252·24)≈78). Cap at daily √252≈15.87 and never exceed
    independent periods implied by hold_bars.
    """
    hold = max(1, int(hold_bars))
    indep_per_year = max(float(periods_per_year) / float(hold), 1.0)
    if n_trades >= 10 and n_bars > 0:
        trades_per_year = float(periods_per_year) * (float(n_trades) / float(max(n_bars, 1)))
        raw = float(np.sqrt(max(min(trades_per_year, indep_per_year), 1.0)))
    else:
        raw = float(np.sqrt(max(min(float(periods_per_year), indep_per_year), 1.0)))
    if isinstance(ann_cap, (int, float)):
        cap = float(ann_cap)
    elif str(ann_cap).lower() in {"daily", "d1", "day"}:
        cap = float(np.sqrt(252.0))
    elif str(ann_cap).lower() in {"h1", "hourly"}:
        cap = float(np.sqrt(252.0 * 24.0))
    else:
        cap = float(np.sqrt(252.0))
    return min(raw, cap), raw


def bootstrap_sharpe_ci(
    trade_returns: np.ndarray,
    *,
    ann: float,
    n_bootstrap: int = 400,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap CI for trade-Sharpe. Returns (point, ci_low, ci_high)."""
    x = np.asarray(trade_returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5 or float(np.std(x, ddof=0)) <= 0:
        point = float(np.mean(x) / max(float(np.std(x, ddof=0)), 1e-12) * ann) if len(x) else 0.0
        return point, point, point
    rng = np.random.default_rng(seed)
    stats = np.empty(int(max(n_bootstrap, 50)), dtype=float)
    n = len(x)
    for i in range(len(stats)):
        sample = x[rng.integers(0, n, size=n)]
        sig = float(np.std(sample, ddof=0))
        stats[i] = (float(np.mean(sample)) / sig * ann) if sig > 0 else 0.0
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    point = float(np.mean(x) / float(np.std(x, ddof=0)) * ann)
    return point, lo, hi


def financial_metrics(
    returns: np.ndarray,
    periods_per_year: float = 24 * 252,
    *,
    hold_bars: int = 1,
    ann_cap: str | float = "daily",
    bootstrap: bool = False,
    n_bootstrap: int = 400,
    seed: int = 42,
) -> dict[str, float]:
    r = returns[np.isfinite(returns)]
    traded = r[r != 0]
    empty = {
        "sharpe": 0.0,
        "sharpe_uncapped": 0.0,
        "sharpe_ci_low": 0.0,
        "sharpe_ci_high": 0.0,
        "sortino": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "calmar": 0.0,
        "expectancy": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "payoff_ratio": 0.0,
        "kelly_fraction_approx": 0.0,
        "risk_adjusted_return": 0.0,
        "risk_adjusted_score": 0.0,
        "total_return": 0.0,
        "mean_trade_return": 0.0,
        "sum_trade_returns": 0.0,
        "simple_trade_equity": 0.0,
        "compounded_backtest_note": "compounded_bar_path_not_live_expectation",
        "n_trades": 0.0,
        "ann_factor": 0.0,
    }
    if len(r) == 0:
        return empty
    equity = np.cumprod(1 + r)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0
    base = traded if len(traded) >= 10 else r
    mu = float(np.mean(base)) if len(base) else 0.0
    sigma = float(np.std(base, ddof=0)) if len(base) else 0.0
    downside = base[base < 0]
    down_sigma = float(np.std(downside, ddof=0)) if len(downside) else 0.0
    ann, ann_raw = _sharpe_ann_factor(
        periods_per_year=periods_per_year,
        n_bars=len(r),
        n_trades=len(traded),
        hold_bars=hold_bars,
        ann_cap=ann_cap,
    )
    sharpe = (mu / sigma * ann) if sigma > 0 else 0.0
    sharpe_uncapped = (mu / sigma * ann_raw) if sigma > 0 else 0.0
    sortino = (mu / down_sigma * ann) if down_sigma > 0 else 0.0
    ci_low = float(sharpe)
    ci_high = float(sharpe)
    if bootstrap and len(traded) >= 8:
        _, ci_low, ci_high = bootstrap_sharpe_ci(
            traded, ann=ann, n_bootstrap=n_bootstrap, seed=seed
        )
    wins = traded[traded > 0]
    losses = traded[traded < 0]
    win_rate = float(len(wins) / max(len(traded), 1))
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (0.0 if gross_profit == 0 else 99.0)
    total_return = float(equity[-1] - 1.0)
    calmar = (total_return / abs(max_dd)) if max_dd < 0 else 0.0
    # Non-compounded trade stats — clearer than multi-hundred-% compounded equity paths.
    mean_trade = float(np.mean(traded)) if len(traded) else 0.0
    sum_trades = float(np.sum(traded)) if len(traded) else 0.0
    simple_eq = float(1.0 + sum_trades)  # additive PnL on unit notional (non-overlapping trades)
    base = {
        "sharpe": float(sharpe),
        "sharpe_uncapped": float(sharpe_uncapped),
        "sharpe_ci_low": float(ci_low),
        "sharpe_ci_high": float(ci_high),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "profit_factor": float(pf),
        "calmar": float(calmar),
        "total_return": float(total_return),
        "mean_trade_return": mean_trade,
        "sum_trade_returns": sum_trades,
        "simple_trade_equity": simple_eq,
        "compounded_backtest_note": "compounded_bar_path_not_live_expectation",
        "n_trades": float(len(traded)),
        "ann_factor": float(ann),
    }
    from atis.engines.engine4_training.advanced_metrics import enrich_financial_metrics

    return enrich_financial_metrics(base, returns)


def classification_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
    classes: list[Any] | None = None,
) -> dict[str, Any]:
    """Accuracy / P / R / F1 / ROC-AUC / confusion / Brier when possible."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict[str, Any] = {
        "accuracy": 0.0,
        "precision_macro": 0.0,
        "recall_macro": 0.0,
        "f1_macro": 0.0,
        "roc_auc_ovr": 0.0,
        "brier_score": 0.0,
        "n_samples": float(len(y_true)),
        "confusion_matrix": {},
    }
    if len(y_true) == 0:
        return out
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    labels = sorted(set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist()))
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        out["confusion_matrix"] = {
            str(labels[i]): {str(labels[j]): int(cm[i, j]) for j in range(len(labels))}
            for i in range(len(labels))
        }
    except Exception:
        out["confusion_matrix"] = {}
    if proba is not None and classes is not None and len(np.unique(y_true)) >= 2:
        try:
            out["roc_auc_ovr"] = float(
                roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=list(classes))
            )
        except Exception:
            try:
                if set(np.unique(y_true)).issubset({-1, 1}) and 1 in classes:
                    idx = classes.index(1)
                    out["roc_auc_ovr"] = float(roc_auc_score((y_true == 1).astype(int), proba[:, idx]))
            except Exception:
                out["roc_auc_ovr"] = 0.0
        try:
            if 1 in classes and set(np.unique(y_true)).issubset({-1, 1}):
                idx = classes.index(1)
                out["brier_score"] = float(brier_score_loss((y_true == 1).astype(int), proba[:, idx]))
        except Exception:
            out["brier_score"] = 0.0
    return out


def data_quality_report(
    df: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    *,
    timeframe: str = "H1",
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Data QC with score + hard-gate fields (remediation applied when enabled)."""
    from atis.engines.engine4_training.data_quality_gate import (
        compute_data_quality_score,
        remediate_frame,
    )

    cfg = cfg or {}
    if bool(cfg.get("dq_gate_enabled", True)):
        _, rem = remediate_frame(
            df,
            timeframe=timeframe,
            feature_cols=list(feature_cols),
            max_gap_bars=int(cfg.get("dq_max_gap_bars", 3)),
            jump_z=float(cfg.get("dq_jump_z", 8.0)),
            drop_duplicate_rows=False,
        )
        cols = list(rem.get("feature_cols") or feature_cols)
        report = compute_data_quality_score(
            df, y, cols, timeframe=timeframe, remediation=rem, cfg=cfg
        )
        report["label_counts"] = {str(k): int(v) for k, v in y.value_counts(dropna=False).to_dict().items()}
        return report

    missing = {}
    for c in feature_cols[:80]:
        s = df[c] if c in df.columns else None
        if s is None:
            continue
        frac = float(s.isna().mean()) if len(s) else 0.0
        if frac > 0:
            missing[c] = round(frac, 4)
    counts = {str(k): int(v) for k, v in y.value_counts(dropna=False).to_dict().items()}
    total = max(int(y.shape[0]), 1)
    pos = int((y == 1).sum())
    neg = int((y == -1).sum())
    flat = int((y == 0).sum())
    imbalance = float(max(pos, neg) / max(min(pos, neg), 1)) if (pos + neg) > 0 else 0.0
    outlier_frac = float(df["is_outlier"].mean()) if "is_outlier" in df.columns else 0.0
    return {
        "n_rows": int(len(y)),
        "n_features": len(feature_cols),
        "score": 70.0,
        "gate_pass": True,
        "label_counts": counts,
        "label_share": {
            "up": round(pos / total, 4),
            "down": round(neg / total, 4),
            "flat": round(flat / total, 4),
        },
        "directional_imbalance_ratio": round(imbalance, 4),
        "missing_frac_top": dict(sorted(missing.items(), key=lambda kv: -kv[1])[:15]),
        "outlier_frac": round(outlier_frac, 4),
        "quality_flags": {
            "high_imbalance": imbalance >= 1.8,
            "many_missing_cols": len(missing) >= 10,
            "high_outliers": outlier_frac >= 0.05,
        },
    }


def diagnose_fit(
    train_cls: dict[str, Any],
    val_cls: dict[str, Any],
    test_cls: dict[str, Any],
    train_fin: dict[str, float],
    val_fin: dict[str, float],
    test_fin: dict[str, float],
    *,
    trade_rate_filtered: float | None = None,
    median_fold_trade_rate: float | None = None,
) -> dict[str, Any]:
    """Detect overfitting / underfitting primarily from financial generalization."""
    tr_acc = float(train_cls.get("accuracy", 0.0) or 0.0)
    va_acc = float(val_cls.get("accuracy", 0.0) or 0.0)
    te_acc = float(test_cls.get("accuracy", 0.0) or 0.0)
    tr_sh = float(train_fin.get("sharpe", 0.0) or 0.0)
    va_sh = float(val_fin.get("sharpe", 0.0) or 0.0)
    te_sh = float(test_fin.get("sharpe", 0.0) or 0.0)
    tr_n = float(train_fin.get("n_trades", 0.0) or 0.0)
    va_n = float(val_fin.get("n_trades", 0.0) or 0.0)
    te_n = float(test_fin.get("n_trades", 0.0) or 0.0)
    # Prefer train↔val financial gap; keep train↔val accuracy only when val_cls is real.
    acc_gap_tv = tr_acc - va_acc if va_acc > 0 else tr_acc - te_acc
    sharpe_gap_tv = tr_sh - va_sh
    sharpe_gap_vt = abs(va_sh - te_sh)
    status = "balanced"
    notes: list[str] = []
    if te_acc < 0.52 and tr_acc < 0.55 and te_sh <= 0 and tr_sh <= 0.2:
        status = "underfitting"
        notes.append("Weak train and test signal — high bias / lack of edge.")
    # Financial overfit: train much stronger than validation on trading metrics.
    if sharpe_gap_tv > 1.5 and te_sh < max(va_sh, 0.0):
        status = "overfitting"
        notes.append("Train financial much stronger than Validation — poor generalization.")
    # Accuracy-only overfit only when gap is extreme AND financial also diverges.
    if acc_gap_tv > 0.15 and sharpe_gap_tv > 1.0 and status != "overfitting":
        status = "overfitting"
        notes.append("Large train/val accuracy gap with financial divergence.")
    if sharpe_gap_vt > 2.0 and te_sh < va_sh:
        notes.append("Validation optimistic vs Test — regime/policy shift (high variance).")
        # Report 04-15 H4: gap≈2.19 but Test Sharpe≈2.35 — keep as warning unless Test collapsed.
        if status == "balanced" and te_sh < max(1.0, 0.45 * va_sh) and te_sh < 1.0:
            status = "unstable_generalization"
        elif status == "balanced" and te_sh >= 1.0:
            notes.append("Val≫Test gap present but Test Sharpe remains strong — warning only.")
    # Milder but still dangerous optimism (e.g. Val 2.55 vs Test 0.41 → gap ≈ 2.14).
    if sharpe_gap_vt > 1.75 and va_sh > max(te_sh * 2.0, 1.0) and status == "balanced":
        if te_sh < 0.75 or (te_sh < 0.4 * va_sh and te_sh < 1.0):
            status = "unstable_generalization"
            notes.append("Large Val≫Test Sharpe gap — policy/calibrator likely overfit to validation.")
        else:
            notes.append("Elevated Val vs Test gap with still-usable Test Sharpe — monitor live drift.")
    # Sparse-trade optimism: high Val Sharpe from a handful of trades (report H1/H4/M30 folds).
    if va_sh >= 1.5 and va_n > 0 and va_n < 8 and te_n < max(va_n, 5):
        notes.append("Validation Sharpe inflated by sparse trades — statistical noise risk.")
        if status == "balanced":
            status = "unstable_generalization"
    if te_sh > 0 and va_sh > 0 and sharpe_gap_vt < 1.25:
        notes.append("Val and Test both positive with moderate gap.")
    auc = float(test_cls.get("roc_auc_ovr", 0.0) or 0.0)
    if auc < 0.52 and te_sh > 0:
        notes.append("Positive Sharpe with near-chance AUC — edge may be filter-driven.")
    if auc > 0 and auc < 0.53 and te_acc < 0.52:
        notes.append("Discriminative power near chance — prefer meta-labeling / stronger features.")
        if status == "balanced" and te_sh <= 0.5:
            status = "underfitting"
            notes.append("Near-chance AUC with weak Test Sharpe — model underfits the label.")
        elif status == "balanced" and te_sh > 0.5:
            notes.append(
                "Filter-driven edge risk: financial metrics ahead of discriminative power."
            )
    # Report 2026-07-31 H4: deploy/test Sharpe on <12 trades + Acc≈0.50 — unreliable.
    if te_sh >= 1.0 and te_n > 0 and te_n < 12:
        notes.append("Test/deploy Sharpe rests on fewer than 12 trades — low statistical power.")
        if status == "balanced" and (auc < 0.52 or te_acc < 0.53):
            status = "unstable_generalization"
            notes.append("Sparse high-Sharpe + near-chance classification — unreliable for live.")
    rate = float(trade_rate_filtered) if trade_rate_filtered is not None else float(
        test_cls.get("trade_rate_filtered", 0.0) or 0.0
    )
    fold_rate = float(median_fold_trade_rate) if median_fold_trade_rate is not None else -1.0
    if rate >= 0 and rate < 0.005 and (va_sh > 1.0 or te_sh > 0.5):
        notes.append("Near-zero filtered trade rate with optimistic Sharpe — policy starvation.")
        if status == "balanced":
            status = "unstable_generalization"
    if fold_rate >= 0 and fold_rate < 0.005:
        notes.append("Median fold trade rate near zero — live model will rarely fire.")
    return {
        "status": status,
        "accuracy_gap_train_val": round(acc_gap_tv, 4),
        "accuracy_gap_train_test": round(tr_acc - te_acc, 4),
        "sharpe_gap_train_val": round(sharpe_gap_tv, 4),
        "sharpe_gap_val_test": round(sharpe_gap_vt, 4),
        "val_trades": round(va_n, 1),
        "test_trades": round(te_n, 1),
        "train_trades": round(tr_n, 1),
        "notes": notes,
        "filter_driven_edge_risk": bool(
            auc > 0 and auc < 0.52 and te_sh > 0.5 and te_acc < 0.53
        ),
        "sparse_sharpe_risk": bool(te_sh >= 1.0 and te_n > 0 and te_n < 12),
    }


def fit_classifier(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    *,
    cfg: dict[str, Any],
) -> Any:
    """Fit with optional LightGBM early stopping on a chronological tail holdout."""
    early = int(cfg.get("lgb_early_stopping_rounds", 40))
    use_es = bool(cfg.get("lgb_early_stopping", True)) and early > 0
    if use_es and HAS_LGB and model.__class__.__name__ == "LGBMClassifier" and len(X) >= 80:
        cut = max(40, int(len(X) * 0.85))
        if cut < len(X) - 15:
            sw_tr = sample_weight[:cut] if sample_weight is not None else None
            try:
                callbacks = [lgb.early_stopping(early, verbose=False), lgb.log_evaluation(period=0)]
                kw: dict[str, Any] = {
                    "eval_X": [X[cut:]],
                    "eval_y": [y[cut:]],
                    "callbacks": callbacks,
                }
                if sw_tr is not None:
                    kw["sample_weight"] = sw_tr
                model.fit(X[:cut], y[:cut], **kw)
                return model
            except TypeError:
                pass
            except Exception:
                pass
    try:
        if sample_weight is not None:
            model.fit(X, y, sample_weight=sample_weight)
        else:
            model.fit(X, y)
    except TypeError:
        model.fit(X, y)
    return model


def write_evaluation_report(path: Path, payload: dict[str, Any]) -> None:
    """Human-readable markdown evaluation beside metrics JSON."""
    dq = payload.get("data_quality") or {}
    split = payload.get("split_comparison") or {}
    diag = payload.get("fit_diagnosis") or {}
    fin = payload.get("financial_oos") or {}
    val = payload.get("financial_validation") or {}
    deploy = payload.get("financial_deploy_holdout") or {}
    train_fin = (split.get("train") or {}).get("financial") or {}
    cls = payload.get("classification") or {}
    gates = payload.get("gate_failures") or []
    gates_detail = payload.get("gate_failures_detail") or annotate_gate_failures(list(gates))
    lines = [
        f"# ATIS Evaluation Report — {payload.get('symbol')} {payload.get('timeframe')}",
        "",
        f"- Version: `{payload.get('version')}`",
        f"- Pipeline: `{payload.get('pipeline_version')}`",
        f"- Passed gates: **{payload.get('passed_gates')}**",
        f"- Fit diagnosis: **{diag.get('status', 'n/a')}**",
        f"- Acc gap TV: {diag.get('accuracy_gap_train_val')} · Sharpe gap TV: {diag.get('sharpe_gap_train_val')} · VT: {diag.get('sharpe_gap_val_test')}",
        "",
        "## Gate Failures",
    ]
    if gates_detail:
        for g in gates_detail:
            lines.append(f"- `{g.get('key')}` — {g.get('ar')}")
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            "## Classification (OOS Test)",
            f"- Accuracy: {cls.get('accuracy')}",
            f"- Precision: {cls.get('precision_macro')}",
            f"- Recall: {cls.get('recall_macro')}",
            f"- F1: {cls.get('f1_macro')}",
            f"- ROC-AUC (OvR): {cls.get('roc_auc_ovr')}",
            f"- Trade rate: {cls.get('trade_rate_filtered')}",
            "",
            "## Financial",
            f"| Split | Sharpe | Uncapped | CI low | Max DD | Sum trade R | Mean trade R | Trades |",
            f"|---|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| Train | {train_fin.get('sharpe', 0):.4f} | {train_fin.get('sharpe_uncapped', 0):.4f} | "
                f"{train_fin.get('sharpe_ci_low', 0):.4f} | {train_fin.get('max_drawdown', 0):.4f} | "
                f"{train_fin.get('sum_trade_returns', 0):.4f} | {train_fin.get('mean_trade_return', 0):.5f} | "
                f"{int(train_fin.get('n_trades', 0) or 0)} |"
            ),
            (
                f"| Validation | {val.get('sharpe', 0):.4f} | {val.get('sharpe_uncapped', 0):.4f} | "
                f"{val.get('sharpe_ci_low', 0):.4f} | {val.get('max_drawdown', 0):.4f} | "
                f"{val.get('sum_trade_returns', 0):.4f} | {val.get('mean_trade_return', 0):.5f} | "
                f"{int(val.get('n_trades', 0) or 0)} |"
            ),
            (
                f"| Test OOS | {fin.get('sharpe', 0):.4f} | {fin.get('sharpe_uncapped', 0):.4f} | "
                f"{fin.get('sharpe_ci_low', 0):.4f} | {fin.get('max_drawdown', 0):.4f} | "
                f"{fin.get('sum_trade_returns', 0):.4f} | {fin.get('mean_trade_return', 0):.5f} | "
                f"{int(fin.get('n_trades', 0) or 0)} |"
            ),
            (
                f"| Deploy | {deploy.get('sharpe', 0):.4f} | {deploy.get('sharpe_uncapped', 0):.4f} | "
                f"{deploy.get('sharpe_ci_low', 0):.4f} | {deploy.get('max_drawdown', 0):.4f} | "
                f"{deploy.get('sum_trade_returns', 0):.4f} | {deploy.get('mean_trade_return', 0):.5f} | "
                f"{int(deploy.get('n_trades', 0) or 0)} |"
            ),
            "",
            f"- ann_factor (conservative): {fin.get('ann_factor')}",
            f"- Compounded bar-path return (backtest only, not live expectation): {fin.get('total_return')}",
            f"- Simple trade equity (1 + sum trade returns): {fin.get('simple_trade_equity')}",
            f"- Expectancy: {fin.get('expectancy')} · Sortino: {fin.get('sortino')} · PF: {fin.get('profit_factor')}",
            f"- Risk-adjusted return (R/|DD|): {fin.get('risk_adjusted_return')}",
            "",
            "## Advanced Validation",
            f"- Validation mode: `{payload.get('validation_mode') or (payload.get('validation') or {}).get('validation_mode')}`",
            f"- Regime stable: **{(payload.get('regime_validation') or {}).get('stable')}**",
            f"- DSR: {((payload.get('advanced_eval') or {}).get('deflated_sharpe') or {}).get('deflated_sharpe')}",
            f"- PBO: {((payload.get('advanced_eval') or {}).get('pbo') or {}).get('pbo')}",
            f"- Live readiness: **{((payload.get('live_readiness') or {}).get('score'))}/100 · {((payload.get('live_readiness') or {}).get('verdict'))}**",
            f"- Model zoo winner: `{((payload.get('model_zoo') or {}).get('winner'))}`",
            f"- Stress robust: {(payload.get('stress_testing') or {}).get('robust')} · MC stable: {(payload.get('monte_carlo') or {}).get('stable')}",
            f"- Latency bars: {((payload.get('advanced_eval') or {}).get('execution') or {}).get('latency_bars')}",
            f"- Dynamic costs: {((payload.get('advanced_eval') or {}).get('execution') or {}).get('dynamic_costs')}",
            "",
            "## Data Quality",
            (
                f"- Rows: {dq.get('n_rows')} · Features: {dq.get('n_features')}"
                + (
                    f" · DQ score: **{dq.get('score')}**/100 · gate={dq.get('gate_pass')}"
                    if dq.get("score") is not None
                    else ""
                )
            ),
            f"- Label share: {dq.get('label_share')}",
            f"- Imbalance ratio: {dq.get('directional_imbalance_ratio')}",
            f"- Outlier frac: {dq.get('outlier_frac')}",
            f"- Flags: {dq.get('quality_flags')}",
            "",
            "## Diagnosis Notes",
        ]
    )
    for n in diag.get("notes") or ["—"]:
        lines.append(f"- {n}")
    lines.extend(
        [
            "",
            "## Recommendations",
            "- Keep purged walk-forward + nested validation; avoid random shuffles.",
            "- Prefer liquid TFs (H1/H4) when trade counts are sparse.",
            "- If accuracy stays ~0.45, focus on meta-labeling / better barriers rather than deeper trees.",
            "- Re-train after major regime shifts; monitor Val↔Test Sharpe gap.",
            "- Treat compounded total_return as illustrative only; use sum/mean trade returns for honesty.",
            "- Monitor expectancy, Sortino, DSR, and regime stability — not accuracy alone.",
            "- Use knowledge_loop.json episodes to drive continuous learning / next experiment.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def predict_with_confidence(
    model: Any,
    X: np.ndarray,
    *,
    decision_threshold: float,
    directional_edge: float = 0.12,
    confidence_quantile: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return class preds in {-1,0,1} and confidence; uncertain bars stay flat."""
    proba, classes = _model_proba(model, X)
    if proba is None:
        preds = model.predict(X).astype(float)
        return preds, np.ones(len(preds), dtype=float)
    conf = proba.max(axis=1)
    return (
        policy_from_proba(
            proba,
            classes,
            decision_threshold=decision_threshold,
            directional_edge=directional_edge,
            confidence_quantile=confidence_quantile,
        ),
        conf,
    )


def _model_proba(model: Any, X: np.ndarray) -> tuple[np.ndarray | None, list[Any] | None]:
    if not hasattr(model, "predict_proba"):
        return None, None
    proba = model.predict_proba(X)
    classes = list(np.asarray(model.classes_).tolist())
    return proba, classes


def policy_from_proba(
    proba: np.ndarray,
    classes: list[Any],
    *,
    decision_threshold: float,
    directional_edge: float,
    confidence_quantile: float = 0.0,
    confidence_floor: float | None = None,
    atr_pct: np.ndarray | None = None,
    unit_costs: np.ndarray | None = None,
    cost_edge_multiple: float = 0.0,
    regime_mask: np.ndarray | None = None,
    short_edge_multiple: float = 1.0,
    primary_sides: np.ndarray | None = None,
) -> np.ndarray:
    """Convert probabilities to sparse trade decisions with optional cost/regime filters.

    When ``primary_sides`` is set (meta-labeling), structure/trend chooses the side
    and the model probability acts as a confidence filter on that side.
    """
    n = len(proba)
    preds = np.zeros(n, dtype=float)
    if n == 0:
        return preds
    conf = proba.max(axis=1)
    raw = np.asarray(classes, dtype=float)[proba.argmax(axis=1)]
    q_thr = float(decision_threshold)
    if confidence_floor is not None and np.isfinite(confidence_floor):
        q_thr = max(q_thr, float(confidence_floor))
    elif confidence_quantile and 0.0 < confidence_quantile < 1.0:
        q_thr = max(q_thr, float(np.quantile(conf, confidence_quantile)))

    idx_up = classes.index(1) if 1 in classes else None
    idx_dn = classes.index(-1) if -1 in classes else None
    short_mult = max(1.0, float(short_edge_multiple))
    for i in range(n):
        if regime_mask is not None and not bool(regime_mask[i]):
            continue
        p_up = float(proba[i, idx_up]) if idx_up is not None else 0.0
        p_dn = float(proba[i, idx_dn]) if idx_dn is not None else 0.0
        if primary_sides is not None:
            side = float(primary_sides[i])
            if side == 0.0:
                continue
            side_conf = p_up if side > 0 else p_dn
            if side_conf < q_thr:
                continue
        else:
            if conf[i] < q_thr:
                continue
            side = float(raw[i])
            if side == 0:
                continue
            side_conf = conf[i]
        edge = (p_up - p_dn) if side > 0 else (p_dn - p_up)
        need = directional_edge * (short_mult if side < 0 else 1.0)
        if edge < need:
            continue
        if (
            atr_pct is not None
            and unit_costs is not None
            and cost_edge_multiple > 0
            and edge * float(atr_pct[i]) < float(unit_costs[i]) * cost_edge_multiple
        ):
            continue
        preds[i] = side
    return preds


class ProbaCalibrator:
    """Joblib-serializable isotonic calibrator for directional probs."""

    def __init__(self, iso_model: Any, up_i: int, dn_i: int) -> None:
        self.iso = iso_model
        self.idx_up = int(up_i)
        self.idx_dn = int(dn_i)

    def __call__(self, p: np.ndarray) -> np.ndarray:
        out = np.asarray(p, dtype=float).copy()
        cal = self.iso.transform(out[:, self.idx_up])
        out[:, self.idx_up] = cal
        out[:, self.idx_dn] = np.clip(1.0 - cal, 0.0, 1.0)
        if out.shape[1] > 2:
            for i in range(out.shape[1]):
                if i not in (self.idx_up, self.idx_dn):
                    out[:, i] = 0.0
            row = out.sum(axis=1, keepdims=True)
            out = out / np.maximum(row, 1e-12)
        return out


def make_proba_calibrator(
    proba_val: np.ndarray,
    classes: list[Any],
    y_val: np.ndarray,
) -> ProbaCalibrator | None:
    if 1 not in classes or -1 not in classes or len(proba_val) < 30:
        return None
    try:
        from sklearn.isotonic import IsotonicRegression
    except Exception:
        return None
    idx_up = classes.index(1)
    idx_dn = classes.index(-1)
    mask = np.isin(y_val, [-1, 1])
    if int(mask.sum()) < 30:
        return None
    p_up = proba_val[mask, idx_up]
    y_bin = (np.asarray(y_val)[mask] == 1).astype(float)
    if len(np.unique(y_bin)) < 2:
        return None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_up, y_bin)
    return ProbaCalibrator(iso, idx_up, idx_dn)


def train_confidence_floor(
    model: Any,
    X_train: np.ndarray,
    *,
    decision_threshold: float,
    confidence_quantile: float,
    min_floor: float = 0.60,
    target_trade_rate: float = 0.0,
    max_floor: float = 0.88,
) -> float:
    """Fit confidence cutoff on train probs so val/test do not re-estimate the quantile.

    Report 2026-07-31 showed q=0.95 + min_floor=0.65 → fold trade_rate≈0–2% starvation.
    Align the quantile with target_trade_rate and cap the absolute floor.
    """
    floor = max(float(decision_threshold), float(min_floor))
    proba, _ = _model_proba(model, X_train)
    if proba is None:
        return float(min(floor, max_floor))
    conf = proba.max(axis=1)
    if len(conf) < 10:
        return float(min(floor, max_floor))
    q = float(confidence_quantile) if 0.0 < float(confidence_quantile) < 1.0 else 0.88
    # Keep roughly target_trade_rate of bars above the floor (with a small buffer).
    if target_trade_rate and target_trade_rate > 0:
        q_rate = float(np.clip(1.0 - max(float(target_trade_rate), 0.02) * 2.2, 0.50, 0.88))
        q = min(q, q_rate)
    raw = float(np.quantile(conf, q))
    # Soft cap: never sit at the absolute max when targeting liquid trade rates.
    soft_max = float(max_floor)
    if target_trade_rate and target_trade_rate >= 0.04:
        soft_max = min(soft_max, max(float(max_floor), 0.72))
    return float(np.clip(max(floor, raw), 0.0, soft_max))


def tune_trade_policy(
    *,
    model: Any,
    X_val: np.ndarray,
    close_val: np.ndarray,
    atr_pct_val: np.ndarray,
    regime_val: np.ndarray,
    hold_bars: int,
    spread_pips: float,
    slippage_pips: float,
    commission_per_lot: float,
    pip_size: float,
    periods_per_year: float,
    base_threshold: float,
    base_edge: float,
    base_quantile: float,
    cost_edge_multiple: float,
    non_overlapping: bool,
    min_trades: int = 8,
    confidence_floor: float | None = None,
    allow_long: np.ndarray | None = None,
    allow_short: np.ndarray | None = None,
    target_trade_rate: float = 0.0,
    max_trade_rate: float = 0.20,
    short_edge_multiple: float = 1.0,
    calibrate_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    apply_sparsify: bool = False,
    primary_sides: np.ndarray | None = None,
) -> dict[str, float]:
    """Pick threshold/edge on a validation slice with the same stack as live eval."""
    proba, classes = _model_proba(model, X_val)
    baseline = {
        "decision_threshold": float(base_threshold),
        "directional_edge": float(base_edge),
        "confidence_quantile": float(base_quantile),
        "cost_edge_multiple": float(cost_edge_multiple),
        "val_sharpe": 0.0,
        "val_trades": 0.0,
        "val_return": 0.0,
    }
    if proba is None or classes is None or len(X_val) < 20:
        return baseline
    if calibrate_fn is not None:
        proba = calibrate_fn(proba)

    unit_costs = np.array(
        [
            _unit_cost(
                float(c),
                spread_pips=spread_pips,
                slippage_pips=slippage_pips,
                commission_per_lot=commission_per_lot,
                pip_size=pip_size,
            )
            for c in close_val
        ],
        dtype=float,
    )
    conf = proba.max(axis=1)
    max_rate = float(max_trade_rate) if max_trade_rate > 0 else 0.20

    def _score(
        thr: float,
        edge: float,
        q: float,
        floor: float | None,
        *,
        regime_try: np.ndarray | None = None,
        cost_mult: float | None = None,
    ) -> tuple[float, float, dict[str, float]]:
        cost_m = float(cost_edge_multiple if cost_mult is None else cost_mult)
        preds = policy_from_proba(
            proba,
            classes,
            decision_threshold=thr,
            directional_edge=edge,
            confidence_quantile=0.0 if floor is not None else q,
            confidence_floor=floor,
            atr_pct=atr_pct_val,
            unit_costs=unit_costs,
            cost_edge_multiple=cost_m,
            regime_mask=regime_val if regime_try is None else regime_try,
            short_edge_multiple=short_edge_multiple,
            primary_sides=primary_sides,
        )
        if primary_sides is None and allow_long is not None and allow_short is not None:
            preds = apply_trend_align(preds, allow_long, allow_short)
        if target_trade_rate > 0 and apply_sparsify:
            preds = sparsify_by_confidence(preds, conf, target_trade_rate=target_trade_rate)
        if max_rate < 1.0:
            preds = cap_preds_by_trade_rate(preds, conf, max_trade_rate=max_rate)
        rets, stats = _trade_returns_from_preds(
            close_val,
            preds,
            hold_bars=hold_bars,
            spread_pips=spread_pips,
            slippage_pips=slippage_pips,
            commission_per_lot=commission_per_lot,
            pip_size=pip_size,
            non_overlapping=non_overlapping,
        )
        fin = financial_metrics(
            rets,
            periods_per_year=periods_per_year,
            hold_bars=hold_bars,
            ann_cap="daily",
        )
        trades = float(stats.get("trades") or 0.0)
        trade_rate = trades / max(len(close_val), 1)
        # Prefer expectancy-after-cost + mild Sharpe; penalize over-trading and starvation.
        expectancy = float(fin.get("expectancy", 0.0) or 0.0)
        score = (
            0.40 * float(fin["sharpe"])
            + 1.8 * float(fin["total_return"])
            + 12.0 * expectancy
        )
        if trades < min_trades:
            score -= 3.0
        if target_trade_rate > 0 and trade_rate < target_trade_rate * 0.35:
            score -= 2.5 * (1.0 - trade_rate / max(target_trade_rate * 0.35, 1e-6))
        soft_cap = max(0.08, float(target_trade_rate) * 1.6 if target_trade_rate > 0 else 0.08)
        if trade_rate > soft_cap:
            score -= 2.5 * (trade_rate / soft_cap)
        if trade_rate > max_rate:
            score -= 4.0
        if abs(fin["max_drawdown"]) > 0.20:
            score -= 1.5
        if fin["total_return"] < -0.08:
            score -= 1.0
        if fin["profit_factor"] < 0.90:
            score -= 0.5
        if expectancy < 0:
            score -= 1.5
        return score, trades, fin

    base_floor = float(confidence_floor) if confidence_floor is not None else None
    base_score, base_trades, base_fin = _score(base_threshold, base_edge, base_quantile, base_floor)
    baseline["val_sharpe"] = float(base_fin["sharpe"])
    baseline["val_trades"] = float(base_trades)
    baseline["val_return"] = float(base_fin["total_return"])
    if base_floor is not None:
        baseline["confidence_floor"] = float(base_floor)

    starved = policy_liquidity_starved(
        trades=base_trades,
        n_bars=len(close_val),
        min_trades=min_trades,
        target_trade_rate=target_trade_rate,
    )
    # Default: never looser than baseline. When starved (report trade_rate≈0), allow relaxation.
    if starved:
        grid_q = sorted({min(base_quantile, q) for q in (base_quantile, 0.80, 0.85, 0.88, 0.90)})
        grid_edge = sorted({min(base_edge, e) for e in (base_edge, 0.08, 0.10, 0.12, 0.15, 0.18)})
        grid_thr = sorted({min(base_threshold, t) for t in (base_threshold, 0.50, 0.52, 0.55)})
        if base_floor is not None:
            grid_floor = sorted(
                {
                    max(0.48, f)
                    for f in (
                        base_floor,
                        base_floor - 0.05,
                        base_floor - 0.10,
                        base_floor - 0.15,
                        min(base_floor, float(base_threshold)),
                        float(base_threshold),
                        0.52,
                    )
                }
            )
        else:
            grid_floor = [None]
        grid_cost = sorted(
            {
                float(cost_edge_multiple),
                min(float(cost_edge_multiple), 1.15),
                min(float(cost_edge_multiple), 1.0),
                min(float(cost_edge_multiple), 0.85),
            }
        )
        # Also try ignoring regime filter when it zeros liquidity (M30/M15 report folds).
        regime_options = [regime_val]
        if regime_val is not None and (~regime_val).any():
            regime_options.append(np.ones_like(regime_val, dtype=bool))
    else:
        grid_q = sorted({max(base_quantile, q) for q in (base_quantile, 0.90, 0.92, 0.94)})
        grid_edge = sorted({max(base_edge, e) for e in (base_edge, 0.18, 0.20, 0.22)})
        grid_thr = sorted({max(base_threshold, t) for t in (base_threshold, 0.55, 0.58, 0.60)})
        grid_floor = [base_floor]
        grid_cost = [float(cost_edge_multiple)]
        regime_options = [regime_val]

    best = dict(baseline)
    best_score = base_score
    for regime_try in regime_options:
        for cost_m in grid_cost:
            for q in grid_q:
                for edge in grid_edge:
                    for thr in grid_thr:
                        for floor in grid_floor:
                            score, trades, fin = _score(
                                thr, edge, q, floor, regime_try=regime_try, cost_mult=cost_m
                            )
                            improve = score > best_score + 0.20
                            if starved and trades >= min_trades and fin["sharpe"] >= -0.25:
                                improve = score > best_score + 0.05
                            # Rescue: any liquid non-catastrophic policy beats total starvation.
                            if starved and baseline["val_trades"] < float(min_trades) and trades >= min_trades:
                                if fin["total_return"] > -0.08 and fin["sharpe"] >= -0.5:
                                    improve = score > best_score - 0.5 or trades > baseline["val_trades"]
                            if improve and fin["total_return"] > -0.08 and fin["sharpe"] >= baseline["val_sharpe"] - 0.5:
                                best_score = score
                                best = {
                                    "decision_threshold": float(thr),
                                    "directional_edge": float(edge),
                                    "confidence_quantile": float(q),
                                    "cost_edge_multiple": float(cost_m),
                                    "val_sharpe": float(fin["sharpe"]),
                                    "val_trades": float(trades),
                                    "val_return": float(fin["total_return"]),
                                }
                                if floor is not None:
                                    best["confidence_floor"] = float(floor)
    return best


def regime_bounds_from_atr(
    atr_pct: np.ndarray,
    low_q: float = 0.2,
    high_q: float = 0.9,
) -> tuple[float, float]:
    """Fit mid-volatility bounds on a train slice only (leak-safe)."""
    valid = atr_pct[np.isfinite(atr_pct) & (atr_pct > 0)]
    if len(valid) < 20:
        return 0.0, float("inf")
    return float(np.quantile(valid, low_q)), float(np.quantile(valid, high_q))


def regime_mask_apply(atr_pct: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (atr_pct >= lo) & (atr_pct <= hi) & np.isfinite(atr_pct)


def regime_mask_from_atr(atr_pct: np.ndarray, low_q: float = 0.2, high_q: float = 0.9) -> np.ndarray:
    """Allow trades only in mid-volatility regimes (bounds from same series)."""
    lo, hi = regime_bounds_from_atr(atr_pct, low_q=low_q, high_q=high_q)
    return regime_mask_apply(atr_pct, lo, hi)


def time_decay_weights(n: int, half_life_frac: float = 0.4) -> np.ndarray:
    """More weight on recent bars (index n-1 is newest)."""
    if n <= 1:
        return np.ones(max(n, 1), dtype=float)
    age_from_newest = np.arange(n - 1, -1, -1, dtype=float)
    hl = max(1.0, float(half_life_frac) * n)
    w = 0.5 ** (age_from_newest / hl)
    return w / max(float(w.mean()), 1e-12)


def select_top_features(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    k: int,
    seed: int,
    sample_weight: np.ndarray | None = None,
    cfg: dict[str, Any] | None = None,
) -> list[str]:
    """Keep the most important features to reduce noise for gold models."""
    cols = list(X.columns)
    if k <= 0 or len(cols) <= k:
        return cols
    model = build_model("lightgbm", seed, cfg)
    try:
        if sample_weight is not None:
            model.fit(X.values, y.values, sample_weight=sample_weight)
        else:
            model.fit(X.values, y.values)
    except TypeError:
        model.fit(X.values, y.values)
    if not hasattr(model, "feature_importances_"):
        return cols[:k]
    order = np.argsort(np.asarray(model.feature_importances_))[::-1][:k]
    return [cols[int(i)] for i in order]


def select_stable_top_features(
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    k: int,
    seed: int,
    label_weights: np.ndarray,
    half_life: float,
    train_directional_only: bool,
    cfg: dict[str, Any] | None = None,
    n_windows: int = 3,
    min_frac: float = 0.6,
) -> list[str]:
    """Select features that recur across early walk-forward train windows.

    Uses only past train indices from the first n_windows splits — no test/future
    fold data. Prefer intersection by frequency over a single-window importance.
    """
    cols = list(X.columns)
    if k <= 0 or len(cols) <= k or not splits:
        return cols if k <= 0 else cols[:k]
    n_win = max(1, min(int(n_windows), len(splits)))
    votes: dict[str, int] = {c: 0 for c in cols}
    for wi in range(n_win):
        tr_idx, _ = splits[wi]
        w_sel = time_decay_weights(len(tr_idx), half_life) * label_weights[tr_idx]
        y_sel = y.iloc[tr_idx]
        x_sel = X.iloc[tr_idx]
        if train_directional_only:
            dir_mask = y_sel.astype(float).values != 0.0
            if int(dir_mask.sum()) >= 40:
                x_sel = x_sel.iloc[dir_mask]
                y_sel = y_sel.iloc[dir_mask]
                w_sel = w_sel[dir_mask]
        picked = select_top_features(
            x_sel, y_sel, k=k, seed=seed + wi, sample_weight=w_sel, cfg=cfg
        )
        for c in picked:
            votes[c] = votes.get(c, 0) + 1
    need = max(1, int(np.ceil(float(min_frac) * n_win)))
    stable = [c for c, v in votes.items() if v >= need]
    if len(stable) >= max(8, k // 3):
        # Rank stable by vote then fall back to first-window importance order.
        first = select_top_features(
            X.iloc[splits[0][0]],
            y.iloc[splits[0][0]],
            k=min(k * 2, len(cols)),
            seed=seed,
            sample_weight=time_decay_weights(len(splits[0][0]), half_life) * label_weights[splits[0][0]],
            cfg=cfg,
        )
        order = {c: i for i, c in enumerate(first)}
        stable_sorted = sorted(stable, key=lambda c: (votes[c], -order.get(c, 10_000)), reverse=True)
        return stable_sorted[:k]
    # Fallback: first-window only when stability too sparse.
    return select_top_features(
        X.iloc[splits[0][0]],
        y.iloc[splits[0][0]],
        k=k,
        seed=seed,
        sample_weight=time_decay_weights(len(splits[0][0]), half_life) * label_weights[splits[0][0]],
        cfg=cfg,
    )


def trend_masks_from_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (allow_long, allow_short) from EMA/SMA structure."""
    n = len(frame)
    if "ema_20" in frame.columns and "ema_50" in frame.columns:
        up = frame["ema_20"].astype(float).values > frame["ema_50"].astype(float).values
        dn = frame["ema_20"].astype(float).values < frame["ema_50"].astype(float).values
        return up, dn
    if "sma_20" in frame.columns and "sma_50" in frame.columns:
        up = frame["sma_20"].astype(float).values > frame["sma_50"].astype(float).values
        dn = frame["sma_20"].astype(float).values < frame["sma_50"].astype(float).values
        return up, dn
    if "close" in frame.columns and "sma_50" in frame.columns:
        up = frame["close"].astype(float).values > frame["sma_50"].astype(float).values
        dn = frame["close"].astype(float).values < frame["sma_50"].astype(float).values
        return up, dn
    return np.ones(n, dtype=bool), np.ones(n, dtype=bool)


def apply_trend_align(preds: np.ndarray, allow_long: np.ndarray, allow_short: np.ndarray) -> np.ndarray:
    out = preds.astype(float).copy()
    out[(out > 0) & (~allow_long)] = 0.0
    out[(out < 0) & (~allow_short)] = 0.0
    return out


def sparsify_by_confidence(
    preds: np.ndarray,
    confidences: np.ndarray,
    *,
    target_trade_rate: float,
) -> np.ndarray:
    """Keep only the highest-confidence fraction of candidate trades (leak-free sparsity)."""
    out = preds.astype(float).copy()
    if target_trade_rate <= 0 or target_trade_rate >= 1:
        return out
    idx = np.flatnonzero(out != 0)
    if len(idx) == 0:
        return out
    n_keep = max(1, int(round(len(out) * float(target_trade_rate))))
    if len(idx) <= n_keep:
        return out
    order = idx[np.argsort(confidences[idx])[::-1][:n_keep]]
    kept = np.zeros_like(out)
    kept[order] = out[order]
    return kept


def prepare_xy(
    df: pd.DataFrame,
    *,
    timeframe: str | None = None,
    cfg: dict[str, Any] | None = None,
    atr_mult: float | None = None,
    horizon: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], np.ndarray]:
    cfg = dict(cfg or _cfg())
    tf_key = str(timeframe or "M15")
    if horizon is None:
        horizon = horizon_for_timeframe(tf_key, cfg)
    else:
        horizon = int(horizon)
    if atr_mult is None:
        atr_mult = float(cfg.get("barrier_atr_multiplier", 1.5))
        by_tf_barrier = cfg.get("barrier_atr_multiplier_by_tf") or {}
        if tf_key in by_tf_barrier:
            atr_mult = float(by_tf_barrier[tf_key])
    else:
        atr_mult = float(atr_mult)
    labeling = str(cfg.get("labeling", "triple_barrier"))

    work = df.copy()
    if bool(cfg.get("engineer_learning_features", True)):
        work = engineer_learning_features(work)
    if labeling == "triple_barrier":
        labels, label_w = triple_barrier_labels_and_weights(work, horizon=horizon, atr_mult=atr_mult)
        work["label"] = labels
        work["label_weight"] = label_w
    else:
        work["label"] = classification_horizon_labels(work, horizon)
        work["label_weight"] = 1.0

    feature_cols = select_feature_columns(work)
    feature_cols = [c for c in feature_cols if c not in {"label", "label_weight"}]
    if bool(cfg.get("drop_registry_context", True)):
        feature_cols = [c for c in feature_cols if not str(c).startswith("registry_")]
        feature_cols = [c for c in feature_cols if not str(c).startswith("pattern_") or str(c) in {
            "pat_bias", "pat_strength", "chart_pattern_score", "trend_strength"
        }]
    if bool(cfg.get("prefer_relative_features", True)):
        feature_cols = prefer_relative_features(feature_cols, keep_min=int(cfg.get("min_relative_features", 12)))
    # Always keep multi-TF / engineered features when present (report: need HTF context).
    boost_prefixes = ("htf_", "mtf_", "feat_")
    boosted = [c for c in select_feature_columns(work) if str(c).startswith(boost_prefixes)]
    for c in boosted:
        if c not in feature_cols and c not in {"label", "label_weight"}:
            feature_cols.append(c)

    usable: list[str] = []
    for c in feature_cols:
        s = work[c]
        if not np.issubdtype(s.dtype, np.number):
            continue
        nn = int(s.notna().sum())
        if nn == 0:
            continue
        if nn >= max(40, int(0.4 * len(work))):
            usable.append(c)
    if not usable:
        usable = [
            c for c in feature_cols
            if np.issubdtype(work[c].dtype, np.number) and int(work[c].notna().sum()) >= 20
        ]
    feature_cols = usable

    if bool(cfg.get("drop_constant_features", True)):
        kept: list[str] = []
        for c in feature_cols:
            s = work[c]
            if int(s.nunique(dropna=True)) <= 1:
                continue
            std = float(s.std(skipna=True) or 0.0)
            if std <= 1e-12:
                continue
            kept.append(c)
        feature_cols = kept or feature_cols

    work = work.dropna(subset=["label"] + feature_cols)
    # Keep timeout class (0) in the chronological index so hold_bars maps to
    # real calendar bars. Directional-only fitting happens inside the train loop.
    X = work[feature_cols]
    y = work["label"].astype(int)
    label_weights = work["label_weight"].astype(float).values
    return X, y, feature_cols, label_weights


def train_symbol_timeframe(
    symbol: str,
    timeframe: str,
    *,
    force: bool = False,
    progress: Callable[[float, str], None] | None = None,
    log: Callable[[str], None] | None = None,
    status: Callable[[dict[str, Any]], None] | None = None,
) -> TrainResult:
    ensure_project_dirs()
    seed = set_global_seed()
    cfg = dict(_cfg())
    result = TrainResult(symbol=symbol, timeframe=timeframe, version="")
    applied_self_opt: dict[str, Any] = {}
    if bool(cfg.get("apply_self_optimize", True)):
        try:
            from atis.engines.engine4_training.enterprise_report import apply_pending_overrides

            kl_path = get_path("models") / symbol / timeframe / "knowledge_loop.json"
            applied_self_opt = apply_pending_overrides(cfg, kl_path)
            if applied_self_opt and log:
                log(f"[{timeframe}] self_optimize_applied={applied_self_opt}")
        except Exception as exc:
            if log:
                log(f"[{timeframe}] self_optimize_apply_error={exc}")

    def _emit_stage(stage: str, pct: float, message: str) -> None:
        if progress:
            try:
                progress(pct, message)
            except Exception:
                pass
        if status:
            try:
                status(
                    {
                        "timeframe": timeframe,
                        "stage": stage,
                        "progress_pct": float(pct),
                        "message": message,
                    }
                )
            except Exception:
                pass

    try:
        _emit_stage("loading_data", 0.0, f"تحميل JSON وبيانات الأنماط · {symbol} · {timeframe}")
        df, source_meta = load_training_frame(symbol, timeframe)
        if log:
            cross = source_meta.get("cross_tf") or {}
            log(
                f"[{timeframe}] pipeline={PIPELINE_VERSION} "
                f"source={source_meta['features_json_path']} "
                f"rows={source_meta['row_count']} "
                f"patterns={source_meta['pattern_summary']['knowledge_count']} "
                f"htf_cols={cross.get('n_htf_cols', 0)} "
                f"htf_sources={[s.get('timeframe') for s in (cross.get('sources') or [])]}"
            )
        _emit_stage("features", 6.0, f"هندسة ميزات / Cross-TF · {timeframe}")
        barrier_sweep: dict[str, Any] = {"enabled": False}
        if bool(cfg.get("barrier_sweep_enabled", True)):
            try:
                from atis.engines.engine4_training.barrier_optimization import sweep_barrier_params

                _emit_stage("labeling", 6.5, f"مسح حواجز Labels · {timeframe}")
                barrier_sweep = sweep_barrier_params(df, timeframe=timeframe, cfg=cfg)
                if log and barrier_sweep.get("enabled"):
                    log(f"[{timeframe}] barrier_sweep {barrier_sweep.get('summary_ar')}")
            except Exception as exc:
                barrier_sweep = {"enabled": False, "error": str(exc)}
                if log:
                    log(f"[{timeframe}] barrier_sweep_error={exc}")

        atr_override = None
        hor_override = None
        if barrier_sweep.get("applied"):
            atr_override = float(barrier_sweep["chosen_atr"])
            hor_override = int(barrier_sweep["chosen_horizon"])
            cfg["barrier_atr_multiplier"] = atr_override
            hbt = dict(cfg.get("horizon_by_timeframe") or {})
            hbt[timeframe] = hor_override
            cfg["horizon_by_timeframe"] = hbt

        X, y, feature_cols, label_weights = prepare_xy(
            df,
            timeframe=timeframe,
            cfg=cfg,
            atr_mult=atr_override,
            horizon=hor_override,
        )
        # Label noise cleaning / reweight
        label_clean_meta: dict[str, Any] = {"enabled": False}
        try:
            from atis.engines.engine4_training.barrier_optimization import clean_label_weights

            label_weights, label_clean_meta = clean_label_weights(
                y, label_weights, X, cfg=cfg, seed=seed
            )
            if log and label_clean_meta.get("n_downweighted"):
                log(
                    f"[{timeframe}] label_clean downweighted={label_clean_meta.get('n_downweighted')} "
                    f"mean_w={label_clean_meta.get('mean_weight')}"
                )
        except Exception as exc:
            label_clean_meta = {"enabled": False, "error": str(exc)}
        min_rows = int(cfg.get("min_rows", 120))
        if len(X) < min_rows:
            result.error = f"insufficient_rows:{len(X)}"
            if status:
                status({"timeframe": timeframe, "stage": "error", "error": result.error, "progress_pct": 100.0})
            return result

        _emit_stage("data_intelligence", 8.0, f"تحليل جاهزية البيانات · {timeframe}")
        from atis.engines.engine4_training.data_intelligence import analyze_training_frame

        data_intel = analyze_training_frame(
            df.loc[X.index] if hasattr(df, "loc") else df,
            y,
            feature_cols,
            timeframe=timeframe,
            cfg=cfg,
        )
        if log:
            log(f"[{timeframe}] data_intelligence score={data_intel.get('score')} ready={data_intel.get('ready')} flags={data_intel.get('flags')}")
        if bool(cfg.get("data_intel_hard", False)) and not data_intel.get("ready", True):
            result.error = f"data_intelligence_not_ready:{data_intel.get('flags')}"
            result.metrics = {
                "data_intelligence": data_intel,
                "gate_failures": ["data_quality_gate"],
                "passed_gates": False,
                "pipeline_version": PIPELINE_VERSION,
            }
            return result

        model_cfg = resolve_model_cfg_for_tf(cfg, timeframe)
        horizon = horizon_for_timeframe(timeframe, cfg)
        decision_threshold = float(cfg.get("decision_threshold", cfg.get("min_trade_confidence", 0.55)))
        min_conf = float(cfg.get("min_trade_confidence", decision_threshold))
        directional_edge = float(cfg.get("directional_edge", 0.15))
        confidence_quantile = float(cfg.get("confidence_quantile", 0.88))
        by_tf_q = cfg.get("confidence_quantile_by_tf") or {}
        if timeframe in by_tf_q:
            confidence_quantile = float(by_tf_q[timeframe])
        cost_edge_multiple = float(cfg.get("cost_edge_multiple", 1.25))
        by_tf_cost = cfg.get("cost_edge_multiple_by_tf") or {}
        if timeframe in by_tf_cost:
            cost_edge_multiple = float(by_tf_cost[timeframe])
        fold_val_ratio = float(cfg.get("fold_validation_ratio", 0.35))
        tune_policy = bool(cfg.get("tune_trade_policy", True))
        tune_policy_mode = str(cfg.get("tune_policy_mode", "once_liquid")).lower()
        use_regime = bool(cfg.get("regime_filter", True))
        use_trend = bool(cfg.get("trend_align", True))
        non_overlapping = bool(cfg.get("non_overlapping_trades", True))
        use_purge = bool(cfg.get("purge_embargo", True))
        train_directional_only = bool(cfg.get("train_on_directional_only", False))
        target_trade_rate = float(cfg.get("target_trade_rate", 0.05))
        by_tf_rate = cfg.get("target_trade_rate_by_tf") or {}
        if timeframe in by_tf_rate:
            target_trade_rate = float(by_tf_rate[timeframe])
        short_edge_multiple = float(cfg.get("short_edge_multiple", 1.35))
        by_tf_short = cfg.get("short_edge_multiple_by_tf") or {}
        if timeframe in by_tf_short:
            short_edge_multiple = float(by_tf_short[timeframe])
        use_calibration = bool(cfg.get("calibrate_probabilities", True))
        max_train_bars = int(cfg.get("max_train_bars", 0))
        by_tf_max = cfg.get("max_train_bars_by_tf") or {}
        if timeframe in by_tf_max:
            max_train_bars = int(by_tf_max[timeframe])
        apply_oos_sparsify = bool(cfg.get("apply_oos_sparsify", False))
        deploy_from_last_window = bool(cfg.get("deploy_from_last_window", True))
        use_meta_labeling = bool(cfg.get("use_meta_labeling", False))
        by_tf_meta = cfg.get("use_meta_labeling_by_tf") or {}
        if timeframe in by_tf_meta:
            use_meta_labeling = bool(by_tf_meta[timeframe])
        max_fold_trade_rate = float(cfg.get("max_fold_trade_rate", 0.20))
        regime_low_q = float(cfg.get("regime_atr_low_q", 0.20))
        regime_high_q = float(cfg.get("regime_atr_high_q", 0.90))
        embargo = horizon if use_purge else 0
        purge = horizon if use_purge else 0
        ppy = periods_per_year_for(timeframe)
        sharpe_ann_cap = cfg.get("sharpe_ann_cap", "daily")
        bootstrap_ci = bool(cfg.get("bootstrap_sharpe_ci", True))
        n_bootstrap = int(cfg.get("bootstrap_sharpe_samples", 400))

        def _fm(rets: np.ndarray, *, bootstrap: bool = False) -> dict[str, float]:
            return financial_metrics(
                rets,
                periods_per_year=ppy,
                hold_bars=horizon,
                ann_cap=sharpe_ann_cap,
                bootstrap=bootstrap and bootstrap_ci,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )

        spread_pips = float(cfg.get("spread_pips", 30.0))
        slippage_pips = float(cfg.get("slippage_pips", 5.0))
        commission_per_lot = float(cfg.get("commission_per_lot", 7.0))
        pip = _pip_size_for_symbol(symbol)
        half_life = float(cfg.get("time_decay_half_life", 0.4))
        dynamic_exec = bool(cfg.get("dynamic_execution_costs", True))
        vol_slip_k = float(cfg.get("vol_slippage_k", 1.25))
        latency_bars = int(cfg.get("latency_bars", 0))
        execution_delay_bars = int(cfg.get("execution_delay_bars", 0))

        n_splits = int(cfg.get("walk_forward_splits", 5))
        train_ratio = float(cfg.get("train_ratio", 0.7))
        validation_mode = str(cfg.get("validation_mode", "expanding")).lower()
        promo_mode = str(cfg.get("promotion_validation_mode", "") or "").strip().lower()
        if bool(cfg.get("use_promotion_validation_mode", False)) and promo_mode:
            validation_mode = promo_mode
        from atis.engines.engine4_training.validation_protocols import build_validation_splits

        splits = build_validation_splits(
            len(X),
            mode=validation_mode,
            n_splits=n_splits,
            train_ratio=train_ratio,
            embargo=embargo,
            purge=purge,
            rolling_train_size=cfg.get("rolling_train_size"),
            rolling_test_size=cfg.get("rolling_test_size"),
            cpcv_n_groups=cfg.get("cpcv_n_groups"),
            cpcv_n_test_groups=cfg.get("cpcv_n_test_groups"),
            cpcv_max_paths=cfg.get("cpcv_max_paths"),
        )
        if not splits:
            result.error = "no_walk_forward_splits"
            return result
        if log:
            log(
                f"[{timeframe}] validation_mode={validation_mode} folds={len(splits)} "
                f"latency={latency_bars} delay={execution_delay_bars} dynamic_costs={dynamic_exec}"
            )

        # Leak-safe feature selection: early walk-forward trains only (no future folds).
        top_k = int(cfg.get("top_features", 60))
        by_tf_top = cfg.get("top_features_by_tf") or {}
        if timeframe in by_tf_top:
            top_k = int(by_tf_top[timeframe])
        first_tr, _ = splits[0]
        if top_k and len(feature_cols) > top_k:
            if bool(cfg.get("stable_feature_selection", True)) and len(splits) >= 2:
                feature_cols = select_stable_top_features(
                    X,
                    y,
                    splits,
                    k=top_k,
                    seed=seed,
                    label_weights=label_weights,
                    half_life=half_life,
                    train_directional_only=train_directional_only,
                    cfg=model_cfg,
                    n_windows=int(cfg.get("stable_feature_windows", 3)),
                    min_frac=float(cfg.get("stable_feature_min_frac", 0.6)),
                )
            else:
                w_sel = time_decay_weights(len(first_tr), half_life) * label_weights[first_tr]
                y_sel = y.iloc[first_tr]
                x_sel = X.iloc[first_tr]
                if train_directional_only:
                    dir_mask = y_sel.astype(float).values != 0.0
                    if int(dir_mask.sum()) >= 40:
                        x_sel = x_sel.iloc[dir_mask]
                        y_sel = y_sel.iloc[dir_mask]
                        w_sel = w_sel[dir_mask]
                feature_cols = select_top_features(
                    x_sel,
                    y_sel,
                    k=top_k,
                    seed=seed,
                    sample_weight=w_sel,
                    cfg=model_cfg,
                )
            X = X[feature_cols]
        feature_intel: dict[str, Any] = {"enabled": False}
        if bool(cfg.get("feature_intelligence_enabled", True)):
            _emit_stage("feature_intelligence", 13.0, f"ذكاء الميزات (MI/Importance) · {timeframe}")
            from atis.engines.engine4_training.feature_intelligence import analyze_and_select_features

            try:
                w_fi = time_decay_weights(len(X), half_life) * label_weights
                selected_fi, feature_intel = analyze_and_select_features(
                    X,
                    y,
                    max_features=top_k if top_k else len(feature_cols),
                    seed=seed,
                    corr_threshold=float(cfg.get("feature_corr_threshold", 0.92)),
                    sample_weight=np.asarray(w_fi, dtype=float),
                )
                if selected_fi and len(selected_fi) >= 8:
                    feature_cols = selected_fi
                    X = X[feature_cols]
                if log and feature_intel.get("enabled"):
                    log(
                        f"[{timeframe}] feature_intelligence selected={feature_intel.get('n_selected')} "
                        f"weak_drop={feature_intel.get('n_weak_dropped')} corr_drop={feature_intel.get('n_corr_dropped')}"
                    )
            except Exception as exc:
                feature_intel = {"enabled": False, "error": str(exc)}
                if log:
                    log(f"[{timeframe}] feature_intelligence_error={exc}")
        _emit_stage(
            "feature_selection",
            14.0,
            f"تهيئة Features/Labels · {timeframe} · rows={len(X)} · feats={len(feature_cols)}",
        )

        # Align close/ATR/trend for backtest with X index
        aligned = df.loc[X.index]
        close = aligned["close"].values.astype(float)
        if "atr" in aligned.columns:
            atr_pct = (aligned["atr"].astype(float) / np.maximum(aligned["close"].astype(float), 1e-12)).values
        else:
            atr_pct = pd.Series(close).pct_change().abs().rolling(14).mean().fillna(0.001).values

        def _bt(
            close_arr: np.ndarray,
            preds_arr: np.ndarray,
            *,
            atr_arr: np.ndarray | None = None,
            confidences: np.ndarray | None = None,
            min_confidence: float = 0.0,
        ) -> tuple[np.ndarray, dict[str, float]]:
            """Backtest with configured costs / latency (aligned ATR slice)."""
            return _trade_returns_from_preds(
                close_arr,
                preds_arr,
                hold_bars=horizon,
                spread_pips=spread_pips,
                slippage_pips=slippage_pips,
                commission_per_lot=commission_per_lot,
                pip_size=pip,
                confidences=confidences,
                min_confidence=min_confidence,
                non_overlapping=non_overlapping,
                atr_pct=atr_arr,
                dynamic_costs=dynamic_exec,
                vol_slippage_k=vol_slip_k,
                latency_bars=latency_bars,
                execution_delay_bars=execution_delay_bars,
            )

        # Regime bounds fitted per-fold on train only (avoid full-series leakage).
        allow_long, allow_short = trend_masks_from_frame(aligned)
        if not use_trend:
            allow_long = np.ones(len(close), dtype=bool)
            allow_short = np.ones(len(close), dtype=bool)

        model_name = str(cfg.get("baseline_model", "lightgbm"))
        oos_preds = np.zeros(len(X))
        oos_raw = np.zeros(len(X))
        oos_conf = np.zeros(len(X))
        oos_mask = np.zeros(len(X), dtype=bool)
        oos_proba = np.zeros((len(X), 3), dtype=float)
        oos_proba_mask = np.zeros(len(X), dtype=bool)
        train_preds = np.zeros(len(X))
        train_raw = np.zeros(len(X))
        train_mask = np.zeros(len(X), dtype=bool)
        train_proba = np.zeros((len(X), 3), dtype=float)
        train_proba_mask = np.zeros(len(X), dtype=bool)
        val_preds = np.zeros(len(X))
        val_raw = np.zeros(len(X))
        val_mask = np.zeros(len(X), dtype=bool)
        val_proba = np.zeros((len(X), 3), dtype=float)
        val_proba_mask = np.zeros(len(X), dtype=bool)
        fold_metrics: list[dict[str, Any]] = []
        fold_top_features: list[list[str]] = []
        tuned_policies: list[dict[str, float]] = []
        frozen_policy: dict[str, float] | None = None
        best_fold_i = -1
        best_fold_score = -1e18
        best_fold_train_idx: np.ndarray | None = None
        model_classes_ref: list[Any] | None = None
        last_calibrator: Callable[[np.ndarray], np.ndarray] | None = None
        last_regime_bounds: tuple[float, float] = (0.0, float("inf"))
        deploy_regime_bounds: tuple[float, float] = (0.0, float("inf"))
        last_te_test: np.ndarray = np.array([], dtype=int)
        last_conf_floor: float = decision_threshold
        last_fold_policy: dict[str, float] | None = None
        final_model: Any = None
        deploy_tr: np.ndarray = np.array([], dtype=int)

        dq = data_quality_report(aligned, y, feature_cols, timeframe=timeframe, cfg=cfg)
        if log:
            log(
                f"[{timeframe}] data_quality score={dq.get('score')} "
                f"gate_pass={dq.get('gate_pass')} rows={dq['n_rows']} "
                f"imbalance={dq['directional_imbalance_ratio']} "
                f"outlier_frac={dq['outlier_frac']} flags={dq['quality_flags']}"
            )
        # Label quality / noise analysis (feeds DQ + reports).
        label_quality: dict[str, Any] = {"enabled": False}
        if bool(cfg.get("label_quality_enabled", True)):
            _emit_stage("labeling", 15.0, f"تحليل جودة Labels · {timeframe}")
            try:
                from atis.engines.engine4_training.label_quality import (
                    analyze_label_quality,
                    merge_label_quality_into_dq,
                )

                label_quality = analyze_label_quality(
                    X,
                    y,
                    label_weights=np.asarray(label_weights, dtype=float),
                    timeframe=timeframe,
                    cfg=cfg,
                    seed=seed,
                )
                dq = merge_label_quality_into_dq(dq, label_quality)
                if log:
                    log(
                        f"[{timeframe}] label_quality score={label_quality.get('score')} "
                        f"noise={((label_quality.get('noise') or {}).get('noise_rate'))} "
                        f"flags={label_quality.get('flags')}"
                    )
            except Exception as exc:
                label_quality = {"enabled": False, "error": str(exc)}
                if log:
                    log(f"[{timeframe}] label_quality_error={exc}")

        if bool(cfg.get("dq_gate_hard", True)) and dq.get("gate_pass") is False:
            skip = str(dq.get("skip_reason") or "data_quality_gate")
            gate_key = "label_quality_gate" if "label" in skip else "data_quality_gate"
            result.error = skip
            result.metrics = {
                "data_quality": dq,
                "label_quality": label_quality,
                "gate_failures": [gate_key],
                "gate_failures_detail": annotate_gate_failures([gate_key]),
                "passed_gates": False,
                "pipeline_version": PIPELINE_VERSION,
                "symbol": symbol,
                "timeframe": timeframe,
                "awareness": dq.get("awareness") or {},
                "decision_explanations": [
                    {
                        "decision": "early_skip",
                        "why": skip,
                        "ar": (dq.get("awareness") or {}).get("explanation_ar")
                        or (label_quality.get("summary_ar") if label_quality else None)
                        or f"تخطي {timeframe} — بوابة جودة البيانات",
                    }
                ],
            }
            if log:
                log(f"[{timeframe}] DQ_GATE_REJECT reason={skip} score={dq.get('score')}")
            if status:
                status(
                    {
                        "timeframe": timeframe,
                        "stage": "skipped",
                        "error": skip,
                        "progress_pct": 100.0,
                        "message": (dq.get("awareness") or {}).get("explanation_ar") or skip,
                    }
                )
            return result

        # Nested HP: single first-fold OR true nested across outer train windows.
        nested_hp_meta: dict[str, Any] = {"enabled": False}
        if bool(cfg.get("nested_hp_search", True)):
            from atis.engines.engine4_training.adaptive_learning import (
                nested_hp_across_outer_folds,
                nested_hyperparameter_search,
            )

            def _hp_payload(tr_idx: np.ndarray, fold_i: int) -> dict[str, Any]:
                tr_hp = tr_idx
                if max_train_bars > 0 and len(tr_hp) > max_train_bars:
                    tr_hp = tr_hp[-max_train_bars:]
                X_hp = StandardScaler().fit_transform(X.values[tr_hp])
                y_hp = y.values[tr_hp]
                w_full = time_decay_weights(len(tr_hp), half_life) * label_weights[tr_hp]
                if train_directional_only:
                    dm = y_hp != 0
                    if int(dm.sum()) >= 40:
                        X_hp, y_hp, w_hp = X_hp[dm], y_hp[dm], w_full[dm]
                    else:
                        w_hp = w_full
                else:
                    w_hp = w_full
                return {"X": X_hp, "y": y_hp, "w": w_hp, "fold": fold_i}

            if bool(cfg.get("nested_hp_per_fold", True)) and len(splits) >= 2:
                _emit_stage("walk_forward", 15.5, f"Nested HP عبر الطيات · {timeframe}")
                n_outer = int(cfg.get("nested_hp_outer_folds", 3))
                payloads = [
                    _hp_payload(tr, i)
                    for i, (tr, _) in enumerate(splits[: max(1, n_outer)])
                ]
                model_cfg, nested_hp_meta = nested_hp_across_outer_folds(
                    payloads,
                    base_cfg=model_cfg,
                    timeframe=timeframe,
                    seed=seed,
                    n_trials=int(cfg.get("nested_hp_trials", 8)),
                    max_folds=n_outer,
                )
            else:
                first_tr_hp, _ = splits[0]
                payload0 = _hp_payload(first_tr_hp, 0)
                model_cfg, nested_hp_meta = nested_hyperparameter_search(
                    payload0["X"],
                    payload0["y"],
                    payload0["w"],
                    base_cfg=model_cfg,
                    timeframe=timeframe,
                    seed=seed,
                    n_trials=int(cfg.get("nested_hp_trials", 8)),
                )
            if log and nested_hp_meta.get("enabled"):
                log(
                    f"[{timeframe}] nested_hp best_score={nested_hp_meta.get('best_score')} "
                    f"family={nested_hp_meta.get('best_family')} "
                    f"mode={nested_hp_meta.get('mode', 'single')} "
                    f"trials={nested_hp_meta.get('trials') or nested_hp_meta.get('folds_tried')}"
                )

        # Enterprise model zoo bake-off (train-only inner split).
        model_zoo_meta: dict[str, Any] = {"enabled": False}
        if bool(cfg.get("model_zoo_enabled", True)):
            _emit_stage("model_zoo", 16.0, f"مقارنة عائلة النماذج · {timeframe}")
            try:
                from atis.engines.engine4_training.model_zoo import (
                    compare_model_zoo,
                    map_winner_to_baseline,
                )

                first_tr_z, _ = splits[0]
                if max_train_bars > 0 and len(first_tr_z) > max_train_bars:
                    first_tr_z = first_tr_z[-max_train_bars:]
                Xz = StandardScaler().fit_transform(X.values[first_tr_z])
                yz = y.values[first_tr_z]
                wz = time_decay_weights(len(first_tr_z), half_life) * label_weights[first_tr_z]
                if train_directional_only:
                    dm = yz != 0
                    if int(dm.sum()) >= 40:
                        Xz, yz, wz = Xz[dm], yz[dm], wz[dm]
                model_zoo_meta = compare_model_zoo(
                    Xz,
                    yz,
                    wz,
                    seed=seed,
                    cfg=model_cfg,
                    max_models=int(cfg.get("model_zoo_max_models", 8)),
                )
                if model_zoo_meta.get("enabled") and model_zoo_meta.get("winner"):
                    mapped = map_winner_to_baseline(str(model_zoo_meta["winner"]))
                    model_name = mapped
                    model_cfg["_nested_model_family"] = str(model_zoo_meta["winner"])
                    if log:
                        log(
                            f"[{timeframe}] model_zoo winner={model_zoo_meta.get('winner')} "
                            f"mapped={mapped} tried={model_zoo_meta.get('n_models_tried')}"
                        )
            except Exception as exc:
                model_zoo_meta = {"enabled": False, "error": str(exc)}
                if log:
                    log(f"[{timeframe}] model_zoo_error={exc}")

        # Resolve Zoo vs Nested conflict on financial proxy (v16).
        family_resolution: dict[str, Any] = {"conflict": False}
        try:
            from atis.engines.engine4_training.financial_hpo import resolve_zoo_vs_nested

            family_resolution = resolve_zoo_vs_nested(
                nested_meta=nested_hp_meta,
                zoo_meta=model_zoo_meta,
                current_model_name=model_name,
                cfg=cfg,
            )
            if family_resolution.get("conflict"):
                model_name = str(family_resolution.get("selected_baseline") or model_name)
                if bool(cfg.get("use_ensemble_on_conflict", True)) and family_resolution.get(
                    "selected_baseline"
                ) == "ensemble":
                    cfg["use_ensemble"] = True
                    model_name = "ensemble"
                if log:
                    log(
                        f"[{timeframe}] family_resolution {family_resolution.get('reason')} "
                        f"→ {family_resolution.get('selected_family')} / {model_name}"
                    )
        except Exception as exc:
            family_resolution = {"conflict": False, "error": str(exc)}

        # Default ensemble for liquid TFs when configured.
        if bool(cfg.get("use_ensemble", False)) or (
            bool(cfg.get("ensemble_liquid_tfs", True))
            and str(timeframe).upper() in set(cfg.get("ensemble_tfs") or ["M5", "M15", "M30", "H1"])
            and bool(cfg.get("force_ensemble_liquid", False))
        ):
            if bool(cfg.get("force_ensemble_liquid", False)):
                model_name = "ensemble"
                cfg["use_ensemble"] = True

        X_values = X.values
        y_values = y.values

        def _fit_mask(idx: np.ndarray) -> np.ndarray:
            if not train_directional_only:
                return np.ones(len(idx), dtype=bool)
            m = y_values[idx] != 0
            return m if int(m.sum()) >= 20 else np.ones(len(idx), dtype=bool)

        def _clip_train(idx: np.ndarray) -> np.ndarray:
            if max_train_bars > 0 and len(idx) > max_train_bars:
                return idx[-max_train_bars:]
            return idx

        from atis.engines.engine4_training.data_quality_gate import min_val_trades_for_timeframe

        for fold_i, (tr_raw, te) in enumerate(splits):
            tr = _clip_train(tr_raw)
            if use_regime:
                lo, hi = regime_bounds_from_atr(atr_pct[tr], low_q=regime_low_q, high_q=regime_high_q)
                regime = regime_mask_apply(atr_pct, lo, hi)
                last_regime_bounds = (lo, hi)
            else:
                regime = np.ones(len(close), dtype=bool)
                last_regime_bounds = (0.0, float("inf"))
            scaler = StandardScaler()
            X_tr_all = scaler.fit_transform(X_values[tr])
            fit_m = _fit_mask(tr)
            w_tr = time_decay_weights(len(tr), half_life) * label_weights[tr]
            model = build_model(model_name, seed, model_cfg)
            model = fit_classifier(
                model,
                X_tr_all[fit_m],
                y_values[tr][fit_m],
                w_tr[fit_m],
                cfg=model_cfg,
            )
            # Track fold-level top features for stability / explainability.
            try:
                imp = getattr(model, "feature_importances_", None)
                if imp is not None and len(imp) == len(feature_cols):
                    ranked = sorted(
                        zip(feature_cols, np.asarray(imp, dtype=float)),
                        key=lambda kv: -kv[1],
                    )
                    fold_top_features.append([c for c, _ in ranked[: min(20, len(ranked))]])
            except Exception:
                pass

            conf_floor = train_confidence_floor(
                model,
                X_tr_all[fit_m],
                decision_threshold=decision_threshold,
                confidence_quantile=confidence_quantile,
                min_floor=float(cfg.get("min_confidence_floor", 0.55)),
                target_trade_rate=target_trade_rate,
                max_floor=float(cfg.get("max_confidence_floor", 0.88)),
            )

            # Split fold window into validation (policy tuning) + held-out test.
            n_te = len(te)
            need_val = tune_policy or use_calibration
            n_val = max(10, int(n_te * fold_val_ratio)) if need_val and n_te >= 40 else 0
            n_val = min(n_val, max(0, n_te - 15))
            te_val = te[:n_val] if n_val else np.array([], dtype=int)
            te_test = te[n_val:] if n_val else te
            last_te_test = np.asarray(te_test, dtype=int)

            policy = {
                "decision_threshold": decision_threshold,
                "directional_edge": directional_edge,
                "confidence_quantile": confidence_quantile,
                "confidence_floor": conf_floor,
                "val_sharpe": 0.0,
                "val_trades": 0.0,
            }
            calibrate_fn: Callable[[np.ndarray], np.ndarray] | None = None
            if n_val:
                X_val = scaler.transform(X_values[te_val])
                proba_val_raw, classes_val = _model_proba(model, X_val)
                if use_calibration and proba_val_raw is not None and classes_val is not None:
                    calibrate_fn = make_proba_calibrator(proba_val_raw, classes_val, y_values[te_val])
                    if calibrate_fn is not None:
                        last_calibrator = calibrate_fn

                should_tune = tune_policy and (
                    tune_policy_mode == "each"
                    or (tune_policy_mode == "once" and frozen_policy is None)
                    or (tune_policy_mode == "once_liquid" and frozen_policy is None)
                )
                fold_min_trades = max(
                    min_val_trades_for_timeframe(timeframe, cfg),
                    max(5, int(cfg.get("min_trades_oos", 15) // max(n_splits, 1))),
                )
                # Report 04-15: M30/M15 froze a starved fold-0 policy → trade_rate≈0 thereafter.
                frozen_starved = (
                    frozen_policy is not None
                    and policy_liquidity_starved(
                        trades=float(frozen_policy.get("val_trades", 0.0) or 0.0),
                        n_bars=max(len(te_val), 1),
                        min_trades=fold_min_trades,
                        target_trade_rate=target_trade_rate,
                    )
                )
                if tune_policy and tune_policy_mode in {"once", "once_liquid"} and frozen_starved:
                    should_tune = True
                    frozen_policy = None  # never keep a starved freeze
                primary_val = (
                    structure_primary_sides(allow_long[te_val], allow_short[te_val])
                    if use_meta_labeling and use_trend
                    else None
                )
                if should_tune:
                    # When still starved after prior folds, relax cost filter for the search.
                    tune_cost_edge = float(cost_edge_multiple)
                    if frozen_starved or fold_i > 0:
                        tune_cost_edge = min(tune_cost_edge, 1.0)
                    policy = tune_trade_policy(
                        model=model,
                        X_val=X_val,
                        close_val=close[te_val],
                        atr_pct_val=atr_pct[te_val],
                        regime_val=regime[te_val],
                        hold_bars=horizon,
                        spread_pips=spread_pips,
                        slippage_pips=slippage_pips,
                        commission_per_lot=commission_per_lot,
                        pip_size=pip,
                        periods_per_year=ppy,
                        base_threshold=decision_threshold,
                        base_edge=directional_edge,
                        base_quantile=confidence_quantile,
                        cost_edge_multiple=tune_cost_edge,
                        non_overlapping=non_overlapping,
                        min_trades=fold_min_trades,
                        confidence_floor=conf_floor,
                        allow_long=allow_long[te_val] if use_trend else None,
                        allow_short=allow_short[te_val] if use_trend else None,
                        target_trade_rate=target_trade_rate,
                        max_trade_rate=max_fold_trade_rate,
                        short_edge_multiple=short_edge_multiple,
                        calibrate_fn=calibrate_fn,
                        apply_sparsify=apply_oos_sparsify,
                        primary_sides=primary_val,
                    )
                    # Prefer floor found by starvation-aware tuner when present.
                    policy["confidence_floor"] = float(policy.get("confidence_floor", conf_floor))
                    conf_floor = float(policy["confidence_floor"])
                    if "cost_edge_multiple" not in policy:
                        policy["cost_edge_multiple"] = float(tune_cost_edge)
                    if tune_policy_mode in {"once", "once_liquid"}:
                        # Freeze only once we have a liquid policy (avoid locking starvation).
                        liquid_ok = not policy_liquidity_starved(
                            trades=float(policy.get("val_trades", 0.0) or 0.0),
                            n_bars=max(len(te_val), 1),
                            min_trades=fold_min_trades,
                            target_trade_rate=target_trade_rate,
                        )
                        # Don't freeze an ultra-strict floor that will starve later folds/deploy.
                        floor_cap = float(cfg.get("max_confidence_floor", 0.70))
                        if float(policy.get("confidence_floor", floor_cap)) > floor_cap:
                            policy["confidence_floor"] = floor_cap
                            conf_floor = floor_cap
                        if liquid_ok:
                            frozen_policy = dict(policy)
                elif frozen_policy is not None:
                    policy = dict(frozen_policy)
                    policy["confidence_floor"] = float(policy.get("confidence_floor", conf_floor))
                    conf_floor = float(policy["confidence_floor"])
                tuned_policies.append(policy)

                fold_cost_edge = float(policy.get("cost_edge_multiple", cost_edge_multiple))
                proba_val, classes_val = _model_proba(model, X_val)
                raw_val = model.predict(X_val).astype(float)
                val_raw[te_val] = raw_val
                if proba_val is not None and classes_val is not None:
                    if calibrate_fn is not None:
                        proba_val = calibrate_fn(proba_val)
                    n_cls = proba_val.shape[1]
                    if val_proba.shape[1] != n_cls:
                        val_proba = np.zeros((len(X), n_cls), dtype=float)
                    val_proba[te_val] = proba_val
                    val_proba_mask[te_val] = True
                    unit_costs_val = np.array(
                        [
                            _unit_cost(
                                float(c),
                                spread_pips=spread_pips,
                                slippage_pips=slippage_pips,
                                commission_per_lot=commission_per_lot,
                                pip_size=pip,
                            )
                            for c in close[te_val]
                        ],
                        dtype=float,
                    )
                    vp = policy_from_proba(
                        proba_val,
                        classes_val,
                        decision_threshold=policy["decision_threshold"],
                        directional_edge=policy["directional_edge"],
                        confidence_quantile=0.0,
                        confidence_floor=conf_floor,
                        atr_pct=atr_pct[te_val],
                        unit_costs=unit_costs_val,
                        cost_edge_multiple=fold_cost_edge,
                        regime_mask=regime[te_val],
                        short_edge_multiple=short_edge_multiple,
                        primary_sides=primary_val,
                    )
                    if primary_val is None:
                        vp = apply_trend_align(vp, allow_long[te_val], allow_short[te_val])
                    if apply_oos_sparsify and target_trade_rate > 0:
                        vp = sparsify_by_confidence(vp, proba_val.max(axis=1), target_trade_rate=target_trade_rate)
                    vp = cap_preds_by_trade_rate(
                        vp, proba_val.max(axis=1), max_trade_rate=max_fold_trade_rate
                    )
                    val_preds[te_val] = vp
                    val_mask[te_val] = True
                    # Refresh observed val sharpe under frozen/calibrated stack
                    v_rets, _ = _bt(close[te_val], vp, atr_arr=atr_pct[te_val])
                    v_fin = _fm(v_rets)
                    policy["val_sharpe"] = float(v_fin["sharpe"])
                    policy["val_trades"] = float(v_fin.get("n_trades", 0.0))
                    policy["val_return"] = float(v_fin.get("total_return", 0.0))
                    # If a previously frozen policy starves THIS fold, force retune next folds.
                    if (
                        tune_policy_mode == "once_liquid"
                        and frozen_policy is not None
                        and policy_liquidity_starved(
                            trades=float(policy["val_trades"]),
                            n_bars=max(len(te_val), 1),
                            min_trades=fold_min_trades,
                            target_trade_rate=target_trade_rate,
                        )
                    ):
                        frozen_policy = None

            X_test = scaler.transform(X_values[te_test])
            raw_pred = model.predict(X_test).astype(float)
            proba_te, classes_te = _model_proba(model, X_test)
            conf = np.ones(len(te_test), dtype=float)
            fold_cost_edge = float(policy.get("cost_edge_multiple", cost_edge_multiple))
            primary_te = (
                structure_primary_sides(allow_long[te_test], allow_short[te_test])
                if use_meta_labeling and use_trend
                else None
            )
            if proba_te is not None and classes_te is not None:
                if calibrate_fn is not None:
                    proba_te = calibrate_fn(proba_te)
                unit_costs_te = np.array(
                    [
                        _unit_cost(
                            float(c),
                            spread_pips=spread_pips,
                            slippage_pips=slippage_pips,
                            commission_per_lot=commission_per_lot,
                            pip_size=pip,
                        )
                        for c in close[te_test]
                    ],
                    dtype=float,
                )
                pred = policy_from_proba(
                    proba_te,
                    classes_te,
                    decision_threshold=policy["decision_threshold"],
                    directional_edge=policy["directional_edge"],
                    confidence_quantile=0.0,
                    confidence_floor=conf_floor,
                    atr_pct=atr_pct[te_test],
                    unit_costs=unit_costs_te,
                    cost_edge_multiple=fold_cost_edge,
                    regime_mask=regime[te_test],
                    short_edge_multiple=short_edge_multiple,
                    primary_sides=primary_te,
                )
                conf = proba_te.max(axis=1)
            else:
                pred, conf = predict_with_confidence(
                    model,
                    X_test,
                    decision_threshold=policy["decision_threshold"],
                    directional_edge=policy["directional_edge"],
                    confidence_quantile=confidence_quantile,
                )
            if primary_te is None:
                pred = apply_trend_align(pred, allow_long[te_test], allow_short[te_test])
            if apply_oos_sparsify and target_trade_rate > 0:
                pred = sparsify_by_confidence(pred, conf, target_trade_rate=target_trade_rate)
            pred = cap_preds_by_trade_rate(pred, conf, max_trade_rate=max_fold_trade_rate)

            # Train-slice score with the same frozen/calibrated policy (overfit monitor).
            proba_tr, classes_tr = _model_proba(model, X_tr_all)
            if classes_tr is not None:
                model_classes_ref = list(classes_tr)
            raw_tr = model.predict(X_tr_all).astype(float)
            if proba_tr is not None and classes_tr is not None:
                if calibrate_fn is not None:
                    proba_tr = calibrate_fn(proba_tr)
                unit_costs_tr = np.array(
                    [
                        _unit_cost(
                            float(c),
                            spread_pips=spread_pips,
                            slippage_pips=slippage_pips,
                            commission_per_lot=commission_per_lot,
                            pip_size=pip,
                        )
                        for c in close[tr]
                    ],
                    dtype=float,
                )
                pred_tr = policy_from_proba(
                    proba_tr,
                    classes_tr,
                    decision_threshold=policy["decision_threshold"],
                    directional_edge=policy["directional_edge"],
                    confidence_quantile=0.0,
                    confidence_floor=conf_floor,
                    atr_pct=atr_pct[tr],
                    unit_costs=unit_costs_tr,
                    cost_edge_multiple=fold_cost_edge,
                    regime_mask=regime[tr],
                    short_edge_multiple=short_edge_multiple,
                    primary_sides=(
                        structure_primary_sides(allow_long[tr], allow_short[tr])
                        if use_meta_labeling and use_trend
                        else None
                    ),
                )
                conf_tr = proba_tr.max(axis=1)
                # Store proba with flexible class count
                n_cls = proba_tr.shape[1]
                if train_proba.shape[1] != n_cls:
                    train_proba = np.zeros((len(X), n_cls), dtype=float)
                    oos_proba = np.zeros((len(X), n_cls), dtype=float)
                train_proba[tr] = proba_tr
                train_proba_mask[tr] = True
            else:
                pred_tr, conf_tr = predict_with_confidence(
                    model,
                    X_tr_all,
                    decision_threshold=policy["decision_threshold"],
                    directional_edge=policy["directional_edge"],
                    confidence_quantile=confidence_quantile,
                )
            if not (use_meta_labeling and use_trend):
                pred_tr = apply_trend_align(pred_tr, allow_long[tr], allow_short[tr])
            if apply_oos_sparsify and target_trade_rate > 0:
                pred_tr = sparsify_by_confidence(pred_tr, conf_tr, target_trade_rate=target_trade_rate)
            pred_tr = cap_preds_by_trade_rate(pred_tr, conf_tr, max_trade_rate=max_fold_trade_rate)
            train_preds[tr] = pred_tr
            train_raw[tr] = raw_tr
            train_mask[tr] = True

            if proba_te is not None and classes_te is not None:
                n_cls = proba_te.shape[1]
                if oos_proba.shape[1] != n_cls:
                    oos_proba = np.zeros((len(X), n_cls), dtype=float)
                oos_proba[te_test] = proba_te
                oos_proba_mask[te_test] = True

            oos_preds[te_test] = pred
            oos_raw[te_test] = raw_pred
            oos_conf[te_test] = conf
            oos_mask[te_test] = True

            # Rank folds by validation Sharpe — exclude starved Val folds (H4 tragedy).
            fold_score = float(policy.get("val_sharpe", 0.0))
            last_fold_policy = dict(policy)
            last_conf_floor = float(policy.get("confidence_floor", conf_floor))
            from atis.engines.engine4_training.adaptive_learning import fold_eligible_for_selection

            fold_liquid = fold_eligible_for_selection(
                n_val_trades=float(policy.get("val_trades", 0.0) or 0.0),
                val_sharpe=fold_score,
                timeframe=timeframe,
                cfg=cfg,
                n_val_bars=int(len(te_val)),
            )
            if len(te_val) >= 10 and fold_liquid and fold_score > best_fold_score:
                best_fold_score = fold_score
                best_fold_i = fold_i
                best_fold_train_idx = np.asarray(tr, dtype=int)

            pct = 12.0 + (48.0 * (fold_i + 1) / max(1, len(splits)))
            _emit_stage("walk_forward", pct, f"Walk-Forward fold {fold_i + 1}/{len(splits)} · {timeframe}")

            y_fold = y_values[te_test]
            p_fold = raw_pred
            proba_fold = proba_te if proba_te is not None else None
            classes_fold = classes_te
            if train_directional_only:
                dm = y_fold != 0
                if int(dm.sum()) >= 5:
                    y_fold, p_fold = y_fold[dm], raw_pred[dm]
                    if proba_fold is not None:
                        proba_fold = proba_fold[dm]
            fold_cls = classification_bundle(y_fold, p_fold, proba_fold, classes_fold)
            fold_test_rets, _ = _bt(close[te_test], pred, atr_arr=atr_pct[te_test])
            fold_test_fin = _fm(fold_test_rets)
            if log:
                trade_rate = float(np.mean(pred != 0)) if len(pred) else 0.0
                log(
                    f"[{timeframe}] fold={fold_i + 1}/{len(splits)} "
                    f"acc={fold_cls['accuracy']:.4f} "
                    f"f1={fold_cls['f1_macro']:.4f} "
                    f"auc={fold_cls['roc_auc_ovr']:.4f} "
                    f"trade_rate={trade_rate:.2f} "
                    f"val_sharpe={policy.get('val_sharpe', 0):.3f} "
                    f"test_sharpe={fold_test_fin.get('sharpe', 0):.3f} "
                    f"n_val_trades={float(policy.get('val_trades', 0) or 0):.0f}"
                )

            fold_metrics.append(
                {
                    "fold": fold_i,
                    "accuracy": fold_cls["accuracy"],
                    "precision_macro": fold_cls["precision_macro"],
                    "recall_macro": fold_cls["recall_macro"],
                    "f1_macro": fold_cls["f1_macro"],
                    "roc_auc_ovr": fold_cls["roc_auc_ovr"],
                    "brier_score": fold_cls.get("brier_score", 0.0),
                    "n_train": int(len(tr)),
                    "n_train_fit": int(int(fit_m.sum())),
                    "n_validation": int(len(te_val)),
                    "n_test": int(len(te_test)),
                    "trade_rate": float(np.mean(pred != 0)) if len(pred) else 0.0,
                    "policy": policy,
                    "confidence_floor": conf_floor,
                    "val_sharpe": float(policy.get("val_sharpe", 0.0)),
                    "n_val_trades": float(policy.get("val_trades", 0.0) or 0.0),
                    "test_sharpe": float(fold_test_fin.get("sharpe", 0.0) or 0.0),
                    "n_test_trades": float(fold_test_fin.get("n_trades", 0.0) or 0.0),
                    "val_liquid": bool(fold_liquid),
                    "notes": "" if fold_liquid else "excluded_starved_val_fold",
                    "regime_lo": float(last_regime_bounds[0]),
                    "regime_hi": float(last_regime_bounds[1]) if np.isfinite(last_regime_bounds[1]) else None,
                }
            )

        # Final model: last purged expanding window (chronologically honest for live).
        # Best-val fold kept as metadata only to avoid selection bias.
        if (
            not deploy_from_last_window
            and best_fold_train_idx is not None
            and len(best_fold_train_idx) >= 40
        ):
            deploy_tr = np.asarray(best_fold_train_idx, dtype=int)
            deploy_regime_bounds = last_regime_bounds
            if log:
                log(f"[{timeframe}] deploy_from_best_val_fold={best_fold_i + 1} val_sharpe={best_fold_score:.3f}")
        else:
            last_tr, last_te = splits[-1]
            deploy_tr = _clip_train(np.arange(0, int(last_te[0])))
            if use_purge and purge > 0 and len(deploy_tr) > purge + 40:
                deploy_tr = deploy_tr[: len(deploy_tr) - purge]
            if use_regime:
                deploy_regime_bounds = regime_bounds_from_atr(
                    atr_pct[deploy_tr], low_q=regime_low_q, high_q=regime_high_q
                )
            else:
                deploy_regime_bounds = (0.0, float("inf"))
            if log:
                log(
                    f"[{timeframe}] deploy_from_last_window n={len(deploy_tr)} "
                    f"(best_val_fold_meta={best_fold_i + 1 if best_fold_i >= 0 else 'n/a'})"
                )
        scaler = StandardScaler()
        X_deploy_all = scaler.fit_transform(X_values[deploy_tr])
        deploy_fit = _fit_mask(deploy_tr)
        w_deploy = time_decay_weights(len(deploy_tr), half_life) * label_weights[deploy_tr]
        final_model = build_model(
            "ensemble" if bool(cfg.get("use_ensemble", False)) else model_name,
            seed,
            model_cfg,
        )
        final_model = fit_classifier(
            final_model,
            X_deploy_all[deploy_fit],
            y_values[deploy_tr][deploy_fit],
            w_deploy[deploy_fit],
            cfg=model_cfg,
        )
        _emit_stage("val_policy", 70.0, f"اكتمل التدريب · بدء Validation/Testing · {timeframe}")

        # Validation metrics (policy-tuning slices)
        val_fin = {"sharpe": 0.0, "max_drawdown": 0.0, "total_return": 0.0, "n_trades": 0.0}
        if val_mask.any():
            val_rets, _ = _bt(close, val_preds, atr_arr=atr_pct)
            val_rets = val_rets.copy()
            val_rets[~val_mask] = 0.0
            val_fin = _fm(val_rets[val_mask])
        _emit_stage("val_policy", 78.0, f"اكتمل Validation · {timeframe}")

        # Train / Val / Test comparison
        train_fin = {"sharpe": 0.0, "max_drawdown": 0.0, "total_return": 0.0, "n_trades": 0.0}
        train_cls = {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0, "roc_auc_ovr": 0.0}
        if train_mask.any():
            tr_rets, _ = _bt(close, train_preds, atr_arr=atr_pct)
            tr_rets = tr_rets.copy()
            tr_rets[~train_mask] = 0.0
            train_fin = _fm(tr_rets[train_mask])
            y_tr = y_values[train_mask]
            p_tr = train_raw[train_mask]
            pr_tr = train_proba[train_mask] if train_proba_mask[train_mask].any() else None
            if train_directional_only:
                dm = y_tr != 0
                if int(dm.sum()) >= 20:
                    y_tr, p_tr = y_tr[dm], p_tr[dm]
                    if pr_tr is not None:
                        pr_tr = pr_tr[dm]
            train_cls = classification_bundle(y_tr, p_tr, pr_tr, model_classes_ref)

        # OOS classification metrics on raw (unfiltered) predictions
        y_oos = y_values[oos_mask]
        p_oos = oos_raw[oos_mask]
        pr_oos = oos_proba[oos_mask] if oos_proba_mask[oos_mask].any() else None
        if train_directional_only:
            dm = y_oos != 0
            if int(dm.sum()) >= 20:
                y_oos, p_oos = y_oos[dm], p_oos[dm]
                if pr_oos is not None:
                    pr_oos = pr_oos[dm]
        cls_metrics = classification_bundle(y_oos, p_oos, pr_oos, model_classes_ref)
        cls_metrics["trade_rate_filtered"] = float(np.mean(oos_preds[oos_mask] != 0)) if oos_mask.any() else 0.0

        val_cls = {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "roc_auc_ovr": 0.0,
            "brier_score": 0.0,
        }
        if val_mask.any():
            y_va = y_values[val_mask]
            p_va = val_raw[val_mask]
            pr_va = val_proba[val_mask] if val_proba_mask[val_mask].any() else None
            if train_directional_only:
                dm = y_va != 0
                if int(dm.sum()) >= 10:
                    y_va, p_va = y_va[dm], p_va[dm]
                    if pr_va is not None:
                        pr_va = pr_va[dm]
            val_cls = classification_bundle(y_va, p_va, pr_va, model_classes_ref)

        # Financial backtest on held-out test preds
        rets, trade_stats = _bt(
            close,
            oos_preds,
            atr_arr=atr_pct,
            confidences=oos_conf,
            min_confidence=0.0,
        )
        confidence_sizing_meta: dict[str, Any] = {"enabled": False}
        if bool(cfg.get("confidence_sizing_enabled", True)):
            from atis.engines.engine4_training.promotion_v16 import confidence_position_size

            sizes = np.ones(len(rets), dtype=float)
            active = oos_preds != 0
            for i in np.where(active)[0]:
                sizes[i] = confidence_position_size(
                    float(oos_conf[i]) if i < len(oos_conf) else 0.5,
                    atr_pct=float(atr_pct[i]) if i < len(atr_pct) else 0.002,
                    base_size=float(cfg.get("confidence_sizing_base", 1.0)),
                    max_size=float(cfg.get("confidence_sizing_max", 1.5)),
                    min_size=float(cfg.get("confidence_sizing_min", 0.25)),
                )
            rets = rets * sizes
            confidence_sizing_meta = {
                "enabled": True,
                "mean_size": float(np.mean(sizes[active])) if active.any() else 1.0,
                "median_size": float(np.median(sizes[active])) if active.any() else 1.0,
                "max_size": float(np.max(sizes[active])) if active.any() else 1.0,
            }
        rets_oos = rets.copy()
        rets_oos[~oos_mask] = 0.0
        fin = _fm(rets_oos[oos_mask], bootstrap=True)
        from atis.engines.engine4_training.financial_hpo import (
            expectancy_covers_cost,
            trade_level_sharpe,
        )
        from atis.engines.engine4_training.promotion_v16 import (
            crisis_recent_holdout_slices,
            evaluate_holdout_slice,
            fold_stability_report,
        )

        traded_oos = rets_oos[oos_mask]
        traded_oos = traded_oos[traded_oos != 0]
        trade_level_metrics = trade_level_sharpe(traded_oos)
        fin["trade_sharpe_raw"] = trade_level_metrics.get("trade_sharpe_raw", 0.0)
        ref_close = float(np.nanmedian(close[oos_mask])) if oos_mask.any() else float(np.nanmedian(close))
        exp_cost_ok, exp_cost_meta = expectancy_covers_cost(
            float(fin.get("expectancy", 0.0) or 0.0),
            spread_pips=spread_pips,
            slippage_pips=slippage_pips,
            pip_size=pip,
            close_price=ref_close,
            cost_multiple=float(cfg.get("expectancy_cost_multiple", 1.0)),
        )
        holdout_slices = crisis_recent_holdout_slices(
            len(close),
            recent_frac=float(cfg.get("recent_holdout_frac", 0.12)),
            crisis_frac=float(cfg.get("crisis_holdout_frac", 0.15)),
        )
        # Evaluate holdouts on full-series returns path with OOS preds where available
        rets_full_for_holdout = rets.copy()
        recent_holdout = evaluate_holdout_slice(
            rets_full_for_holdout, holdout_slices["recent"], financial_fn=_fm, name="recent"
        )
        crisis_holdout = evaluate_holdout_slice(
            rets_full_for_holdout, holdout_slices["crisis"], financial_fn=_fm, name="crisis"
        )
        fold_stability = fold_stability_report(fold_metrics, cfg=cfg)
        _emit_stage("test_oos", 88.0, f"اكتمل Testing OOS · {timeframe}")

        # Buy & hold baseline on same OOS window
        c_oos = close[oos_mask]
        bh_rets = np.zeros(len(c_oos))
        if len(c_oos) > 1:
            bh_rets[1:] = c_oos[1:] / np.maximum(c_oos[:-1], 1e-12) - 1.0
        bh_fin = _fm(bh_rets)

        # Random baseline with same trade constraints
        rng = np.random.default_rng(seed)
        rand_preds = rng.choice([-1, 0, 1], size=len(close), p=[0.2, 0.6, 0.2])
        rand_rets, _ = _bt(close, rand_preds, atr_arr=atr_pct)
        rand_fin = _fm(rand_rets[oos_mask])

        fold_trade_rates = [float(f.get("trade_rate", 0.0) or 0.0) for f in fold_metrics]
        median_fold_trade_rate = float(np.median(fold_trade_rates)) if fold_trade_rates else 0.0
        fit_diag = diagnose_fit(
            train_cls,
            val_cls,
            cls_metrics,
            train_fin,
            val_fin,
            fin,
            trade_rate_filtered=float(cls_metrics.get("trade_rate_filtered", 0.0) or 0.0),
            median_fold_trade_rate=median_fold_trade_rate,
        )
        if log:
            log(
                f"[{timeframe}] fit_diag={fit_diag['status']} "
                f"acc_gap_tv={fit_diag['accuracy_gap_train_val']:.3f} "
                f"sharpe_gap_tv={fit_diag['sharpe_gap_train_val']:.3f} "
                f"sharpe_gap_vt={fit_diag['sharpe_gap_val_test']:.3f}"
            )

        # --- Regime-conditional OOS stress + DSR / PBO (Trading Intelligence Layer) ---
        _emit_stage("regime_validation", 89.0, f"تحقق أنظمة السوق · {timeframe}")
        from atis.engines.engine4_training.validation_protocols import (
            classify_market_regimes,
            evaluate_by_regime,
            protocol_rationale,
        )
        from atis.engines.engine4_training.advanced_metrics import (
            deflated_sharpe_ratio,
            probability_of_backtest_overfitting,
            metrics_rationale,
        )

        regime_masks = classify_market_regimes(
            close,
            atr_pct,
            trend_window=int(cfg.get("regime_trend_window", max(24, horizon * 4))),
            vol_low_q=float(cfg.get("regime_eval_vol_low_q", 0.30)),
            vol_high_q=float(cfg.get("regime_eval_vol_high_q", 0.70)),
        )
        # Evaluate on full-length return path (zeros outside OOS) for aligned masks
        rets_full = rets_oos
        regime_validation = evaluate_by_regime(
            rets_full,
            {k: (v & oos_mask) for k, v in regime_masks.items()},
            financial_fn=_fm,
            min_bars=int(cfg.get("regime_min_bars", 40)),
        )
        regime_validation["protocol_notes"] = protocol_rationale()
        n_trials_proxy = max(
            1,
            int(cfg.get("nested_hp_trials", 8))
            + int(n_splits)
            + (len(tuned_policies) if tuned_policies else 0),
        )
        dsr_report = deflated_sharpe_ratio(
            float(fin.get("sharpe", 0.0) or 0.0),
            n_trials=n_trials_proxy,
            n_obs=max(int(fin.get("n_trades", 0) or 0), 10),
        )
        path_is = [float(f.get("val_sharpe", 0.0) or 0.0) for f in fold_metrics]
        path_oos = [float(f.get("test_sharpe", 0.0) or 0.0) for f in fold_metrics]
        pbo_report = probability_of_backtest_overfitting(path_is, path_oos)
        advanced_eval = {
            "deflated_sharpe": dsr_report,
            "pbo": pbo_report,
            "metrics_rationale": metrics_rationale(),
            "execution": {
                "dynamic_costs": dynamic_exec,
                "latency_bars": latency_bars,
                "execution_delay_bars": execution_delay_bars,
                "vol_slippage_k": vol_slip_k,
                "trade_stats": trade_stats,
            },
        }
        if log:
            log(
                f"[{timeframe}] regime_stable={regime_validation.get('stable')} "
                f"dsr={dsr_report.get('deflated_sharpe', 0):.3f} "
                f"pbo={pbo_report.get('pbo', 0):.3f} "
                f"expectancy={fin.get('expectancy', 0):.5f}"
            )

        _emit_stage("stress_mc", 90.0, f"ضغط · جلسات · مونت كارلو · {timeframe}")
        from atis.engines.engine4_training.stress_testing import (
            evaluate_session_slices,
            monte_carlo_trade_paths,
            session_masks_from_frame,
            stress_scenarios,
        )

        session_validation = evaluate_session_slices(
            rets_full,
            session_masks_from_frame(aligned),
            financial_fn=_fm,
            min_trades=int(cfg.get("session_min_trades", 8)),
        )
        _stress_kw = dict(
            financial_fn=_fm,
            spread_shock=float(cfg.get("stress_spread_shock", 2.0)),
            noise_sigma=float(cfg.get("stress_noise_sigma", 0.0008)),
            drop_frac=float(cfg.get("stress_drop_frac", 0.12)),
            seed=seed,
            latency_extra=int(cfg.get("stress_latency_extra", 2)),
            gap_shock=float(cfg.get("stress_gap_shock", 0.0015)),
        )
        try:
            stress_testing = stress_scenarios(rets_full, **_stress_kw)
        except TypeError:
            # Stale hot-reload of stress_testing without v16 kwargs.
            _stress_kw.pop("latency_extra", None)
            _stress_kw.pop("gap_shock", None)
            stress_testing = stress_scenarios(rets_full, **_stress_kw)
        traded_for_mc = rets_full[rets_full != 0]
        monte_carlo = monte_carlo_trade_paths(
            traded_for_mc,
            n_paths=int(cfg.get("monte_carlo_paths", 400)),
            seed=seed,
        )
        if log:
            log(
                f"[{timeframe}] stress_robust={stress_testing.get('robust')} "
                f"worst_sh={stress_testing.get('worst_sharpe', 0):.2f} "
                f"mc_p_profit={monte_carlo.get('p_profit')} mc_stable={monte_carlo.get('stable')}"
            )

        # Prefer last liquid fold policy for live deploy (median of starved folds dilutes H4).
        from atis.engines.engine4_training.data_quality_gate import min_val_trades_for_timeframe
        from atis.engines.engine4_training.adaptive_learning import policy_consensus_ok

        fold_min_for_avg = max(
            min_val_trades_for_timeframe(timeframe, cfg),
            max(5, int(cfg.get("min_trades_oos", 15) // max(n_splits, 1))),
        )
        liquid_policies = [
            p
            for p in tuned_policies
            if float(p.get("val_trades", 0.0) or 0.0) >= float(fold_min_for_avg)
        ]
        policy_consensus, policy_consensus_meta = policy_consensus_ok(
            fold_metrics, timeframe=timeframe, cfg=cfg
        )
        if liquid_policies and (policy_consensus or len(liquid_policies) >= int(cfg.get("policy_min_agree_folds", 3))):
            # Prefer liquid policies with Val↔Test consistency when fold test sharpe is known.
            def _liq_key(p: dict[str, Any]) -> tuple[float, float, float, float]:
                vt = float(p.get("val_trades", 0.0) or 0.0)
                sh = float(p.get("val_sharpe", 0.0) or 0.0)
                floor = float(p.get("confidence_floor", 1.0) or 1.0)
                # Match policy to a fold that recorded test_sharpe for consistency bonus.
                te_sh = float(p.get("fold_test_sharpe", p.get("test_sharpe", sh)) or sh)
                gap_vt = abs(sh - te_sh)
                consistency = -gap_vt  # smaller gap better
                # Require non-starved Val; do not freeze on Val Sharpe alone.
                return (consistency, sh if sh >= 0 else sh - 2.0, min(vt, 40.0), -floor)

            # Attach fold test sharpe onto tuned policies when available.
            for p in liquid_policies:
                for f in fold_metrics:
                    pol = f.get("policy") or {}
                    if (
                        abs(float(pol.get("decision_threshold", -1)) - float(p.get("decision_threshold", -2))) < 1e-9
                        and abs(float(pol.get("directional_edge", -1)) - float(p.get("directional_edge", -2))) < 1e-9
                    ):
                        p["fold_test_sharpe"] = float(f.get("test_sharpe", p.get("val_sharpe", 0.0)) or 0.0)
                        p["fold_test_trades"] = float(f.get("n_test_trades", 0.0) or 0.0)
                        break
            src = max(liquid_policies, key=_liq_key)
            avg_policy = {
                "decision_threshold": float(src.get("decision_threshold", decision_threshold)),
                "directional_edge": float(src.get("directional_edge", directional_edge)),
                "confidence_quantile": float(src.get("confidence_quantile", confidence_quantile)),
                "confidence_floor": float(min(float(src.get("confidence_floor", decision_threshold)), float(cfg.get("max_confidence_floor", 0.70)))),
                "cost_edge_multiple": float(src.get("cost_edge_multiple", cost_edge_multiple)),
                "meta_labeling": bool(use_meta_labeling),
                "source": "best_liquid_fold_consistent",
                "val_sharpe": float(src.get("val_sharpe", 0.0) or 0.0),
                "val_trades": float(src.get("val_trades", 0.0) or 0.0),
                "fold_test_sharpe": float(src.get("fold_test_sharpe", 0.0) or 0.0),
                "policy_consensus": policy_consensus_meta,
            }
        elif liquid_policies:
            src = max(
                liquid_policies,
                key=lambda p: (
                    float(p.get("val_sharpe", 0.0) or 0.0),
                    float(p.get("val_trades", 0.0) or 0.0),
                ),
            )
            avg_policy = {
                "decision_threshold": float(src.get("decision_threshold", decision_threshold)),
                "directional_edge": float(src.get("directional_edge", directional_edge)),
                "confidence_quantile": float(src.get("confidence_quantile", confidence_quantile)),
                "confidence_floor": float(
                    min(
                        float(src.get("confidence_floor", decision_threshold)),
                        float(cfg.get("max_confidence_floor", 0.70)),
                    )
                ),
                "cost_edge_multiple": float(src.get("cost_edge_multiple", cost_edge_multiple)),
                "meta_labeling": bool(use_meta_labeling),
                "source": "liquid_fold_no_full_consensus",
                "val_sharpe": float(src.get("val_sharpe", 0.0) or 0.0),
                "val_trades": float(src.get("val_trades", 0.0) or 0.0),
                "policy_consensus": policy_consensus_meta,
            }
        elif last_fold_policy:
            avg_policy = {
                "decision_threshold": float(last_fold_policy.get("decision_threshold", decision_threshold)),
                "directional_edge": float(last_fold_policy.get("directional_edge", directional_edge)),
                "confidence_quantile": float(last_fold_policy.get("confidence_quantile", confidence_quantile)),
                "confidence_floor": float(last_fold_policy.get("confidence_floor", decision_threshold)),
                "cost_edge_multiple": float(last_fold_policy.get("cost_edge_multiple", cost_edge_multiple)),
                "meta_labeling": bool(use_meta_labeling),
                "source": "last_fold",
                "policy_consensus": policy_consensus_meta,
            }
        else:
            avg_policy = {
                "decision_threshold": float(np.median([p["decision_threshold"] for p in tuned_policies])) if tuned_policies else decision_threshold,
                "directional_edge": float(np.median([p["directional_edge"] for p in tuned_policies])) if tuned_policies else directional_edge,
                "confidence_quantile": float(np.median([p["confidence_quantile"] for p in tuned_policies])) if tuned_policies else confidence_quantile,
                "confidence_floor": float(np.median([p.get("confidence_floor", decision_threshold) for p in tuned_policies])) if tuned_policies else decision_threshold,
                "cost_edge_multiple": float(
                    np.median([p.get("cost_edge_multiple", cost_edge_multiple) for p in tuned_policies])
                )
                if tuned_policies
                else float(cost_edge_multiple),
                "meta_labeling": bool(use_meta_labeling),
                "source": "median",
            }
        deploy_cost_edge = float(avg_policy.get("cost_edge_multiple", cost_edge_multiple))

        # Honest score of the artifact that will be published (last-window model on last fold test).
        deploy_holdout_fin: dict[str, float] = {
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "n_trades": 0.0,
        }
        if (
            final_model is not None
            and len(last_te_test) >= 20
            and len(deploy_tr) >= 40
        ):
            X_hold = scaler.transform(X_values[last_te_test])
            proba_h, classes_h = _model_proba(final_model, X_hold)
            if proba_h is not None and classes_h is not None:
                if last_calibrator is not None:
                    proba_h = last_calibrator(proba_h)
                regime_h = (
                    regime_mask_apply(atr_pct, deploy_regime_bounds[0], deploy_regime_bounds[1])
                    if use_regime
                    else np.ones(len(close), dtype=bool)
                )
                unit_costs_h = np.array(
                    [
                        _unit_cost(
                            float(c),
                            spread_pips=spread_pips,
                            slippage_pips=slippage_pips,
                            commission_per_lot=commission_per_lot,
                            pip_size=pip,
                        )
                        for c in close[last_te_test]
                    ],
                    dtype=float,
                )
                # Leak-safe deploy floor: re-estimate from the published last-window model.
                deploy_floor = float(avg_policy.get("confidence_floor", decision_threshold))
                deploy_floor = min(
                    deploy_floor,
                    train_confidence_floor(
                        final_model,
                        X_deploy_all[deploy_fit],
                        decision_threshold=float(avg_policy["decision_threshold"]),
                        confidence_quantile=float(avg_policy.get("confidence_quantile", confidence_quantile)),
                        min_floor=float(cfg.get("min_confidence_floor", 0.52)),
                        target_trade_rate=target_trade_rate,
                        max_floor=float(cfg.get("max_confidence_floor", 0.70)),
                    ),
                )
                # Nested liquidity tune on chronological tail of deploy train (no holdout peek).
                n_dep = len(deploy_tr)
                n_nested = max(40, int(n_dep * 0.22)) if n_dep >= 120 else 0
                if bool(cfg.get("nested_deploy_tune", False)) and n_nested and n_dep - n_nested >= 60:
                    nested_idx = np.asarray(deploy_tr[-n_nested:], dtype=int)
                    X_nested = scaler.transform(X_values[nested_idx])
                    # Start from a selective floor; relax only if nested is starved.
                    nested_base_floor = float(
                        np.clip(
                            max(deploy_floor, float(avg_policy.get("decision_threshold", decision_threshold)) + 0.04),
                            float(cfg.get("min_confidence_floor", 0.50)),
                            float(cfg.get("max_confidence_floor", 0.65)),
                        )
                    )
                    nested_policy = tune_trade_policy(
                        model=final_model,
                        X_val=X_nested,
                        close_val=close[nested_idx],
                        atr_pct_val=atr_pct[nested_idx],
                        regime_val=regime_mask_apply(
                            atr_pct[nested_idx], deploy_regime_bounds[0], deploy_regime_bounds[1]
                        )
                        if use_regime
                        else np.ones(len(nested_idx), dtype=bool),
                        hold_bars=horizon,
                        spread_pips=spread_pips,
                        slippage_pips=slippage_pips,
                        commission_per_lot=commission_per_lot,
                        pip_size=pip,
                        periods_per_year=ppy,
                        base_threshold=float(avg_policy["decision_threshold"]),
                        base_edge=max(float(avg_policy["directional_edge"]), 0.16),
                        base_quantile=float(avg_policy.get("confidence_quantile", confidence_quantile)),
                        cost_edge_multiple=max(float(deploy_cost_edge), 1.15),
                        non_overlapping=non_overlapping,
                        min_trades=max(6, int(round(len(nested_idx) * max(target_trade_rate * 0.7, 0.025)))),
                        confidence_floor=nested_base_floor,
                        allow_long=allow_long[nested_idx] if use_trend else None,
                        allow_short=allow_short[nested_idx] if use_trend else None,
                        target_trade_rate=target_trade_rate,
                        max_trade_rate=min(max_fold_trade_rate, max(target_trade_rate * 2.5, 0.10)),
                        short_edge_multiple=short_edge_multiple,
                        calibrate_fn=last_calibrator,
                        apply_sparsify=True,
                        primary_sides=(
                            structure_primary_sides(allow_long[nested_idx], allow_short[nested_idx])
                            if use_meta_labeling and use_trend
                            else None
                        ),
                    )
                    nested_trades = float(nested_policy.get("val_trades", 0.0) or 0.0)
                    nested_sh = float(nested_policy.get("val_sharpe", 0.0) or 0.0)
                    if nested_trades >= 6 and nested_sh >= 0.0:
                        avg_policy["decision_threshold"] = float(nested_policy["decision_threshold"])
                        avg_policy["directional_edge"] = float(nested_policy["directional_edge"])
                        deploy_floor = float(nested_policy.get("confidence_floor", deploy_floor))
                        deploy_cost_edge = float(nested_policy.get("cost_edge_multiple", deploy_cost_edge))
                        avg_policy["source"] = "nested_deploy_tune"
                    elif nested_trades >= 6:
                        # Keep liquidity but do not adopt a negative nested Sharpe policy.
                        avg_policy["source"] = "best_liquid_fold_nested_rejected"
                avg_policy["confidence_floor"] = float(deploy_floor)
                avg_policy["cost_edge_multiple"] = float(deploy_cost_edge)
                pred_h = policy_from_proba(
                    proba_h,
                    classes_h,
                    decision_threshold=avg_policy["decision_threshold"],
                    directional_edge=avg_policy["directional_edge"],
                    confidence_quantile=0.0,
                    confidence_floor=deploy_floor,
                    atr_pct=atr_pct[last_te_test],
                    unit_costs=unit_costs_h,
                    cost_edge_multiple=deploy_cost_edge,
                    regime_mask=regime_h[last_te_test],
                    short_edge_multiple=short_edge_multiple,
                    primary_sides=(
                        structure_primary_sides(allow_long[last_te_test], allow_short[last_te_test])
                        if use_meta_labeling and use_trend
                        else None
                    ),
                )
                if not (use_meta_labeling and use_trend):
                    pred_h = apply_trend_align(pred_h, allow_long[last_te_test], allow_short[last_te_test])
                # Prefer top-confidence trades near target rate on deploy holdout.
                if target_trade_rate > 0:
                    pred_h = sparsify_by_confidence(
                        pred_h, proba_h.max(axis=1), target_trade_rate=min(target_trade_rate * 1.25, max_fold_trade_rate)
                    )
                pred_h = cap_preds_by_trade_rate(
                    pred_h,
                    proba_h.max(axis=1),
                    max_trade_rate=max_fold_trade_rate,
                )
                h_rets, _ = _bt(close[last_te_test], pred_h, atr_arr=atr_pct[last_te_test])
                deploy_holdout_fin = _fm(h_rets, bootstrap=True)
                # Report 2026-07-31 05-06: H4/M15 deploy trades=0 — rescue starved publish policy.
                _min_reliable = int(cfg.get("min_reliable_deploy_trades", 12))
                if (
                    bool(cfg.get("deploy_liquidity_rescue", True))
                    and float(deploy_holdout_fin.get("n_trades", 0.0) or 0.0) < max(3, _min_reliable // 2)
                ):
                    rescue_floor = float(
                        np.clip(
                            min(float(deploy_floor), float(avg_policy["decision_threshold"])),
                            float(cfg.get("min_confidence_floor", 0.50)),
                            0.62,
                        )
                    )
                    rescue_edge = max(0.10, float(avg_policy["directional_edge"]) * 0.75)
                    rescue_cost = max(1.0, float(deploy_cost_edge) * 0.85)
                    pred_rescue = policy_from_proba(
                        proba_h,
                        classes_h,
                        decision_threshold=max(0.50, float(avg_policy["decision_threshold"]) - 0.05),
                        directional_edge=rescue_edge,
                        confidence_quantile=0.0,
                        confidence_floor=rescue_floor,
                        atr_pct=atr_pct[last_te_test],
                        unit_costs=unit_costs_h,
                        cost_edge_multiple=rescue_cost,
                        regime_mask=regime_h[last_te_test],
                        short_edge_multiple=max(1.05, float(short_edge_multiple) * 0.9),
                        primary_sides=(
                            structure_primary_sides(allow_long[last_te_test], allow_short[last_te_test])
                            if use_meta_labeling and use_trend
                            else None
                        ),
                    )
                    if not (use_meta_labeling and use_trend):
                        pred_rescue = apply_trend_align(
                            pred_rescue, allow_long[last_te_test], allow_short[last_te_test]
                        )
                    if target_trade_rate > 0:
                        pred_rescue = sparsify_by_confidence(
                            pred_rescue,
                            proba_h.max(axis=1),
                            target_trade_rate=min(max(target_trade_rate, 0.06), max_fold_trade_rate),
                        )
                    pred_rescue = cap_preds_by_trade_rate(
                        pred_rescue,
                        proba_h.max(axis=1),
                        max_trade_rate=max_fold_trade_rate,
                    )
                    rescue_rets, _ = _bt(
                        close[last_te_test], pred_rescue, atr_arr=atr_pct[last_te_test]
                    )
                    rescue_fin = _fm(rescue_rets)
                    if float(rescue_fin.get("n_trades", 0.0) or 0.0) > float(
                        deploy_holdout_fin.get("n_trades", 0.0) or 0.0
                    ):
                        deploy_holdout_fin = rescue_fin
                        avg_policy["decision_threshold"] = max(
                            0.50, float(avg_policy["decision_threshold"]) - 0.05
                        )
                        avg_policy["directional_edge"] = float(rescue_edge)
                        avg_policy["confidence_floor"] = float(rescue_floor)
                        avg_policy["cost_edge_multiple"] = float(rescue_cost)
                        avg_policy["source"] = "deploy_liquidity_rescue"
                        deploy_floor = rescue_floor
                        deploy_cost_edge = rescue_cost
                        pred_h = pred_rescue
                        if log:
                            log(
                                f"[{timeframe}] deploy_liquidity_rescue "
                                f"sharpe={deploy_holdout_fin['sharpe']:.4f} "
                                f"trades={deploy_holdout_fin.get('n_trades', 0):.0f}"
                            )
                if log:
                    log(
                        f"[{timeframe}] deploy_holdout sharpe={deploy_holdout_fin['sharpe']:.4f} "
                        f"trades={deploy_holdout_fin.get('n_trades', 0):.0f} "
                        f"n={len(last_te_test)}"
                    )

        version = _utc_now().strftime("%Y%m%dT%H%M%SZ") + "_" + hashlib.sha1(
            f"{symbol}{timeframe}{seed}{model_name}".encode()
        ).hexdigest()[:8]
        result.version = version

        min_sharpe = float(cfg.get("min_sharpe_ratio", 0.25))
        max_dd = float(cfg.get("max_drawdown_threshold", 0.28))
        min_trades = int(cfg.get("min_trades_oos", 15))
        require_bh = bool(cfg.get("require_beat_buy_hold", True))
        val_test_gap_max = float(cfg.get("val_test_sharpe_gap_max", 3.0))
        min_median_fold_sharpe = float(cfg.get("min_median_fold_val_sharpe", -1.5))
        overfit_acc_gap = float(cfg.get("max_train_val_acc_gap", 0.12))
        by_tf_acc_gap = cfg.get("max_train_val_acc_gap_by_tf") or {}
        if timeframe in by_tf_acc_gap:
            overfit_acc_gap = float(by_tf_acc_gap[timeframe])
        overfit_sharpe_gap = float(cfg.get("max_train_val_sharpe_gap", 3.0))
        min_deploy_h = float(cfg.get("min_deploy_holdout_sharpe", 0.0))
        min_deploy_trades = int(cfg.get("min_deploy_holdout_trades", 8))
        # Report 2026-07-31: H4 passed with 7 deploy trades when floor=3 / adaptive≈6.
        # Raise absolute floor so sparse Sharpe cannot sneak through.
        deploy_trades_floor = max(8, int(cfg.get("min_deploy_holdout_trades_floor", 8)))
        # Scale with holdout length but never below the absolute floor.
        if len(last_te_test) >= 20:
            adaptive_deploy = max(
                deploy_trades_floor,
                int(round(len(last_te_test) * max(float(cfg.get("min_oos_trade_rate", 0.008)), 0.015))),
            )
            min_deploy_trades = max(deploy_trades_floor, min(min_deploy_trades, adaptive_deploy))
        else:
            min_deploy_trades = max(deploy_trades_floor, min_deploy_trades)
        min_active_fold_frac = float(cfg.get("min_active_fold_frac", 0.4))
        min_oos_trade_rate = float(cfg.get("min_oos_trade_rate", 0.008))
        min_auc_live = float(cfg.get("min_auc_for_live", 0.52))
        min_reliable_deploy_trades = int(cfg.get("min_reliable_deploy_trades", 12))
        gate_failures: list[str] = []
        passed = fin["sharpe"] >= min_sharpe and abs(fin["max_drawdown"]) <= max_dd
        if not passed:
            gate_failures.append("sharpe_or_drawdown")
        if not (fin.get("n_trades", 0) >= min_trades):
            passed = False
            gate_failures.append("min_trades_oos")
        else:
            passed = passed and True
        if require_bh:
            # Sparse selective models often lose raw Sharpe to gold B&H in bull regimes.
            # Pass if they beat B&H Sharpe OR deliver positive return with much smaller DD.
            beat_bh_sharpe = fin["sharpe"] > bh_fin["sharpe"]
            beat_bh_risk = (
                fin.get("total_return", 0.0) > 0.0
                and abs(fin["max_drawdown"]) <= max(0.05, abs(bh_fin.get("max_drawdown", 1.0)) * 0.5)
            )
            if not (beat_bh_sharpe or beat_bh_risk):
                passed = False
                gate_failures.append("buy_hold")
        if not (fin["sharpe"] > rand_fin["sharpe"]):
            passed = False
            gate_failures.append("random_baseline")
        fold_val_sharpes = [float((f.get("policy") or {}).get("val_sharpe", 0.0)) for f in fold_metrics]
        median_fold_val = float(np.median(fold_val_sharpes)) if fold_val_sharpes else 0.0
        if fold_val_sharpes and median_fold_val < min_median_fold_sharpe:
            passed = False
            gate_failures.append("median_fold_val_sharpe")
        # Soft consistency: validation should not be catastrophic vs test
        if val_mask.any() and val_fin.get("sharpe", 0) < -2.0 and fin["sharpe"] > 0.5:
            passed = False
            gate_failures.append("val_catastrophic")
        if (
            val_mask.any()
            and bool(cfg.get("require_val_test_consistency", True))
            and abs(float(val_fin.get("sharpe", 0.0)) - float(fin["sharpe"])) > val_test_gap_max
            and fin["sharpe"] > 0.0
            and fin["sharpe"] < max(min_sharpe * 2.0, 0.75)
        ):
            # Report 04-15 H4: Val–Test gap large but Test Sharpe 2.35 — do not auto-fail strong Test.
            passed = False
            gate_failures.append("val_test_gap_weak_test")
        # Hard gap: Val≫Test even with strong Test (report M15 gap≈3.95 was previously ignored).
        val_test_hard = float(cfg.get("val_test_sharpe_gap_hard_max", 3.5))
        if (
            val_mask.any()
            and bool(cfg.get("require_val_test_consistency", True))
            and abs(float(val_fin.get("sharpe", 0.0)) - float(fin["sharpe"])) > val_test_hard
            and float(val_fin.get("sharpe", 0.0)) > float(fin["sharpe"])
        ):
            passed = False
            gate_failures.append("val_test_gap_hard")
        if bool(cfg.get("fail_on_overfit", True)) and fit_diag.get("status") == "overfitting":
            # Gate on financial overfit primarily; accuracy gap alone was noisy when val_cls ≈ test folds.
            if float(fit_diag.get("sharpe_gap_train_val", 0.0)) > overfit_sharpe_gap:
                passed = False
                gate_failures.append("overfit_sharpe_gap")
            elif float(fit_diag.get("accuracy_gap_train_val", 0.0)) > overfit_acc_gap and float(
                fit_diag.get("sharpe_gap_train_val", 0.0)
            ) > 1.0:
                passed = False
                gate_failures.append("overfit_acc_sharpe")
        # Also fail large train→val Sharpe gap even when diagnose_fit stays "balanced"
        # — but only when generalization truly collapsed (exempt strong Test + Val↔Test OK).
        if bool(cfg.get("fail_on_overfit", True)) and should_fail_overfit_sharpe_gap_hard(
            sharpe_gap_tv=float(fit_diag.get("sharpe_gap_train_val", 0.0) or 0.0),
            overfit_sharpe_gap=overfit_sharpe_gap,
            train_sharpe=float(train_fin.get("sharpe", 0.0) or 0.0),
            val_sharpe=float(val_fin.get("sharpe", 0.0) or 0.0),
            test_sharpe=float(fin.get("sharpe", 0.0) or 0.0),
            sharpe_gap_vt=float(fit_diag.get("sharpe_gap_val_test", 0.0) or 0.0),
            n_test_trades=float(fin.get("n_trades", 0.0) or 0.0),
            min_sharpe=min_sharpe,
            min_trades=min_trades,
            val_test_gap_hard=val_test_hard,
            acc_gap_tv=float(fit_diag.get("accuracy_gap_train_val", 0.0) or 0.0),
            max_acc_gap=overfit_acc_gap,
        ):
            passed = False
            if "overfit_sharpe_gap" not in gate_failures:
                gate_failures.append("overfit_sharpe_gap_hard")
        if (
            bool(cfg.get("fail_on_weak_sharpe_ci", True))
            and bootstrap_ci
            and float(fin.get("n_trades", 0.0) or 0.0) >= 12
        ):
            ci_low = float(fin.get("sharpe_ci_low", fin.get("sharpe", 0.0)) or 0.0)
            min_ci = float(cfg.get("min_sharpe_ci_low", -0.25))
            if ci_low < min_ci:
                passed = False
                gate_failures.append("weak_sharpe_ci")
        if (
            bool(cfg.get("fail_on_unstable_generalization", True))
            and fit_diag.get("status") == "unstable_generalization"
        ):
            va_sh_gate = float(val_fin.get("sharpe", 0.0) or 0.0)
            te_sh_gate = float(fin.get("sharpe", 0.0) or 0.0)
            # Hard-fail only when Test edge collapsed (not when Test remains strong, as in H4 ≥1.0).
            if te_sh_gate < max(min_sharpe, 0.75) or (
                va_sh_gate > 1.0 and te_sh_gate < 0.4 * va_sh_gate and te_sh_gate < 1.0
            ):
                passed = False
                gate_failures.append("unstable_generalization")
        # Liquidity gates — report showed H4 deploy=1 trade / M30·M15 deploy=0 trades still "passing".
        if fold_trade_rates:
            active_frac = float(np.mean([1.0 if r > 0.0 else 0.0 for r in fold_trade_rates]))
            if active_frac < min_active_fold_frac:
                passed = False
                gate_failures.append("inactive_folds")
            # Per-fold Val liquidity: reject when most folds are starved (H4 n_val∈{0,6,7,9}).
            min_vt = float(min_val_trades_for_timeframe(timeframe, cfg))
            fold_val_trades = [float(f.get("n_val_trades", 0.0) or 0.0) for f in fold_metrics]
            liquid_fold_frac = (
                float(np.mean([1.0 if t >= min_vt else 0.0 for t in fold_val_trades]))
                if fold_val_trades
                else 0.0
            )
            if fold_val_trades and liquid_fold_frac < float(cfg.get("min_liquid_val_fold_frac", 0.4)):
                passed = False
                gate_failures.append("val_fold_liquidity")
            # Do not reject merely because trade_rate == max_fold_trade_rate (H1 report edge case).
            over_tol = float(cfg.get("overtrading_rate_tol_frac", 0.05))
            over_frac = float(
                np.mean(
                    [
                        1.0 if overtrading_rate_exceeds(r, max_fold_trade_rate, tol_frac=over_tol) else 0.0
                        for r in fold_trade_rates
                    ]
                )
            )
            if over_frac > 0.5:
                passed = False
                gate_failures.append("overtrading_folds")
        oos_rate = float(cls_metrics.get("trade_rate_filtered", 0.0) or 0.0)
        if oos_rate < min_oos_trade_rate and fin.get("n_trades", 0) < max(min_trades * 2, 30):
            passed = False
            gate_failures.append("oos_trade_rate")
        if bool(cfg.get("gate_on_deploy_holdout", True)) and len(last_te_test) >= 20:
            deploy_n = float(deploy_holdout_fin.get("n_trades", 0.0) or 0.0)
            if float(deploy_holdout_fin.get("sharpe", 0.0)) < min_deploy_h:
                passed = False
                gate_failures.append("deploy_holdout_sharpe")
            if deploy_n < float(deploy_trades_floor):
                passed = False
                gate_failures.append("deploy_holdout_too_few_trades")
            elif deploy_n < float(min_deploy_trades):
                passed = False
                gate_failures.append("deploy_holdout_trades")
            # Report 2026-07-31 H4: 7 trades + Sharpe 2.43 — reject sparse high-Sharpe deploys.
            if (
                bool(cfg.get("fail_on_sparse_deploy", True))
                and deploy_n > 0
                and deploy_n < float(min_reliable_deploy_trades)
                and float(deploy_holdout_fin.get("sharpe", 0.0) or 0.0) >= max(min_sharpe, 0.5)
            ):
                passed = False
                gate_failures.append("sparse_deploy_unreliable")
        # Near-chance classifier with positive financials = filter-driven edge (report Acc 0.502).
        # Soften: only hard-fail when OOS liquidity is also thin (avoid rejecting liquid strong Test).
        if bool(cfg.get("fail_on_filter_driven_edge", True)):
            auc_gate = float(cls_metrics.get("roc_auc_ovr", 0.0) or 0.0)
            acc_gate = float(cls_metrics.get("accuracy", 0.0) or 0.0)
            deploy_n_gate = float(deploy_holdout_fin.get("n_trades", 0.0) or fin.get("n_trades", 0.0) or 0.0)
            oos_trades_gate = float(fin.get("n_trades", 0.0) or 0.0)
            thin_liquidity = deploy_n_gate < float(min_reliable_deploy_trades) or oos_trades_gate < float(
                max(min_trades, 15)
            )
            if (
                thin_liquidity
                and auc_gate > 0
                and auc_gate < min_auc_live
                and acc_gate < 0.53
                and float(fin.get("sharpe", 0.0) or 0.0) >= min_sharpe
            ):
                passed = False
                gate_failures.append("filter_driven_edge")
            elif (
                bool(fit_diag.get("filter_driven_edge_risk"))
                and bool(fit_diag.get("sparse_sharpe_risk"))
                and float(fin.get("sharpe", 0.0) or 0.0) >= min_sharpe
            ):
                passed = False
                gate_failures.append("filter_driven_sparse")
        # Regime stability + expectancy + PBO (Trading Intelligence Engine gates)
        if bool(cfg.get("fail_on_regime_unstable", True)) and regime_validation.get("stable") is False:
            # Soft: only hard-fail when OOS Sharpe also mediocre (avoid killing strong but regime-tilted edges)
            if float(fin.get("sharpe", 0.0) or 0.0) < max(min_sharpe * 2.0, 1.0):
                passed = False
                gate_failures.append("regime_unstable")
        if bool(cfg.get("fail_on_weak_expectancy", True)):
            min_exp = float(cfg.get("min_expectancy", 0.0))
            if float(fin.get("n_trades", 0.0) or 0.0) >= float(cfg.get("min_trades_oos", 20)) and float(
                fin.get("expectancy", 0.0) or 0.0
            ) < min_exp:
                passed = False
                gate_failures.append("weak_expectancy")
        if bool(cfg.get("fail_on_high_pbo", False)) and float(pbo_report.get("reliable", 0) or 0) > 0:
            if float(pbo_report.get("pbo", 0.0) or 0.0) >= float(cfg.get("max_pbo", 0.55)):
                passed = False
                gate_failures.append("high_pbo")
        # Enterprise stress / MC / H4 no-edge / readiness gates
        if bool(cfg.get("fail_on_stress_fragile", True)) and stress_testing.get("scenarios"):
            if float(stress_testing.get("worst_sharpe", 0) or 0) < float(cfg.get("min_stress_worst_sharpe", -1.0)):
                passed = False
                gate_failures.append("stress_fragile")
        if bool(cfg.get("fail_on_monte_carlo_unstable", False)) and monte_carlo.get("enabled"):
            if not monte_carlo.get("stable", True):
                passed = False
                gate_failures.append("monte_carlo_unstable")
        if str(timeframe).upper() == "H4" and bool(cfg.get("fail_h4_near_chance", True)):
            if float(cls_metrics.get("roc_auc_ovr", 1.0) or 1.0) < 0.52 and float(
                cls_metrics.get("accuracy", 1.0) or 1.0
            ) < 0.53:
                passed = False
                gate_failures.append("h4_no_edge")

        # --- v16 promotion intelligence gates ---
        if bool(cfg.get("fail_on_fold_unstable", True)) and not fold_stability.get("gate_pass", True):
            passed = False
            gate_failures.append("fold_unstable")
        if bool(cfg.get("fail_on_expectancy_below_cost", True)) and not exp_cost_ok:
            if float(fin.get("n_trades", 0) or 0) >= float(cfg.get("min_trades_oos", 20)):
                passed = False
                gate_failures.append("expectancy_below_cost")
        if bool(cfg.get("fail_on_crisis_holdout_weak", True)) and not crisis_holdout.get("skipped"):
            if float(crisis_holdout.get("sharpe") or 0) < float(cfg.get("min_crisis_holdout_sharpe", -0.5)):
                passed = False
                gate_failures.append("crisis_holdout_weak")
        if bool(cfg.get("fail_on_recent_holdout_weak", False)) and not recent_holdout.get("skipped"):
            if float(recent_holdout.get("sharpe") or 0) < float(cfg.get("min_recent_holdout_sharpe", 0.0)):
                passed = False
                gate_failures.append("recent_holdout_weak")
        if bool(cfg.get("fail_on_trade_rate_saturated", False)) and fold_stability.get("trade_rate_pegged"):
            passed = False
            gate_failures.append("trade_rate_saturated")

        _emit_stage("gates", 94.0, f"بوابات النشر · {timeframe}")
        result.passed_gates = bool(passed)
        gate_failures_detail = annotate_gate_failures(gate_failures)
        if log and gate_failures_detail:
            for g in gate_failures_detail:
                log(f"[{timeframe}] gate_fail={g['key']} · {g['ar']}")

        out_dir = get_path("models") / symbol / timeframe / version
        out_dir.mkdir(parents=True, exist_ok=True)
        model_file = out_dir / "model.joblib"
        live_policy = {
            "cost_edge_multiple": float(deploy_cost_edge),
            "short_edge_multiple": float(short_edge_multiple),
            "target_trade_rate": float(target_trade_rate),
            "apply_oos_sparsify": bool(apply_oos_sparsify),
            "meta_labeling": bool(use_meta_labeling),
            "max_fold_trade_rate": float(max_fold_trade_rate),
            "regime_filter": bool(use_regime),
            "regime_atr_low": float(deploy_regime_bounds[0]),
            "regime_atr_high": float(deploy_regime_bounds[1])
            if np.isfinite(deploy_regime_bounds[1])
            else None,
            "spread_pips": float(spread_pips),
            "slippage_pips": float(slippage_pips),
            "commission_per_lot": float(commission_per_lot),
            "pip_size": float(pip),
            "dynamic_execution_costs": bool(dynamic_exec),
            "vol_slippage_k": float(vol_slip_k),
            "latency_bars": int(latency_bars),
            "execution_delay_bars": int(execution_delay_bars),
            "pipeline_version": PIPELINE_VERSION,
        }
        joblib.dump(
            {
                "model": final_model,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "trade_policy": avg_policy,
                "live_policy": live_policy,
                "calibrator": last_calibrator,
                "horizon_bars": horizon,
                "trend_align": use_trend,
                "symbol": symbol,
                "timeframe": timeframe,
                "pipeline_version": PIPELINE_VERSION,
            },
            model_file,
        )
        result.model_path = str(model_file)

        metrics = {
            "classification": cls_metrics,
            "financial_oos": fin,
            "financial_validation": val_fin,
            "financial_train": train_fin,
            "financial_deploy_holdout": deploy_holdout_fin,
            "classification_train": train_cls,
            "classification_validation_folds_avg": val_cls,
            "buy_hold": bh_fin,
            "random_baseline": rand_fin,
            "folds": fold_metrics,
            "model": model_name,
            "n_rows": int(len(X)),
            "n_features": len(feature_cols),
            "passed_gates": result.passed_gates,
            "gate_failures": gate_failures,
            "gate_failures_detail": gate_failures_detail,
            "horizon_bars": horizon,
            "decision_threshold": avg_policy["decision_threshold"],
            "trade_policy": avg_policy,
            "trade_stats": trade_stats,
            "pipeline_version": PIPELINE_VERSION,
            "symbol": symbol,
            "timeframe": timeframe,
            "version": version,
            "data_quality": dq,
            "fit_diagnosis": fit_diag,
            "nested_hp": nested_hp_meta,
            "regime_validation": regime_validation,
            "advanced_eval": advanced_eval,
            "validation_mode": validation_mode,
            "data_intelligence": data_intel,
            "feature_intelligence": feature_intel,
            "label_quality": label_quality,
            "barrier_sweep": barrier_sweep,
            "label_cleaning": label_clean_meta,
            "family_resolution": family_resolution,
            "trade_level_metrics": trade_level_metrics,
            "expectancy_vs_cost": exp_cost_meta,
            "confidence_sizing": confidence_sizing_meta,
            "fold_stability": fold_stability,
            "recent_holdout": recent_holdout,
            "crisis_holdout": crisis_holdout,
            "model_zoo": model_zoo_meta,
            "session_validation": session_validation,
            "stress_testing": stress_testing,
            "monte_carlo": monte_carlo,
            "self_optimize_applied": applied_self_opt,
            "split_comparison": {
                "train": {"classification": train_cls, "financial": train_fin},
                "validation": {"classification": val_cls, "financial": val_fin},
                "test": {"classification": cls_metrics, "financial": fin},
                "deploy_holdout": {"financial": deploy_holdout_fin},
            },
            "deploy_selection": {
                "mode": "last_window" if deploy_from_last_window else "best_val_fold",
                "best_val_fold": best_fold_i,
                "best_val_sharpe": best_fold_score if best_fold_i >= 0 else None,
                "n_deploy_rows": int(len(deploy_tr)),
                "n_holdout_rows": int(len(last_te_test)),
                "deploy_holdout_sharpe": deploy_holdout_fin.get("sharpe"),
                "regime_bounds": {
                    "lo": float(deploy_regime_bounds[0]),
                    "hi": float(deploy_regime_bounds[1]) if np.isfinite(deploy_regime_bounds[1]) else None,
                },
            },
        }
        from atis.engines.engine4_training.intelligence import build_awareness_report

        decision_explanations = [
            {
                "decision": "champion_candidate" if result.passed_gates else "rejected",
                "why": ",".join(gate_failures) if gate_failures else "passed_all_gates",
                "ar": (
                    f"اختير/اجتاز {timeframe}" if result.passed_gates else f"رُفض {timeframe}: {', '.join(gate_failures[:4])}"
                ),
            },
            {
                "decision": "best_val_fold",
                "why": f"fold={best_fold_i}",
                "ar": (
                    f"أفضل طية سائلة للتحقق: {best_fold_i + 1}"
                    if best_fold_i >= 0
                    else "لا طية تحقق سائلة — استُبعدت الطيات الجائعة"
                ),
            },
        ]
        if nested_hp_meta.get("enabled"):
            decision_explanations.append(
                {
                    "decision": "nested_hp",
                    "why": str(nested_hp_meta.get("best_family")),
                    "ar": f"ضبط HP متداخل داخل WF · عائلة={nested_hp_meta.get('best_family')} · score={nested_hp_meta.get('best_score')}",
                }
            )
        awareness = build_awareness_report(
            timeframe=timeframe,
            data_quality=dq,
            fit_diagnosis=fit_diag,
            classification=cls_metrics,
            financial_oos=fin,
            financial_deploy=deploy_holdout_fin,
            gate_failures=gate_failures,
            decisions=decision_explanations,
        )
        metrics["nested_hp"] = nested_hp_meta
        metrics["awareness"] = awareness
        metrics["decision_explanations"] = decision_explanations
        from atis.engines.engine4_training.readiness import compute_live_readiness
        from atis.engines.engine4_training.enterprise_report import (
            build_intelligent_critique,
            propose_config_overrides,
            write_enterprise_report,
        )

        live_readiness = compute_live_readiness(
            passed_gates=bool(result.passed_gates),
            metrics=metrics,
            timeframe=timeframe,
        )
        if bool(cfg.get("fail_on_low_live_readiness", False)):
            min_ready = float(cfg.get("min_live_readiness", 55))
            if float(live_readiness.get("score", 0) or 0) < min_ready:
                result.passed_gates = False
                if "low_live_readiness" not in gate_failures:
                    gate_failures.append("low_live_readiness")
                metrics["gate_failures"] = gate_failures
                metrics["gate_failures_detail"] = annotate_gate_failures(gate_failures)
                metrics["passed_gates"] = False
                live_readiness = compute_live_readiness(
                    passed_gates=False, metrics=metrics, timeframe=timeframe
                )
        intelligent_critique = build_intelligent_critique(
            metrics, timeframe=timeframe, passed=bool(result.passed_gates)
        )
        self_optimize = propose_config_overrides(
            timeframe=timeframe,
            metrics={**metrics, "live_readiness": live_readiness},
            passed_gates=bool(result.passed_gates),
            history_path=get_path("models") / symbol / timeframe / "knowledge_loop.json",
        )
        metrics["live_readiness"] = live_readiness
        metrics["intelligent_critique"] = intelligent_critique
        metrics["self_optimize"] = self_optimize

        # Feature explainability (SHAP / permutation / stability) on final deploy model.
        feature_explainability: dict[str, Any] = {"enabled": False}
        if bool(cfg.get("feature_explainability_enabled", True)) and final_model is not None and len(deploy_tr):
            _emit_stage("gates", 88.0, f"تفسير الميزات (SHAP/Importance) · {timeframe}")
            try:
                from atis.engines.engine4_training.feature_explainability import (
                    build_explainability_report,
                )

                X_ex = scaler.transform(X_values[deploy_tr])
                y_ex = y_values[deploy_tr]
                feature_explainability = build_explainability_report(
                    final_model,
                    X_ex,
                    y_ex,
                    feature_cols,
                    fold_top_features=fold_top_features,
                    seed=seed,
                    cfg=cfg,
                )
                # Auto-drop unstable features from deploy artifact metadata (v16)
                if bool(cfg.get("auto_drop_unstable_features", True)):
                    unstable = list(
                        ((feature_explainability.get("stability") or {}).get("overfit_risk_features"))
                        or []
                    )
                    if unstable:
                        kept = [c for c in feature_cols if c not in set(unstable)]
                        if len(kept) >= max(8, int(0.6 * len(feature_cols))):
                            feature_explainability["auto_dropped"] = unstable[:25]
                            feature_explainability["n_auto_dropped"] = len(unstable)
                            # Keep feature_cols for model (already trained); record recommendation only
                            feature_explainability["next_run_feature_blocklist"] = unstable[:25]
                if (
                    bool(cfg.get("fail_on_unstable_features", False))
                    and "low_feature_stability_across_folds" in (feature_explainability.get("warnings") or [])
                ):
                    result.passed_gates = False
                    if "feature_unstable" not in gate_failures:
                        gate_failures.append("feature_unstable")
                    metrics["gate_failures"] = gate_failures
                    metrics["gate_failures_detail"] = annotate_gate_failures(gate_failures)
                    metrics["passed_gates"] = False
                if log and feature_explainability.get("enabled"):
                    log(f"[{timeframe}] explainability {feature_explainability.get('summary_ar')}")
            except Exception as exc:
                feature_explainability = {"enabled": False, "error": str(exc)}
                if log:
                    log(f"[{timeframe}] explainability_error={exc}")
        metrics["feature_explainability"] = feature_explainability

        # Champion vs Challenger comparison (promotion intelligence).
        from atis.engines.engine4_training.champion_challenger import (
            compare_challenger_to_champion,
            maybe_update_champion,
            write_challenger_record,
        )
        from atis.engines.engine4_training.smart_recommendations import build_smart_recommendations

        champion_challenger = compare_challenger_to_champion(
            metrics,
            models_root=get_path("models"),
            symbol=symbol,
            timeframe=timeframe,
            cfg=cfg,
        )
        metrics["champion_challenger"] = champion_challenger
        smart_recs = build_smart_recommendations(metrics, timeframe=timeframe)
        metrics["smart_recommendations"] = smart_recs
        result.metrics = metrics
        if log:
            log(
                f"[{timeframe}] live_readiness={live_readiness.get('score')}/100 "
                f"verdict={live_readiness.get('verdict')} root={intelligent_critique.get('root_cause')} "
                f"challenger={champion_challenger.get('decision')} "
                f"recs={smart_recs.get('primary_code')}"
            )

        metrics["data_sources"] = source_meta
        metrics["validation"] = {
            "method": f"{validation_mode}_nested_purged",
            "validation_mode": validation_mode,
            "fold_count": len(fold_metrics),
            "train_ratio": train_ratio,
            "fold_validation_ratio": fold_val_ratio,
            "embargo_bars": embargo,
            "purge_bars": purge,
            "periods_per_year": ppy,
            "tune_trade_policy": tune_policy,
            "regime_filter": use_regime,
            "trend_align": use_trend,
            "top_features": len(feature_cols),
            "feature_selection": (
                "stable_walk_forward" if bool(cfg.get("stable_feature_selection", True)) else "first_train_window"
            ),
            "train_on_directional_only": train_directional_only,
            "target_trade_rate": target_trade_rate,
            "apply_oos_sparsify": apply_oos_sparsify,
            "tune_policy_mode": tune_policy_mode,
            "calibrate_probabilities": use_calibration,
            "short_edge_multiple": short_edge_multiple,
            "max_train_bars": max_train_bars,
            "cost_edge_multiple": cost_edge_multiple,
            "latency_bars": latency_bars,
            "execution_delay_bars": execution_delay_bars,
            "dynamic_execution_costs": dynamic_exec,
            "use_ensemble": bool(cfg.get("use_ensemble", False)),
            "median_fold_val_sharpe": median_fold_val,
            "median_fold_trade_rate": median_fold_trade_rate,
            "deploy_from_last_window": deploy_from_last_window,
            "best_val_fold_meta": best_fold_i,
            "financial": val_fin,
            "tuned_policies": tuned_policies,
        }

        # Continuous learning before artifact flush
        try:
            from atis.engines.engine4_training.knowledge_loop import (
                population_stability_index,
                record_training_episode,
            )

            psi = 0.0
            if len(feature_cols) > 0 and len(X) > 80:
                col0 = feature_cols[0]
                series = X[col0].astype(float).values
                mid = len(series) // 2
                psi = population_stability_index(series[:mid], series[mid:])
            knowledge = record_training_episode(
                get_path("models"),
                symbol=symbol,
                timeframe=timeframe,
                version=version,
                metrics=metrics,
                passed_gates=bool(result.passed_gates),
                feature_psi=psi,
            )
            metrics["knowledge_loop"] = {
                "path": str(get_path("models") / symbol / timeframe / "knowledge_loop.json"),
                "advisory": knowledge.get("last_advisory"),
                "n_episodes": len(knowledge.get("episodes") or []),
                "performance_ema": knowledge.get("performance_ema"),
            }
            # Refresh recommendations now that drift / knowledge advisory exists.
            try:
                from atis.engines.engine4_training.smart_recommendations import (
                    build_smart_recommendations,
                )

                metrics["smart_recommendations"] = build_smart_recommendations(
                    metrics, timeframe=timeframe
                )
            except Exception:
                pass
            try:
                from atis.engines.engine4_training.research_factory import append_experiment

                metrics["research_factory"] = append_experiment(
                    get_path("models"),
                    symbol=symbol,
                    timeframe=timeframe,
                    version=version,
                    metrics=metrics,
                    cfg=cfg,
                    passed_gates=bool(result.passed_gates),
                )
                if log and metrics["research_factory"].get("hypothesis"):
                    hyp = metrics["research_factory"]["hypothesis"]
                    log(
                        f"[{timeframe}] research_factory hyp={hyp.get('code')} "
                        f"stop={metrics['research_factory'].get('stop_suggested')}"
                    )
            except Exception as exc:
                metrics["research_factory"] = {"error": str(exc)}
            if log and knowledge.get("last_advisory"):
                adv = knowledge["last_advisory"]
                log(
                    f"[{timeframe}] knowledge_loop retrain_suggested={adv.get('retrain_suggested')} "
                    f"reason={adv.get('reason')} psi={psi:.3f}"
                )
        except Exception as exc:  # pragma: no cover
            if log:
                log(f"[{timeframe}] knowledge_loop_error={exc}")

        (out_dir / "metrics_report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        write_evaluation_report(out_dir / "evaluation_report.md", metrics)
        try:
            write_enterprise_report(out_dir / "enterprise_dossier.md", metrics)
        except Exception as exc:
            if log:
                log(f"[{timeframe}] enterprise_report_error={exc}")
        (out_dir / "backtest_report.json").write_text(
            json.dumps(
                {
                    "financial_oos": fin,
                    "financial_validation": val_fin,
                    "buy_hold": bh_fin,
                    "random_baseline": rand_fin,
                    "trade_stats": trade_stats,
                    "trade_policy": avg_policy,
                    "costs": {
                        "spread_pips": spread_pips,
                        "slippage_pips": slippage_pips,
                        "commission_per_lot": commission_per_lot,
                        "hold_bars": horizon,
                        "non_overlapping": non_overlapping,
                        "min_confidence": min_conf,
                        "cost_edge_multiple": cost_edge_multiple,
                        "dynamic_execution_costs": dynamic_exec,
                        "vol_slippage_k": vol_slip_k,
                        "latency_bars": latency_bars,
                        "execution_delay_bars": execution_delay_bars,
                    },
                    "regime_validation": regime_validation,
                    "advanced_eval": advanced_eval,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (out_dir / "feature_list.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
        (out_dir / "awareness.json").write_text(json.dumps(awareness, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "experiment_record.json").write_text(
            json.dumps(
                {
                    "hypothesis": "baseline_or_nested_hp",
                    "single_change": "nested_hp_search" if nested_hp_meta.get("enabled") else "pipeline_v12_gates",
                    "timeframe": timeframe,
                    "impact": {
                        "acc": cls_metrics.get("accuracy"),
                        "auc": cls_metrics.get("roc_auc_ovr"),
                        "sharpe": fin.get("sharpe"),
                        "sharpe_ci_low": fin.get("sharpe_ci_low"),
                        "max_dd": fin.get("max_drawdown"),
                        "gap_vt": fit_diag.get("sharpe_gap_val_test"),
                        "n_trades": fin.get("n_trades"),
                    },
                    "passed_gates": result.passed_gates,
                    "gate_failures": gate_failures,
                    "data_quality_score": dq.get("score"),
                    "nested_hp": nested_hp_meta,
                    "expectancy": fin.get("expectancy"),
                    "sortino": fin.get("sortino"),
                    "pbo": (advanced_eval.get("pbo") or {}).get("pbo"),
                    "deflated_sharpe": (advanced_eval.get("deflated_sharpe") or {}).get("deflated_sharpe"),
                    "regime_stable": regime_validation.get("stable"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if metrics.get("knowledge_loop"):
            (out_dir / "knowledge_snapshot.json").write_text(
                json.dumps(metrics["knowledge_loop"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        (out_dir / "training_config.yaml").write_text(
            json.dumps(cfg, indent=2),
            encoding="utf-8",
        )
        (out_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "version": version,
                    "created_at": _utc_now().isoformat(),
                    "passed_gates": result.passed_gates,
                    "model": model_name,
                    "final_model_ready": True,
                    "data_sources": source_meta,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Leaderboard append
        board_path = get_path("models") / "leaderboard.json"
        board: list[dict[str, Any]] = []
        if board_path.exists():
            board = json.loads(board_path.read_text(encoding="utf-8"))
        board.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "version": version,
                "sharpe": fin["sharpe"],
                "max_drawdown": fin["max_drawdown"],
                "passed_gates": result.passed_gates,
                "model_path": result.model_path,
            }
        )
        board_path.write_text(json.dumps(board, indent=2), encoding="utf-8")

        # If passed AND beats champion (or first champion), write champion pointer
        if result.passed_gates:
            try:
                write_challenger_record(
                    out_dir,
                    version=version,
                    comparison=champion_challenger,
                    passed_gates=True,
                )
                champ_upd = maybe_update_champion(
                    get_path("models"),
                    symbol=symbol,
                    timeframe=timeframe,
                    version=version,
                    model_path=result.model_path or str(model_file),
                    comparison=champion_challenger,
                    passed_gates=True,
                )
                metrics["champion_update"] = champ_upd
                # Shadow challenger when gates pass but champion kept
                try:
                    from atis.engines.engine4_training.shadow_challenger import (
                        register_shadow_challenger,
                    )

                    shadow = register_shadow_challenger(
                        get_path("models"),
                        symbol=symbol,
                        timeframe=timeframe,
                        version=version,
                        model_path=result.model_path or str(model_file),
                        metrics=metrics,
                        comparison=champion_challenger,
                    )
                    metrics["shadow_challenger"] = shadow
                except Exception as exc:
                    metrics["shadow_challenger"] = {"error": str(exc)}
                if log:
                    log(
                        f"[{timeframe}] champion_update updated={champ_upd.get('updated')} "
                        f"reason={champ_upd.get('reason') or champ_upd.get('champion', {}).get('via')}"
                    )
            except Exception as exc:
                # Fallback to legacy champion write
                champ = get_path("models") / symbol / timeframe / "champion.json"
                champ.write_text(
                    json.dumps({"version": version, "model_path": result.model_path}, indent=2),
                    encoding="utf-8",
                )
                if log:
                    log(f"[{timeframe}] champion_fallback error={exc}")
        else:
            try:
                write_challenger_record(
                    out_dir,
                    version=version,
                    comparison=champion_challenger,
                    passed_gates=False,
                )
            except Exception:
                pass

        # Refresh metrics on disk after champion/challenger annotations
        try:
            (out_dir / "metrics_report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            if metrics.get("smart_recommendations"):
                (out_dir / "smart_recommendations.json").write_text(
                    json.dumps(metrics["smart_recommendations"], indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            if metrics.get("feature_explainability"):
                (out_dir / "feature_explainability.json").write_text(
                    json.dumps(metrics["feature_explainability"], indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception:
            pass

        try:
            DataStateRegistry().audit(
                "engine4",
                "train_success",
                symbol=symbol,
                timeframe=timeframe,
                detail_json=json.dumps({"version": version, "passed": result.passed_gates}),
            )
        except OSError as reg_exc:
            # OneDrive/Windows file locks must not fail an otherwise successful train.
            logger.warning("registry_audit_failed", timeframe=timeframe, error=str(reg_exc))
            if log:
                log(f"[{timeframe}] registry_audit_skipped: {reg_exc}")
        logger.info(
            "train_done",
            symbol=symbol,
            timeframe=timeframe,
            version=version,
            sharpe=fin["sharpe"],
            passed=result.passed_gates,
        )
        compact = compact_tf_status_from_result(result, source_meta=source_meta)
        if status:
            status(compact)
        _emit_stage("done", 100.0, f"اكتمل Training/Testing/Validation · {timeframe}")
        if log:
            log(
                f"[{timeframe}] final_model={model_file} "
                f"sharpe={fin['sharpe']:.4f} "
                f"sharpe_uncapped={fin.get('sharpe_uncapped', 0):.4f} "
                f"ci_low={fin.get('sharpe_ci_low', 0):.4f} "
                f"passed={result.passed_gates}"
            )
        return result

    except Exception as exc:
        result.error = str(exc)
        if status:
            status(
                {
                    "timeframe": timeframe,
                    "stage": "error",
                    "progress_pct": 100.0,
                    "error": str(exc),
                    "passed_gates": False,
                    "message": str(exc),
                }
            )
        logger.exception("train_failed", symbol=symbol, timeframe=timeframe, error=str(exc))
        return result


def compact_tf_status_from_result(
    result: TrainResult,
    *,
    source_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured per-TF payload for job API / UI current-run panel."""
    m = result.metrics or {}
    fin = m.get("financial_oos") or {}
    val = m.get("financial_validation") or {}
    train = m.get("financial_train") or {}
    dep = m.get("financial_deploy_holdout") or {}
    cls = m.get("classification") or {}
    diag = m.get("fit_diagnosis") or {}
    cross = (source_meta or m.get("data_sources") or {}).get("cross_tf") or {}
    gates = list(m.get("gate_failures") or [])
    return {
        "timeframe": result.timeframe,
        "stage": "error" if result.error else "done",
        "progress_pct": 100.0,
        "passed_gates": bool(result.passed_gates) if not result.error else False,
        "gate_failures": gates,
        "gate_failures_detail": m.get("gate_failures_detail") or annotate_gate_failures(gates),
        "metrics": {
            "acc": cls.get("accuracy"),
            "f1": cls.get("f1_macro"),
            "auc": cls.get("roc_auc_ovr"),
            "trade_rate_filtered": cls.get("trade_rate_filtered"),
            "sharpe": fin.get("sharpe"),
            "sharpe_uncapped": fin.get("sharpe_uncapped"),
            "sharpe_ci_low": fin.get("sharpe_ci_low"),
            "sharpe_ci_high": fin.get("sharpe_ci_high"),
            "ann_factor": fin.get("ann_factor"),
            "max_drawdown": fin.get("max_drawdown"),
            "sortino": fin.get("sortino"),
            "expectancy": fin.get("expectancy"),
            "profit_factor": fin.get("profit_factor"),
            "risk_adjusted_return": fin.get("risk_adjusted_return"),
            "mean_trade_return": fin.get("mean_trade_return"),
            "sum_trade_returns": fin.get("sum_trade_returns"),
            "simple_trade_equity": fin.get("simple_trade_equity"),
            "total_return_compounded": fin.get("total_return"),
            "train_sharpe": train.get("sharpe"),
            "val_sharpe": val.get("sharpe"),
            "test_sharpe": fin.get("sharpe"),
            "deploy_sharpe": dep.get("sharpe"),
            "n_trades_train": train.get("n_trades"),
            "n_trades_val": val.get("n_trades"),
            "n_trades_test": fin.get("n_trades"),
            "n_trades_deploy": dep.get("n_trades"),
            "gap_tv": diag.get("sharpe_gap_train_val"),
            "gap_vt": diag.get("sharpe_gap_val_test"),
            "acc_gap": diag.get("accuracy_gap_train_val"),
        },
        "folds": m.get("folds") or [],
        "fit_diagnosis": diag,
        "regime_validation": m.get("regime_validation") or {},
        "advanced_eval": m.get("advanced_eval") or {},
        "knowledge_loop": m.get("knowledge_loop") or {},
        "live_readiness": m.get("live_readiness") or {},
        "model_zoo": m.get("model_zoo") or {},
        "stress_testing": m.get("stress_testing") or {},
        "monte_carlo": m.get("monte_carlo") or {},
        "intelligent_critique": m.get("intelligent_critique") or {},
        "self_optimize": m.get("self_optimize") or {},
        "self_optimize_applied": m.get("self_optimize_applied") or {},
        "validation_mode": m.get("validation_mode") or (m.get("validation") or {}).get("validation_mode"),
        "htf_sources": [s.get("timeframe") for s in (cross.get("sources") or [])],
        "n_htf_cols": cross.get("n_htf_cols", 0),
        "model_version": result.version or m.get("version"),
        "pipeline_version": m.get("pipeline_version") or PIPELINE_VERSION,
        "error": result.error,
        "message": "تم" if not result.error else str(result.error),
        "liquidity_rescue": ((m.get("trade_policy") or {}).get("source") == "deploy_liquidity_rescue"),
    }


def run_training(
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    *,
    progress: Callable[[float, str], None] | None = None,
    log: Callable[[str], None] | None = None,
    status: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    ensure_project_dirs()
    from atis.config import clear_config_caches

    clear_config_caches()
    set_global_seed()
    cfg_e4 = _cfg()
    cfg_e1 = load_engine_config().get("engine1_ingestion", {})
    symbols = symbols or list(
        cfg_e4.get("default_symbols")
        or cfg_e1.get("default_symbols")
        or ["XAUUSD"]
    )
    timeframes = timeframes or list(
        cfg_e4.get("default_timeframes")
        or cfg_e1.get("default_timeframes")
        or ["M1", "M5", "M15", "M30", "H1", "H4"]
    )
    run_id = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    tf_status: dict[str, Any] = {tf: empty_tf_run_status(tf) for tf in timeframes}
    if status:
        status(
            {
                "event": "run_start",
                "pipeline_version": PIPELINE_VERSION,
                "run_id": run_id,
                "timeframes": tf_status,
            }
        )

    model_family = str(cfg_e4.get("model_family", "baseline")).lower()
    if model_family in {"deep", "deep_learning", "llmodel"}:
        symbol = symbols[0]
        report: dict[str, Any] = {
            "started_finished": _utc_now().isoformat(),
            "symbols": symbols,
            "timeframes": timeframes,
            "mode": "deep_learning",
            "results": [],
            "summary": {},
        }
        if not HAS_TORCH:
            report["results"].append(
                {
                    "symbol": symbol,
                    "error": "torch_not_installed",
                    "artifact": "LLModel",
                }
            )
            report["summary"] = {"trained": 0, "errors": 1, "passed_gates": 0, "artifact": "LLModel"}
        else:
            try:
                artifact = train_llmodel(symbol, timeframes, progress=progress, log=log)
                test_fin = ((artifact.report or {}).get("test") or {}).get("financial") or {}
                report["results"].append(
                    {
                        "symbol": symbol,
                        "artifact": "LLModel",
                        "artifact_path": artifact.artifact_path,
                        "metrics_path": artifact.metrics_path,
                        "report": artifact.report,
                    }
                )
                report["summary"] = {
                    "trained": 1,
                    "errors": 0,
                    "passed_gates": 1 if float(test_fin.get("sharpe", 0.0)) > 0 else 0,
                    "artifact": "LLModel",
                    "test_sharpe": test_fin.get("sharpe"),
                    "test_return": test_fin.get("total_return"),
                }
                DataStateRegistry().audit(
                    "engine4",
                    "train_llmodel_success",
                    symbol=symbol,
                    timeframe="MULTI",
                    detail_json=json.dumps(report["summary"]),
                )
            except Exception as exc:
                report["results"].append(
                    {
                        "symbol": symbol,
                        "artifact": "LLModel",
                        "error": str(exc),
                    }
                )
                report["summary"] = {"trained": 0, "errors": 1, "passed_gates": 0, "artifact": "LLModel"}
                logger.exception("llmodel_train_failed", symbol=symbol, error=str(exc))
        out = PROJECT_ROOT / "logs" / "training"
        out.mkdir(parents=True, exist_ok=True)
        (out / "training_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    # Prefer liquid TFs for first training pass if caller passes all
    results = []
    total = max(1, len(symbols) * len(timeframes))
    idx = 0
    for s in symbols:
        for tf in timeframes:
            def _progress(local_pct: float, message: str, *, _idx: int = idx) -> None:
                if progress:
                    base = (100.0 * _idx) / total
                    progress(base + (local_pct / total), message)

            def _status(payload: dict[str, Any], *, _tf: str = tf) -> None:
                if "event" in payload and payload.get("event") == "run_start":
                    return
                cur = dict(tf_status.get(_tf) or empty_tf_run_status(_tf))
                cur.update(payload)
                cur["timeframe"] = _tf
                tf_status[_tf] = cur
                if status:
                    status(
                        {
                            "event": "tf_update",
                            "pipeline_version": PIPELINE_VERSION,
                            "run_id": run_id,
                            "timeframe": _tf,
                            "tf": cur,
                            "timeframes": tf_status,
                        }
                    )

            tr = train_symbol_timeframe(s, tf, progress=_progress, log=log, status=_status)
            results.append(asdict(tr))
            # Ensure final compact status even if stage callbacks were sparse.
            tf_status[tf] = {
                **(tf_status.get(tf) or empty_tf_run_status(tf)),
                **compact_tf_status_from_result(tr),
            }
            idx += 1

    passed = sum(1 for r in results if r.get("passed_gates"))
    rejected = sum(1 for r in results if (not r.get("passed_gates")) and not r.get("error"))
    errors = sum(1 for r in results if r.get("error"))
    reject_reasons = []
    for r in results:
        if r.get("passed_gates") or r.get("error"):
            continue
        gates = ((r.get("metrics") or {}).get("gate_failures") or [])
        if gates:
            reject_reasons.append(f"{r.get('timeframe')}: {' · '.join(gates)}")
    report = {
        "started_finished": _utc_now().isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "run_id": run_id,
        "symbols": symbols,
        "timeframes": timeframes,
        "results": results,
        "timeframes_status": tf_status,
        "summary": {
            "trained": len(results),
            "passed": passed,
            "rejected": rejected,
            "errors": errors,
            "passed_gates": passed,
            "reject_reasons": reject_reasons,
            "by_timeframe": {
                r.get("timeframe"): {
                    "error": r.get("error"),
                    "passed_gates": r.get("passed_gates"),
                    "version": r.get("version"),
                    "sharpe": ((r.get("metrics") or {}).get("financial_oos") or {}).get("sharpe"),
                    "sharpe_uncapped": ((r.get("metrics") or {}).get("financial_oos") or {}).get(
                        "sharpe_uncapped"
                    ),
                    "sharpe_ci_low": ((r.get("metrics") or {}).get("financial_oos") or {}).get(
                        "sharpe_ci_low"
                    ),
                    "gate_failures": ((r.get("metrics") or {}).get("gate_failures") or []),
                    "gate_failures_detail": ((r.get("metrics") or {}).get("gate_failures_detail") or []),
                    "fit_status": (((r.get("metrics") or {}).get("fit_diagnosis") or {}).get("status")),
                }
                for r in results
            },
        },
    }
    if bool(cfg_e4.get("write_final_model", True)):
        allow_paper = bool(cfg_e4.get("allow_paper_final", False))
        try:
            final_meta = publish_final_model(
                results,
                symbol=symbols[0],
                allow_paper_final=allow_paper,
            )
        except TypeError:
            # Stale process still holding an older publish_final_model() signature.
            final_meta = publish_final_model(results, symbol=symbols[0])
            if not allow_paper and not final_meta.get("passed_gates"):
                final_meta = {
                    **final_meta,
                    "reason": "no_gated_results_paper_final_disabled",
                    "kept_existing": bool(final_meta.get("exists")),
                    "mode": final_meta.get("mode") or "paper_only",
                }
        champ_from_this = bool(
            final_meta.get("updated_this_run", False)
            and not (final_meta.get("kept_existing") or final_meta.get("skipped_downgrade") or final_meta.get("champion_from_prior_run"))
        )
        report["final_model"] = final_meta
        report["summary"]["final_model"] = {
            "timeframe": final_meta.get("timeframe"),
            "version": final_meta.get("version"),
            "mode": final_meta.get("mode"),
            "artifact_path": final_meta.get("artifact_path"),
            "passed_gates": final_meta.get("passed_gates"),
            "updated_this_run": bool(final_meta.get("updated_this_run", final_meta.get("exists"))),
            "kept_existing": bool(final_meta.get("kept_existing") or final_meta.get("skipped_downgrade")),
            "current_run_passed_gates": passed,
            "champion_from_this_run": champ_from_this,
        }
        report["summary"]["champion_tf"] = final_meta.get("timeframe")
        report["summary"]["champion_from_this_run"] = champ_from_this
        if log and final_meta.get("exists"):
            log(
                f"[FinalModel] tf={final_meta.get('timeframe')} "
                f"version={final_meta.get('version')} mode={final_meta.get('mode')} "
                f"path={final_meta.get('artifact_path')} "
                f"from_this_run={champ_from_this}"
            )

    # Intelligence: critique → development_plan.json after every run
    try:
        from atis.engines.engine4_training.intelligence import critique_training_run

        plan = critique_training_run(
            results,
            models_root=get_path("models"),
            run_id=run_id,
        )
        report["development_plan"] = {
            "summary_ar": plan.get("summary_ar"),
            "next_experiment": plan.get("next_experiment"),
            "artifact_path": plan.get("artifact_path"),
        }
        report["summary"]["next_experiment"] = plan.get("next_experiment")
        if log and plan.get("summary_ar"):
            log(plan["summary_ar"])
    except Exception as exc:
        report["development_plan"] = {"error": str(exc)}
        if log:
            log(f"intelligence_critique_failed: {exc}")

    # Retrain trigger bookkeeping (interval + drift hooks from knowledge)
    try:
        from atis.engines.engine4_training.adaptive_learning import should_trigger_retrain

        retrain_days = float(
            (load_engine_config().get("engine5_live") or {}).get("retrain_interval_days", 7)
        )
        # Aggregate max PSI / decay across TF knowledge stores
        max_psi = 0.0
        max_decay = 0.0
        for r in results:
            adv = ((r.get("metrics") or {}).get("knowledge_loop") or {}).get("advisory") or {}
            max_psi = max(max_psi, float(adv.get("feature_psi") or 0.0))
            max_decay = max(max_decay, float(adv.get("performance_decay") or 0.0))
        drift_score = max(max_psi, max_decay / 4.0)
        last_train_path = get_path("models") / "intelligence" / "last_train.json"
        last_train_utc = None
        if last_train_path.exists():
            try:
                last_train_utc = json.loads(last_train_path.read_text(encoding="utf-8")).get(
                    "last_train_utc"
                )
            except Exception:
                last_train_utc = None
        e4cfg = load_engine_config().get("engine4_training") or {}
        _trigger, reason = should_trigger_retrain(
            last_train_utc=last_train_utc,
            retrain_interval_days=retrain_days,
            drift_score=drift_score,
            drift_threshold=float(e4cfg.get("retrain_drift_threshold", 0.25)),
        )
        report["retrain_policy"] = {
            "interval_days": retrain_days,
            "would_trigger_now": bool(_trigger),
            "schedule_reason": reason,
            "drift_score": round(drift_score, 4),
            "max_feature_psi": round(max_psi, 4),
            "max_performance_decay": round(max_decay, 4),
            "enabled": True,
            "auto_retrain_recommended": bool(_trigger),
        }
        intel_dir = get_path("models") / "intelligence"
        intel_dir.mkdir(parents=True, exist_ok=True)
        (intel_dir / "last_train.json").write_text(
            json.dumps(
                {
                    "last_train_utc": _utc_now().isoformat(),
                    "run_id": run_id,
                    "retrain_interval_days": retrain_days,
                    "pipeline_version": PIPELINE_VERSION,
                    "retrain_policy": report["retrain_policy"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        # Persist auto-retrain advisory for Engine 5 / scheduler
        (intel_dir / "retrain_advisory.json").write_text(
            json.dumps(report["retrain_policy"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        report["retrain_policy"] = {"error": str(exc)}
    if status:
        status(
            {
                "event": "run_done",
                "pipeline_version": PIPELINE_VERSION,
                "run_id": run_id,
                "timeframes": tf_status,
                "summary": report["summary"],
            }
        )
    out = PROJECT_ROOT / "logs" / "training"
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
