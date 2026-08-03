"""Fold stability, crisis/recent holdouts, and promotion helpers (v16+).

Phase A: Consistency Score across folds/regimes + regime-balanced holdouts.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _cv(values: list[float]) -> float | None:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) < 2:
        return None
    mu = float(np.mean(arr))
    sig = float(np.std(arr, ddof=0))
    if abs(mu) < 1e-9:
        return float(sig) if sig > 0 else 0.0
    return float(sig / abs(mu))


def consistency_score_report(
    *,
    fold_metrics: list[dict[str, Any]] | None = None,
    regime_validation: dict[str, Any] | None = None,
    financial_oos: dict[str, Any] | None = None,
    classification: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """0–100 Consistency Score: low dispersion of WR / F1 / expectancy across folds & regimes.

    Soft diagnostic by default; readiness may blend / cap live_ready when weak.
    """
    cfg = cfg or {}
    fold_metrics = list(fold_metrics or [])
    regime_validation = regime_validation or {}
    fin = financial_oos or {}
    cls = classification or {}

    fold_wr = [float(r.get("win_rate", (r.get("policy") or {}).get("win_rate", np.nan)) or np.nan) for r in fold_metrics]
    fold_f1 = [float(r.get("f1_macro", r.get("f1", np.nan)) or np.nan) for r in fold_metrics]
    fold_exp = []
    for r in fold_metrics:
        pol = r.get("policy") or {}
        v = r.get("expectancy", pol.get("expectancy", pol.get("val_expectancy")))
        if v is not None:
            fold_exp.append(float(v))
    # Prefer test_sharpe dispersion as expectancy proxy when fold expectancy absent
    fold_sh = [float(r.get("test_sharpe", (r.get("policy") or {}).get("val_sharpe", np.nan)) or np.nan) for r in fold_metrics]

    regimes = regime_validation.get("regimes") or {}
    reg_wr: list[float] = []
    reg_exp: list[float] = []
    reg_f1: list[float] = []
    n_regimes_ok = 0
    for _name, row in regimes.items():
        if not isinstance(row, dict) or row.get("skipped"):
            continue
        n_regimes_ok += 1
        if row.get("win_rate") is not None:
            reg_wr.append(float(row["win_rate"]))
        if row.get("expectancy") is not None:
            reg_exp.append(float(row["expectancy"]))
        if row.get("f1_macro") is not None:
            reg_f1.append(float(row["f1_macro"]))
        elif row.get("f1") is not None:
            reg_f1.append(float(row["f1"]))

    cv_wr = _cv([v for v in fold_wr if np.isfinite(v)])
    cv_f1 = _cv([v for v in fold_f1 if np.isfinite(v)])
    cv_exp = _cv(fold_exp) if len(fold_exp) >= 2 else _cv([v for v in fold_sh if np.isfinite(v)])
    cv_reg_wr = _cv(reg_wr)
    cv_reg_exp = _cv(reg_exp)

    # Score: start 100, subtract for high CV (dispersion)
    score = 100.0
    deductions: list[str] = []
    notes: list[str] = []

    def _pen(cv: float | None, thr: float, label: str, weight: float) -> None:
        nonlocal score
        if cv is None:
            return
        if cv > thr * 2:
            score -= weight
            deductions.append(f"{label}_severe")
        elif cv > thr:
            score -= weight * 0.55
            deductions.append(label)

    _pen(cv_wr, float(cfg.get("consistency_max_cv_win_rate", 0.25)), "fold_wr_dispersion", 18.0)
    _pen(cv_f1, float(cfg.get("consistency_max_cv_f1", 0.30)), "fold_f1_dispersion", 14.0)
    _pen(cv_exp, float(cfg.get("consistency_max_cv_expectancy", 0.85)), "fold_exp_dispersion", 16.0)
    _pen(cv_reg_wr, float(cfg.get("consistency_max_cv_regime_wr", 0.35)), "regime_wr_dispersion", 14.0)
    _pen(cv_reg_exp, float(cfg.get("consistency_max_cv_regime_exp", 1.20)), "regime_exp_dispersion", 12.0)

    if n_regimes_ok < int(cfg.get("consistency_min_regimes", 2)):
        score -= 8
        deductions.append("insufficient_regime_coverage")
        notes.append("need_more_regime_slices")

    if regime_validation.get("stable") is False:
        score -= 10
        deductions.append("regime_unstable_flag")

    # Mild bonus when OOS WR/F1 present and folds agree on positive expectancy
    oos_wr = float(fin.get("win_rate", 0) or 0)
    oos_f1 = float(cls.get("f1_macro", 0) or 0)
    if oos_wr >= 0.52 and oos_f1 >= 0.40 and not deductions:
        score = min(100.0, score + 4)
        notes.append("oos_quality_aligned")

    score = max(0.0, min(100.0, score))
    min_ok = float(cfg.get("min_consistency_score", 55.0))
    return {
        "enabled": True,
        "score": round(score, 1),
        "gate_pass": score >= min_ok,
        "min_score": min_ok,
        "deductions": deductions[:12],
        "notes": notes[:8],
        "dispersion": {
            "cv_fold_win_rate": None if cv_wr is None else round(cv_wr, 4),
            "cv_fold_f1": None if cv_f1 is None else round(cv_f1, 4),
            "cv_fold_expectancy": None if cv_exp is None else round(cv_exp, 4),
            "cv_regime_win_rate": None if cv_reg_wr is None else round(cv_reg_wr, 4),
            "cv_regime_expectancy": None if cv_reg_exp is None else round(cv_reg_exp, 4),
        },
        "n_folds": len(fold_metrics),
        "n_regimes_ok": n_regimes_ok,
        "summary_ar": (
            f"اتساق الأداء: {score:.0f}/100 · أنظمة={n_regimes_ok} · "
            f"{'مقبول' if score >= min_ok else 'ضعيف'}"
            + (f" · خصم={','.join(deductions[:3])}" if deductions else "")
        ),
        "summary_en": (
            f"Consistency {score:.0f}/100 across folds/regimes"
            + (f"; deductions={','.join(deductions[:3])}" if deductions else "")
        ),
    }


def regime_balanced_holdout_report(
    regime_validation: dict[str, Any] | None,
    *,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require positive expectancy (or non-collapse) across enough market regimes.

    Activated when ``regime_balanced_holdouts`` is true (diagnosis / config).
    """
    cfg = cfg or {}
    enabled = bool(cfg.get("regime_balanced_holdouts", False))
    regimes = (regime_validation or {}).get("regimes") or {}
    need = int(cfg.get("regime_balanced_min_regimes", 3))
    min_wr = float(cfg.get("regime_balanced_min_win_rate", 0.48))
    min_exp = float(cfg.get("regime_balanced_min_expectancy", -1e-6))
    ok_names: list[str] = []
    weak_names: list[str] = []
    for name, row in regimes.items():
        if not isinstance(row, dict) or row.get("skipped"):
            continue
        wr = float(row.get("win_rate", 0) or 0)
        exp = float(row.get("expectancy", 0) or 0)
        if wr >= min_wr and exp >= min_exp:
            ok_names.append(str(name))
        else:
            weak_names.append(str(name))
    n_ok = len(ok_names)
    gate_pass = (not enabled) or (n_ok >= need)
    return {
        "enabled": enabled,
        "n_ok": n_ok,
        "need": need,
        "ok_regimes": ok_names,
        "weak_regimes": weak_names,
        "gate_pass": gate_pass,
        "summary_ar": (
            f"Holdouts متوازنة بالأنظمة: {n_ok}/{need} · "
            f"{'اجتاز' if gate_pass else 'فشل'}"
            + (f" · ضعيف={','.join(weak_names[:3])}" if weak_names else "")
        ),
    }


def fold_stability_report(
    fold_metrics: list[dict[str, Any]],
    *,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require consistent Val Sharpe across liquid folds (not one lucky fold)."""
    cfg = cfg or {}
    sharpes: list[float] = []
    rates: list[float] = []
    liquid = 0
    min_val = float(cfg.get("fold_stability_min_val_trades", 10))
    for row in fold_metrics:
        n_vt = float(row.get("n_val_trades", (row.get("policy") or {}).get("val_trades", 0)) or 0)
        sh = float((row.get("policy") or {}).get("val_sharpe", row.get("val_sharpe", 0)) or 0)
        rate = float(row.get("trade_rate", 0) or 0)
        if n_vt >= min_val:
            liquid += 1
            sharpes.append(sh)
            rates.append(rate)
    if len(sharpes) < 2:
        return {
            "enabled": True,
            "stable": False,
            "reason": "insufficient_liquid_folds",
            "n_liquid": liquid,
            "early_folds_weak": False,
            "gate_pass": not bool(cfg.get("fail_on_fold_unstable", True)),
        }
    arr = np.asarray(sharpes, dtype=float)
    med = float(np.median(arr))
    iqr = float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25))
    frac_positive = float(np.mean(arr > 0))
    max_iqr = float(cfg.get("fold_stability_max_iqr", 4.0))
    min_pos_frac = float(cfg.get("fold_stability_min_pos_frac", 0.6))
    unstable = (iqr > max_iqr and med < 3.0) or (frac_positive < min_pos_frac)

    # Peg detection uses the effective trade-rate cap (caller should resolve by_tf).
    # Do NOT floor at 0.08 — that hides saturation when cap is 0.05–0.07 (H1 desaturate).
    rate_cap = float(cfg.get("max_fold_trade_rate", 0.15) or 0.15)
    by_tf_cap = cfg.get("max_fold_trade_rate_by_tf") or {}
    tf_hint = str(cfg.get("_active_timeframe") or cfg.get("timeframe") or "").upper()
    if tf_hint and tf_hint in by_tf_cap:
        rate_cap = float(by_tf_cap[tf_hint])
    peg_thr = max(0.02, float(rate_cap) * 0.90) if rate_cap > 0 else 0.08
    rate_pegged = bool(rates) and float(np.mean(np.asarray(rates) >= peg_thr)) >= 0.8

    # Early-fold weakness (M15 CPCV pattern: first paths near chance / negative OOS).
    early_frac = float(cfg.get("early_fold_frac", 0.40))
    n_early = max(1, int(np.ceil(len(fold_metrics) * early_frac)))
    early = list(fold_metrics[:n_early])
    early_acc = [float(r.get("accuracy", 0) or 0) for r in early]
    early_auc = [float(r.get("roc_auc_ovr", 0) or 0) for r in early]
    early_test = [float(r.get("test_sharpe", 0) or 0) for r in early]
    mean_early_acc = float(np.mean(early_acc)) if early_acc else 1.0
    mean_early_auc = float(np.mean(early_auc)) if early_auc else 1.0
    mean_early_test = float(np.mean(early_test)) if early_test else 0.0
    frac_early_neg = float(np.mean(np.asarray(early_test) < 0)) if early_test else 0.0
    min_early_acc = float(cfg.get("early_fold_min_acc", 0.58))
    min_early_auc = float(cfg.get("early_fold_min_auc", 0.60))
    min_early_test = float(cfg.get("early_fold_min_mean_test_sharpe", 0.0))
    max_early_neg_frac = float(cfg.get("early_fold_max_neg_frac", 0.34))
    # Acc alone is noisy with high flat share; allow AUC rescue when signal is present
    # (M15 early Acc≈0.55 with AUC≈0.74 was a false "no edge" smell).
    early_weak = (
        (mean_early_acc < min_early_acc and mean_early_auc < min_early_auc)
        or mean_early_test < min_early_test
        or frac_early_neg > max_early_neg_frac
    )

    gate_pass = (not unstable) if bool(cfg.get("fail_on_fold_unstable", True)) else True
    return {
        "enabled": True,
        "stable": not unstable,
        "n_liquid": liquid,
        "median_val_sharpe": round(med, 4),
        "iqr_val_sharpe": round(iqr, 4),
        "frac_positive_folds": round(frac_positive, 4),
        "trade_rate_pegged": rate_pegged,
        "trade_rate_peg_threshold": round(peg_thr, 4),
        "early_folds_weak": bool(early_weak),
        "early_fold_stats": {
            "n_early": n_early,
            "mean_accuracy": round(mean_early_acc, 4),
            "mean_auc": round(mean_early_auc, 4),
            "mean_test_sharpe": round(mean_early_test, 4),
            "frac_negative_test": round(frac_early_neg, 4),
            "auc_rescue": bool(mean_early_acc < min_early_acc and mean_early_auc >= min_early_auc),
        },
        "gate_pass": gate_pass,
        "summary_ar": (
            f"ثبات الطيات: {'مستقر' if not unstable else 'غير مستقر'} · "
            f"median={med:.2f} · IQR={iqr:.2f} · +folds={frac_positive:.0%}"
            + (f" · طيات مبكرة ضعيفة" if early_weak else "")
        ),
    }


def inflated_sharpe_report(
    fin: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
    auc: float | None = None,
) -> dict[str, Any]:
    """Flag annualization / path inflation vs trade-level reality.

    Uncapped / bar-path Sharpe is diagnostic-only for live ranking.
    Primary honesty signals: trade_sharpe_raw, uncapped ceiling, path-vs-trade gap,
    and Sharpe↔AUC mismatch (filter-driven inflation).

    M5 (20260803): uncapped≈22 with trade_raw≈0.71 and Val≈Test≈Deploy must NOT
    hard-fail solely on uncapped_above_max — that rejects honest HF edges.
    """
    cfg = cfg or {}
    capped = float(fin.get("sharpe", 0) or 0)
    uncapped = float(fin.get("sharpe_uncapped", 0) or 0)
    trade_raw = float(fin.get("trade_sharpe_raw", 0) or 0)
    max_unc = float(cfg.get("max_sharpe_uncapped", 12.0))
    max_ratio = float(cfg.get("max_uncapped_to_capped_ratio", 2.5))
    min_trade = float(cfg.get("min_trade_sharpe_raw", -0.05))
    enforce_min_trade = bool(cfg.get("enforce_min_trade_sharpe_raw", True))
    # Trade-supported rescue: strong per-trade Sharpe excuses HF uncapped path blow-up.
    # M1 often has ratio≈3 solely from √(trades/year) before the daily cap — not dishonest
    # when trade_raw is strong (this run: trade_raw=0.78, ratio=3.07).
    rescue_min_trade = float(cfg.get("uncapped_rescue_min_trade_sharpe", 0.35))
    max_ratio_rescue = float(cfg.get("max_uncapped_ratio_with_trade_rescue", 4.0))
    ratio = uncapped / max(abs(capped), 0.25)
    reasons: list[str] = []
    notes: list[str] = []
    inflated = False
    trade_supported = trade_raw >= rescue_min_trade
    if uncapped > max_unc:
        if trade_supported and ratio <= max_ratio_rescue:
            notes.append("uncapped_high_but_trade_supported")
        else:
            inflated = True
            reasons.append("uncapped_above_max")
    if capped > 0.5 and ratio > max_ratio:
        if trade_supported and ratio <= max_ratio_rescue:
            notes.append("ratio_high_but_trade_supported")
        else:
            inflated = True
            reasons.append("uncapped_to_capped_ratio")
    # Path Sharpe huge but trade Sharpe weak/near-zero
    path_vs_trade = capped > 3.0 and trade_raw < float(cfg.get("max_path_vs_trade_gap", 0.35))
    if path_vs_trade:
        inflated = True
        reasons.append("path_vs_trade_gap")
    # Enforce unused min_trade_sharpe_raw when path Sharpe looks strong
    weak_trade = False
    if enforce_min_trade and capped > 2.0 and trade_raw < min_trade:
        inflated = True
        weak_trade = True
        reasons.append("trade_sharpe_below_min")
    # High path Sharpe with near-chance AUC = filter/metric inflation smell
    auc_mismatch = False
    if auc is not None and capped >= 4.0 and float(auc) < 0.52:
        inflated = True
        auc_mismatch = True
        reasons.append("sharpe_auc_mismatch")
    rescued = bool(trade_supported and (uncapped > max_unc or ratio > max_ratio) and not inflated)
    return {
        "inflated": bool(inflated),
        "sharpe": round(capped, 4),
        "sharpe_uncapped": round(uncapped, 4),
        "trade_sharpe_raw": round(trade_raw, 4),
        "uncapped_ratio": round(ratio, 4),
        "path_vs_trade_gap": bool(path_vs_trade),
        "weak_trade_sharpe": bool(weak_trade),
        "sharpe_auc_mismatch": bool(auc_mismatch),
        "trade_supported_uncapped": rescued,
        "reasons": reasons,
        "notes": notes,
        "diagnostic_only_uncapped": True,
        "gate_pass": (not inflated) if bool(cfg.get("fail_on_inflated_sharpe", True)) else True,
        "summary_ar": (
            f"تضخيم Sharpe: {'نعم' if inflated else 'لا'} · "
            f"capped={capped:.2f} · uncapped={uncapped:.2f} · trade_raw={trade_raw:.3f}"
            + (f" · أسباب={','.join(reasons)}" if reasons else "")
            + (f" · ملاحظات={','.join(notes)}" if notes else "")
        ),
    }


def crisis_recent_holdout_slices(
    n: int,
    *,
    recent_frac: float = 0.12,
    crisis_frac: float = 0.15,
) -> dict[str, np.ndarray]:
    """Index slices for recent-window and mid-sample 'crisis-like' stress holdout.

    Crisis proxy: middle band of the series (often contains regime transitions in
    multi-year gold samples). Recent: last fraction of bars.
    """
    n = int(n)
    recent_n = max(30, int(n * float(recent_frac)))
    crisis_n = max(30, int(n * float(crisis_frac)))
    recent = np.arange(max(0, n - recent_n), n)
    mid = n // 2
    half = crisis_n // 2
    lo = max(0, mid - half)
    hi = min(n, lo + crisis_n)
    crisis = np.arange(lo, hi)
    return {"recent": recent, "crisis": crisis}


def evaluate_holdout_slice(
    returns: np.ndarray,
    idx: np.ndarray,
    *,
    financial_fn,
    name: str,
) -> dict[str, Any]:
    if len(idx) < 10:
        return {"name": name, "skipped": True, "reason": "too_short"}
    sub = np.asarray(returns)[idx]
    fin = financial_fn(sub)
    return {
        "name": name,
        "skipped": False,
        "n_bars": int(len(idx)),
        "sharpe": fin.get("sharpe"),
        "expectancy": fin.get("expectancy"),
        "max_drawdown": fin.get("max_drawdown"),
        "n_trades": fin.get("n_trades"),
        "total_return": fin.get("total_return"),
    }


def confidence_position_size(
    confidence: float,
    *,
    atr_pct: float,
    base_size: float = 1.0,
    max_size: float = 1.5,
    min_size: float = 0.25,
    target_atr: float = 0.002,
) -> float:
    """Vol-targeted size scaled by model confidence (Kelly-lite, capped)."""
    conf = float(np.clip(confidence, 0.0, 1.0))
    vol_scale = float(target_atr) / max(float(atr_pct), 1e-6)
    vol_scale = float(np.clip(vol_scale, 0.4, 1.6))
    # Confidence above 0.55 scales up gently
    conf_scale = 0.6 + 0.8 * max(0.0, conf - 0.50) / 0.50
    size = float(base_size) * vol_scale * conf_scale
    return float(np.clip(size, min_size, max_size))
