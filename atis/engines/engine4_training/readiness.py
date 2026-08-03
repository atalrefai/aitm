"""Live-trading readiness score (0–100) with explicit gate evidence.

Never awards live_ready / 100 when material honesty, saturation, early-fold,
or Train/Val/Test gap weaknesses are present.
"""

from __future__ import annotations

from typing import Any


def compute_live_readiness(
    *,
    passed_gates: bool,
    metrics: dict[str, Any],
    timeframe: str = "",
) -> dict[str, Any]:
    fin = metrics.get("financial_oos") or {}
    cls = metrics.get("classification") or {}
    fit = metrics.get("fit_diagnosis") or {}
    regime = metrics.get("regime_validation") or {}
    adv = metrics.get("advanced_eval") or {}
    stress = metrics.get("stress_testing") or {}
    mc = metrics.get("monte_carlo") or {}
    sessions = metrics.get("session_validation") or {}
    zoo = metrics.get("model_zoo") or {}
    data_intel = metrics.get("data_intelligence") or {}
    diagnosis = metrics.get("self_diagnosis") or {}

    score = 0.0
    reasons: list[str] = []
    deductions: list[str] = []
    material_warnings: list[str] = []

    if passed_gates:
        score += 25
        reasons.append("passed_promotion_gates")
    else:
        deductions.append("failed_promotion_gates")

    sharpe = float(fin.get("sharpe", 0) or 0)
    ci = float(fin.get("sharpe_ci_low", 0) or 0)
    exp = float(fin.get("expectancy", 0) or 0)
    auc = float(cls.get("roc_auc_ovr", 0.5) or 0.5)
    n_tr = float(fin.get("n_trades", 0) or 0)
    trade_raw = float(fin.get("trade_sharpe_raw", 0) or 0)

    # Prefer trade-level Sharpe for scoring when available (anti-inflation).
    scoring_sharpe = sharpe
    if n_tr >= 20 and trade_raw > 0:
        # Soft blend — never let uncapped path dominate readiness
        scoring_sharpe = min(sharpe, max(trade_raw * 8.0, trade_raw))

    if scoring_sharpe >= 1.0:
        score += 12
        reasons.append("oos_sharpe>=1")
    elif scoring_sharpe >= 0.25:
        score += 6
    else:
        deductions.append("weak_oos_sharpe")

    if ci >= 0.0:
        score += 8
        reasons.append("sharpe_ci_low_nonneg")
    else:
        deductions.append("negative_sharpe_ci")

    if exp > 0:
        score += 8
        reasons.append("positive_expectancy")
    else:
        deductions.append("nonpositive_expectancy")

    if auc >= 0.55:
        score += 8
        reasons.append("auc_edge")
    elif auc < 0.52:
        deductions.append("near_chance_auc")
        score -= 8
        material_warnings.append("near_chance_auc")

    if n_tr >= 40:
        score += 6
    elif n_tr < 15:
        deductions.append("sparse_oos_trades")
        score -= 6

    fit_st = str(fit.get("status") or "")
    if fit_st == "balanced":
        score += 10
        reasons.append("fit_balanced")
    elif fit_st == "overfitting":
        deductions.append("overfitting_flag")
        score -= 10
        material_warnings.append("overfitting")
    elif fit_st == "unstable_generalization":
        deductions.append("unstable_generalization")
        score -= 12
        material_warnings.append("unstable_generalization")

    gap_tv = abs(float(fit.get("sharpe_gap_train_val", 0) or 0))
    gap_vt = abs(float(fit.get("sharpe_gap_val_test", 0) or 0))
    if gap_tv > 2.0:
        deductions.append("train_val_sharpe_gap")
        score -= 10
        material_warnings.append("train_val_gap")
    if gap_vt > 3.0:
        deductions.append("val_test_sharpe_gap")
        score -= 10
        material_warnings.append("val_test_gap")
    elif gap_vt > 2.0:
        deductions.append("val_test_sharpe_gap_warn")
        score -= 5
        material_warnings.append("val_test_gap_warn")

    if regime.get("stable") is True:
        score += 6
        reasons.append("regime_stable")
    elif regime.get("stable") is False:
        deductions.append("regime_unstable")
        score -= 6

    dsr = float((adv.get("deflated_sharpe") or {}).get("deflated_sharpe", 0) or 0)
    pbo = float((adv.get("pbo") or {}).get("pbo", 0.5) or 0.5)
    uncapped = float(fin.get("sharpe_uncapped", 0) or 0)
    # Do not bonus DSR when it is non-discriminative (≈1 with absurd uncapped)
    if dsr >= 0.7 and uncapped <= 15:
        score += 5
    elif dsr >= 0.99 and uncapped > 15:
        deductions.append("dsr_non_discriminative")
        score -= 6
        material_warnings.append("dsr_non_discriminative")
    pbo_rep = adv.get("pbo") or {}
    pbo_material = float(pbo_rep.get("material", 0.0) or 0.0) > 0.5
    pbo_soft = float(pbo_rep.get("soft_warn", 0.0) or 0.0) > 0.5
    if pbo <= 0.45:
        score += 4
    elif pbo >= 0.55 and pbo_material:
        deductions.append("elevated_pbo")
        score -= 4
        material_warnings.append("elevated_pbo")
    elif pbo >= 0.55 and pbo_soft:
        # Rank-flip only (OOS retention healthy) — soft warn, not readiness hit
        deductions.append("pbo_rank_instability_soft")
        reasons.append("pbo_soft_warn_retained_oos")

    # Anti-inflation / trade realism — material when present
    infl = fin.get("sharpe_inflation") or metrics.get("sharpe_inflation") or {}
    if infl.get("inflated"):
        deductions.append("inflated_sharpe")
        score -= 15
        material_warnings.append("inflated_sharpe")
    elif uncapped > 15:
        deductions.append("high_uncapped_sharpe")
        score -= 8
        material_warnings.append("high_uncapped_sharpe")
    if trade_raw >= 0.4:
        score += 4
        reasons.append("healthy_trade_sharpe")
    elif n_tr >= 40 and trade_raw < 0.15 and sharpe > 4:
        deductions.append("path_sharpe_vs_trade_gap")
        score -= 8
        material_warnings.append("path_vs_trade")

    fold_st = metrics.get("fold_stability") or {}
    if fold_st.get("early_folds_weak"):
        deductions.append("early_folds_weak")
        score -= 12
        material_warnings.append("early_folds_weak")
    if fold_st.get("trade_rate_pegged"):
        deductions.append("trade_rate_saturated")
        score -= 12
        material_warnings.append("trade_rate_saturated")
    if fold_st.get("stable") is False:
        deductions.append("fold_unstable")
        score -= 8
        material_warnings.append("fold_unstable")

    # Self-diagnosis scores (if present) — soft blend / hard cap
    if diagnosis:
        honesty = float(diagnosis.get("metric_honesty_score", 100) or 100)
        gen = float(diagnosis.get("generalization_score", 100) or 100)
        if honesty < 60:
            deductions.append("low_metric_honesty")
            score -= 10
            material_warnings.append("low_metric_honesty")
        if gen < 60:
            deductions.append("low_generalization")
            score -= 8
            material_warnings.append("low_generalization")

    # Consistency Score (Phase A) — soft penalty; weak consistency blocks live_ready
    consistency = metrics.get("consistency") or {}
    if consistency.get("enabled"):
        cscore = float(consistency.get("score", 100) or 100)
        if cscore >= 75:
            score += 4
            reasons.append("consistency_strong")
        elif cscore < 55:
            deductions.append("low_consistency")
            score -= 8
            material_warnings.append("low_consistency")
        elif cscore < 65:
            deductions.append("consistency_warn")
            score -= 4

    regime_bal = metrics.get("regime_balanced_holdouts") or {}
    if regime_bal.get("enabled") and regime_bal.get("gate_pass") is False:
        deductions.append("regime_balanced_holdouts_weak")
        score -= 8
        material_warnings.append("regime_balanced_holdouts_weak")

    feat_x = metrics.get("feature_explainability") or {}
    shap = feat_x.get("shap") or {}
    if shap.get("enabled"):
        score += 3
        reasons.append(f"shap={shap.get('method', 'ok')}")
    elif bool((metrics.get("cfg_hint") or {}).get("shap_enabled", True)) is False:
        pass
    else:
        if feat_x.get("enabled") and not shap.get("enabled"):
            deductions.append("shap_missing")
            score -= 3

    if stress.get("robust"):
        # Do not fully trust stress when inflation smells dominate
        if "inflated_sharpe" in material_warnings or "dsr_non_discriminative" in material_warnings:
            score += 2
            reasons.append("stress_robust_discounted")
        else:
            score += 5
            reasons.append("stress_robust")
    elif stress.get("scenarios"):
        if float(stress.get("worst_sharpe", 0) or 0) < -1.0:
            deductions.append("stress_fragile")
            score -= 5

    if mc.get("stable"):
        if float(mc.get("p_profit", 0) or 0) >= 0.95 and uncapped > 15:
            deductions.append("mc_non_discriminative")
            score -= 4
            material_warnings.append("mc_non_discriminative")
        else:
            score += 5
            reasons.append("monte_carlo_stable")
    elif mc.get("enabled") and float(mc.get("p_profit", 0) or 0) < 0.5:
        deductions.append("mc_low_profit_prob")
        score -= 4

    sess = sessions.get("sessions") or {}
    pos_sess = sum(1 for v in sess.values() if not v.get("skipped") and float(v.get("sharpe", 0) or 0) > 0)
    if pos_sess >= 2:
        score += 4
        reasons.append("multi_session_edge")

    if data_intel.get("ready"):
        score += 3
    if zoo.get("enabled") and zoo.get("winner"):
        score += 2
        reasons.append(f"zoo_winner={zoo.get('winner')}")

    tf = str(timeframe).upper()
    if tf == "H4" and auc < 0.55:
        score -= 15
        deductions.append("h4_near_chance_block")
        material_warnings.append("h4_near_chance")
    if tf == "H1" and fit_st == "overfitting":
        score -= 5
        deductions.append("h1_overfit_penalty")
    if tf == "M15" and fold_st.get("early_folds_weak"):
        score -= 4
        deductions.append("m15_early_path_penalty")

    # Hard ceiling: never 100 with material warnings; never live_ready if smells
    score = max(0.0, min(100.0, score))
    if material_warnings:
        score = min(score, 88.0)

    inflated_block = bool(infl.get("inflated")) or "inflated_sharpe" in material_warnings
    saturated_block = bool(fold_st.get("trade_rate_pegged"))
    early_block = bool(fold_st.get("early_folds_weak"))
    gap_block = "train_val_gap" in material_warnings or "val_test_gap" in material_warnings

    if (
        score >= 75
        and passed_gates
        and fit_st != "unstable_generalization"
        and not inflated_block
        and not saturated_block
        and not early_block
        and not gap_block
        and not material_warnings
    ):
        verdict = "live_ready"
        verdict_ar = "جاهز للتداول الآلي (بحدود مخاطر)"
    elif (
        score >= 75
        and passed_gates
        and fit_st != "unstable_generalization"
        and not inflated_block
        and not saturated_block
    ):
        # Soft material warnings (e.g. val_test_gap_warn) → paper, not live
        verdict = "paper_ready"
        verdict_ar = "جاهز للورق — تحذيرات مادية تمنع live_ready"
    elif score >= 55 and passed_gates:
        verdict = "paper_ready"
        verdict_ar = "جاهز للورق / مراقبة حية محدودة"
    elif score >= 40:
        verdict = "research_only"
        verdict_ar = "بحثي فقط — يحتاج تحسين"
    else:
        verdict = "blocked"
        verdict_ar = "محظور للنشر"

    return {
        "score": round(score, 1),
        "verdict": verdict,
        "verdict_ar": verdict_ar,
        "reasons": reasons[:12],
        "deductions": deductions[:16],
        "material_warnings": material_warnings[:12],
        "timeframe": timeframe,
    }
