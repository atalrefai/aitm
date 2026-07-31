"""Live-trading readiness score (0–100) with explicit gate evidence."""

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

    score = 0.0
    reasons: list[str] = []
    deductions: list[str] = []

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

    if sharpe >= 1.0:
        score += 12
        reasons.append("oos_sharpe>=1")
    elif sharpe >= 0.25:
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
    elif fit_st == "unstable_generalization":
        deductions.append("unstable_generalization")
        score -= 12

    if regime.get("stable") is True:
        score += 6
        reasons.append("regime_stable")
    elif regime.get("stable") is False:
        deductions.append("regime_unstable")
        score -= 6

    dsr = float((adv.get("deflated_sharpe") or {}).get("deflated_sharpe", 0) or 0)
    pbo = float((adv.get("pbo") or {}).get("pbo", 0.5) or 0.5)
    if dsr >= 0.7:
        score += 5
    if pbo <= 0.45:
        score += 4
    elif pbo >= 0.55:
        deductions.append("elevated_pbo")
        score -= 4

    if stress.get("robust"):
        score += 5
        reasons.append("stress_robust")
    elif stress.get("scenarios"):
        if float(stress.get("worst_sharpe", 0) or 0) < -1.0:
            deductions.append("stress_fragile")
            score -= 5

    if mc.get("stable"):
        score += 5
        reasons.append("monte_carlo_stable")
    elif mc.get("enabled") and float(mc.get("p_profit", 0) or 0) < 0.5:
        deductions.append("mc_low_profit_prob")
        score -= 4

    # Session coverage: at least one session positive
    sess = (sessions.get("sessions") or {})
    pos_sess = sum(1 for v in sess.values() if not v.get("skipped") and float(v.get("sharpe", 0) or 0) > 0)
    if pos_sess >= 2:
        score += 4
        reasons.append("multi_session_edge")

    if data_intel.get("ready"):
        score += 3
    if zoo.get("enabled") and zoo.get("winner"):
        score += 2
        reasons.append(f"zoo_winner={zoo.get('winner')}")

    # TF-specific caution from report lessons
    tf = str(timeframe).upper()
    if tf == "H4" and auc < 0.55:
        score -= 15
        deductions.append("h4_near_chance_block")
    if tf == "H1" and fit_st == "overfitting":
        score -= 5
        deductions.append("h1_overfit_penalty")

    score = max(0.0, min(100.0, score))
    # Verdict bands
    if score >= 75 and passed_gates and fit_st != "unstable_generalization":
        verdict = "live_ready"
        verdict_ar = "جاهز للتداول الآلي (بحدود مخاطر)"
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
        "deductions": deductions[:12],
        "timeframe": timeframe,
    }
