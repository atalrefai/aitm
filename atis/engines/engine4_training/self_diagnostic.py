"""Self-Diagnostic Engine — causal post-run diagnosis for Engine-4.

Produces a machine-readable diagnosis object after every TF run:
primary root cause, evidence table, honesty/generalization/live scores,
ranked next actions, and bilingual narrative for risk meetings.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# North-star taxonomy (must match acceptance criteria / research factory).
PRIMARY_ROOT_CAUSES = (
    "TradePolicy",
    "Model/HP",
    "Features",
    "Labels",
    "DataQuality",
    "ValidationDesign",
    "MetricInflation",
    "RegimeShift",
)

_GATE_TO_CAUSE: dict[str, str] = {
    "trade_rate_saturated": "TradePolicy",
    "overtrading_folds": "TradePolicy",
    "oos_trade_rate": "TradePolicy",
    "inactive_folds": "TradePolicy",
    "inflated_sharpe": "MetricInflation",
    "overfit_sharpe_gap": "Model/HP",
    "overfit_sharpe_gap_hard": "Model/HP",
    "overfit_acc_sharpe": "Model/HP",
    "overfit_champion_blocked": "Model/HP",
    "early_folds_weak": "ValidationDesign",
    "fold_unstable": "ValidationDesign",
    "high_pbo": "Model/HP",
    "median_fold_val_sharpe": "ValidationDesign",
    "val_test_gap_weak_test": "RegimeShift",
    "val_test_gap_hard": "RegimeShift",
    "unstable_generalization": "RegimeShift",
    "crisis_holdout_weak": "RegimeShift",
    "recent_holdout_weak": "RegimeShift",
    "regime_unstable": "RegimeShift",
    "filter_driven_edge": "Features",
    "filter_driven_sparse": "Features",
    "feature_unstable": "Features",
    "h4_no_edge": "Labels",
    "label_quality_gate": "Labels",
    "data_quality_gate": "DataQuality",
    "weak_sharpe_ci": "MetricInflation",
    "sparse_deploy_unreliable": "MetricInflation",
    "monte_carlo_unstable": "ValidationDesign",
    "stress_fragile": "ValidationDesign",
}

_CAUSE_PRIORITY = {
    "MetricInflation": 0,
    "TradePolicy": 1,
    "Model/HP": 2,
    "ValidationDesign": 3,
    "Labels": 4,
    "Features": 5,
    "DataQuality": 6,
    "RegimeShift": 7,
}

_CAUSE_AR: dict[str, str] = {
    "TradePolicy": "سياسة التداول / معدل الصفقات",
    "Model/HP": "النموذج / المعاملات الفائقة",
    "Features": "الميزات",
    "Labels": "التسميات / الحواجز",
    "DataQuality": "جودة البيانات",
    "ValidationDesign": "تصميم التحقق الزمني",
    "MetricInflation": "تضخيم المقاييس",
    "RegimeShift": "تحول النظام / فجوة التعميم",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if v == v else default  # NaN guard
    except (TypeError, ValueError):
        return default


def _effective_rate_cap(cfg: dict[str, Any], timeframe: str = "") -> float:
    """Resolve max_fold_trade_rate with optional by_tf override."""
    rate_cap = _f(cfg.get("max_fold_trade_rate"), 0.12)
    tf = str(timeframe or cfg.get("_active_timeframe") or cfg.get("timeframe") or "").upper()
    by_tf = cfg.get("max_fold_trade_rate_by_tf") or {}
    if tf and tf in by_tf:
        rate_cap = _f(by_tf[tf], rate_cap)
    # Prefer policy snapshot from the run when present (already TF-resolved).
    pol = cfg.get("_resolved_max_fold_trade_rate")
    if pol is not None:
        rate_cap = _f(pol, rate_cap)
    # Also accept metrics-embedded effective rate when diagnosis cfg carries it.
    pol2 = (cfg.get("trade_policy") or {}).get("max_fold_trade_rate") if isinstance(cfg.get("trade_policy"), dict) else None
    if pol2 is not None:
        rate_cap = _f(pol2, rate_cap)
    return rate_cap


def _clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(x)))


def detect_metric_dishonesty(
    metrics: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flag non-discriminative / inflated detectors (DSR≈1, MC always-pass, etc.)."""
    cfg = cfg or {}
    fin = metrics.get("financial_oos") or {}
    infl = metrics.get("sharpe_inflation") or fin.get("sharpe_inflation") or {}
    adv = metrics.get("advanced_eval") or {}
    dsr = _f((adv.get("deflated_sharpe") or {}).get("deflated_sharpe"), 0.0)
    mc = metrics.get("monte_carlo") or {}
    stress = metrics.get("stress_testing") or {}
    cls = metrics.get("classification") or {}
    uncapped = _f(fin.get("sharpe_uncapped"))
    capped = _f(fin.get("sharpe"))
    trade_raw = _f(fin.get("trade_sharpe_raw") or (metrics.get("trade_level_metrics") or {}).get("trade_sharpe_raw"))
    auc = _f(cls.get("roc_auc_ovr"), 0.5)
    max_unc = _f(cfg.get("max_sharpe_uncapped"), 20.0)
    ratio = uncapped / max(abs(capped), 0.25)

    flags: list[str] = []
    rescue_min_trade = _f(cfg.get("uncapped_rescue_min_trade_sharpe"), 0.35)
    max_ratio_rescue = _f(cfg.get("max_uncapped_ratio_with_trade_rescue"), 4.0)
    trade_supported = trade_raw >= rescue_min_trade and ratio <= max_ratio_rescue
    if uncapped > max_unc and not trade_supported:
        flags.append("uncapped_absurd")
    if capped > 0.5 and ratio > _f(cfg.get("max_uncapped_to_capped_ratio"), 3.0) and not trade_supported:
        flags.append("uncapped_to_capped_ratio")
    if capped > 3.0 and trade_raw < _f(cfg.get("max_path_vs_trade_gap"), 0.12):
        flags.append("path_vs_trade_gap")
    if capped >= 4.0 and auc < 0.52:
        flags.append("sharpe_auc_mismatch")
    # DSR saturates at 1.0 for absurd high-Sharpe — non-discriminative detector
    # Skip when trade-level reality already supports the edge (M1/M5 honest HF).
    if dsr >= 0.99 and uncapped > max(max_unc * 0.75, 12.0) and not trade_supported:
        flags.append("dsr_non_discriminative")
    if (
        mc.get("enabled")
        and mc.get("stable")
        and _f(mc.get("p_profit"), 0) >= 0.95
        and uncapped > 15
        and not trade_supported
    ):
        flags.append("mc_always_pass")
    if (
        stress.get("robust")
        and flags
        and _f(stress.get("worst_sharpe"), 0) > 2.0
        and uncapped > 15
        and not trade_supported
    ):
        flags.append("stress_non_discriminative")
    if bool(infl.get("inflated")):
        flags.append("inflated_sharpe_gate")

    return {
        "dishonest": bool(flags),
        "flags": flags,
        "uncapped": round(uncapped, 4),
        "capped": round(capped, 4),
        "trade_sharpe_raw": round(trade_raw, 4),
        "uncapped_ratio": round(ratio, 4),
        "trade_supported_uncapped": bool(trade_supported and uncapped > max_unc),
        "dsr": round(dsr, 6),
        "auc": round(auc, 4),
        "mc_p_profit": mc.get("p_profit"),
    }


def compute_metric_honesty_score(
    metrics: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """0–100: how trustworthy reported risk metrics are for live decisions."""
    cfg = cfg or {}
    dish = detect_metric_dishonesty(metrics, cfg=cfg)
    fold_st = metrics.get("fold_stability") or {}
    fin = metrics.get("financial_oos") or {}
    score = 100.0
    deductions: list[str] = []

    for flag in dish["flags"]:
        if flag in {"uncapped_absurd", "inflated_sharpe_gate"}:
            score -= 30
            deductions.append(flag)
        elif flag in {"path_vs_trade_gap", "uncapped_to_capped_ratio", "sharpe_auc_mismatch"}:
            score -= 18
            deductions.append(flag)
        elif flag in {"dsr_non_discriminative", "mc_always_pass", "stress_non_discriminative"}:
            score -= 12
            deductions.append(flag)

    if fold_st.get("trade_rate_pegged"):
        score -= 15
        deductions.append("trade_rate_pegged")

    n_tr = _f(fin.get("n_trades"))
    if n_tr > 0 and n_tr < 12:
        score -= 20
        deductions.append("too_few_trades_for_ci")
    elif n_tr < 20:
        score -= 8
        deductions.append("sparse_trades")

    score = _clip(score)
    return {
        "score": round(score, 1),
        "deductions": deductions,
        "dishonesty": dish,
    }


def compute_generalization_score(metrics: dict[str, Any]) -> dict[str, Any]:
    """0–100: Train↔Val↔Test stability + early/late fold consistency."""
    fit = metrics.get("fit_diagnosis") or {}
    fold_st = metrics.get("fold_stability") or {}
    score = 100.0
    deductions: list[str] = []

    gap_tv = abs(_f(fit.get("sharpe_gap_train_val")))
    gap_vt = abs(_f(fit.get("sharpe_gap_val_test")))
    if gap_tv > 3.0:
        score -= 28
        deductions.append("train_val_gap_severe")
    elif gap_tv > 2.0:
        score -= 16
        deductions.append("train_val_gap")
    if gap_vt > 3.0:
        score -= 22
        deductions.append("val_test_gap_severe")
    elif gap_vt > 2.0:
        score -= 12
        deductions.append("val_test_gap")

    status = str(fit.get("status") or "")
    if status == "overfitting":
        score -= 18
        deductions.append("overfitting")
    elif status == "unstable_generalization":
        score -= 20
        deductions.append("unstable_generalization")
    elif status == "underfitting":
        score -= 14
        deductions.append("underfitting")

    if fold_st.get("early_folds_weak"):
        score -= 16
        deductions.append("early_folds_weak")
    if fold_st.get("stable") is False:
        score -= 12
        deductions.append("fold_unstable")

    if bool(fit.get("filter_driven_edge_risk")):
        score -= 10
        deductions.append("filter_driven_edge_risk")

    score = _clip(score)
    return {
        "score": round(score, 1),
        "deductions": deductions,
        "sharpe_gap_train_val": round(gap_tv, 4),
        "sharpe_gap_val_test": round(gap_vt, 4),
        "fit_status": status,
    }


def compute_live_tradability_score(
    metrics: dict[str, Any],
    *,
    passed_gates: bool,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """0–100: post-cost edge, trade-rate realism, stress, deploy reliability."""
    cfg = cfg or {}
    fin = metrics.get("financial_oos") or {}
    fold_st = metrics.get("fold_stability") or {}
    stress = metrics.get("stress_testing") or {}
    deploy = metrics.get("financial_deploy_holdout") or {}
    exp_cost = metrics.get("expectancy_vs_cost") or {}
    honesty = compute_metric_honesty_score(metrics, cfg=cfg)
    gen = compute_generalization_score(metrics)

    score = 55.0 if passed_gates else 25.0
    deductions: list[str] = []
    reasons: list[str] = []

    if passed_gates:
        reasons.append("passed_gates")
    else:
        deductions.append("failed_gates")

    if _f(fin.get("expectancy")) > 0:
        score += 10
        reasons.append("positive_expectancy")
    else:
        score -= 12
        deductions.append("nonpositive_expectancy")

    if exp_cost.get("covers") is True:
        score += 8
        reasons.append("expectancy_covers_cost")
    elif exp_cost.get("covers") is False:
        score -= 10
        deductions.append("expectancy_below_cost")

    if fold_st.get("trade_rate_pegged"):
        score -= 18
        deductions.append("trade_rate_saturated")

    if honesty["score"] < 50:
        score -= 20
        deductions.append("low_metric_honesty")
    elif honesty["score"] < 70:
        score -= 10
        deductions.append("moderate_metric_honesty")

    if gen["score"] < 50:
        score -= 16
        deductions.append("weak_generalization")
    elif gen["score"] < 70:
        score -= 8
        deductions.append("moderate_generalization")

    if stress.get("robust"):
        score += 5
        reasons.append("stress_robust")
    elif _f(stress.get("worst_sharpe")) < -1.0:
        score -= 8
        deductions.append("stress_fragile")

    deploy_n = _f(deploy.get("n_trades"))
    deploy_sh = _f(deploy.get("sharpe"))
    if deploy_n >= 20 and deploy_sh >= 0:
        score += 6
        reasons.append("deploy_ok")
    elif deploy_n > 0 and (deploy_n < 10 or deploy_sh < 0):
        score -= 10
        deductions.append("deploy_unreliable")

    score = _clip(score)
    return {
        "score": round(score, 1),
        "deductions": deductions,
        "reasons": reasons,
    }


def _evidence_rows(
    metrics: dict[str, Any],
    gate_failures: list[str],
    *,
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = cfg or {}
    fin = metrics.get("financial_oos") or {}
    train = metrics.get("financial_train") or {}
    val = metrics.get("financial_validation") or {}
    cls = metrics.get("classification") or {}
    fit = metrics.get("fit_diagnosis") or {}
    fold_st = metrics.get("fold_stability") or {}
    infl = metrics.get("sharpe_inflation") or fin.get("sharpe_inflation") or {}
    adv = metrics.get("advanced_eval") or {}
    mc = metrics.get("monte_carlo") or {}
    dish = detect_metric_dishonesty(metrics, cfg=cfg)
    rate_cap = _effective_rate_cap(cfg, str(metrics.get("timeframe") or ""))
    # Prefer run-resolved cap stored on metrics when diagnosis cfg is still base yaml.
    rate_cap = _f((metrics.get("trade_policy") or {}).get("max_fold_trade_rate"), rate_cap)
    rate_cap = _f(metrics.get("max_fold_trade_rate"), rate_cap)
    rows: list[dict[str, Any]] = []

    def add(gate: str, metric: str, value: Any, threshold: Any, note: str) -> None:
        rows.append(
            {
                "gate": gate,
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "note": note,
            }
        )

    if "trade_rate_saturated" in gate_failures or fold_st.get("trade_rate_pegged"):
        add(
            "trade_rate_saturated",
            "trade_rate_pegged_frac",
            fold_st.get("trade_rate_pegged"),
            f">={0.92 * rate_cap:.3f} on ≥80% liquid folds",
            f"median_fold_rate≈cap({rate_cap}); policy fill-to-cap smell",
        )
    if "inflated_sharpe" in gate_failures or infl.get("inflated"):
        add(
            "inflated_sharpe",
            "sharpe_uncapped",
            infl.get("sharpe_uncapped", fin.get("sharpe_uncapped")),
            cfg.get("max_sharpe_uncapped", 20.0),
            (
                f"ratio={infl.get('uncapped_ratio')} · trade_raw={infl.get('trade_sharpe_raw')} · "
                f"path_vs_trade={infl.get('path_vs_trade_gap')}"
            ),
        )
    if "early_folds_weak" in gate_failures or fold_st.get("early_folds_weak"):
        early = fold_st.get("early_fold_stats") or {}
        add(
            "early_folds_weak",
            "early_mean_acc/test_sharpe",
            {
                "mean_accuracy": early.get("mean_accuracy"),
                "mean_test_sharpe": early.get("mean_test_sharpe"),
                "frac_negative_test": early.get("frac_negative_test"),
                "n_early": early.get("n_early"),
            },
            {
                "min_acc": cfg.get("early_fold_min_acc", 0.58),
                "min_mean_test": cfg.get("early_fold_min_mean_test_sharpe", 0.0),
                "max_neg_frac": cfg.get("early_fold_max_neg_frac", 0.34),
            },
            "early CPCV/WF paths weak vs late paths",
        )
    if any(g.startswith("overfit") for g in gate_failures) or fit.get("status") == "overfitting":
        add(
            "overfit_sharpe_gap",
            "sharpe_gap_train_val",
            fit.get("sharpe_gap_train_val"),
            cfg.get("max_train_val_sharpe_gap", 2.0),
            (
                f"Train={train.get('sharpe')} → Val={val.get('sharpe')} → Test={fin.get('sharpe')} · "
                f"acc_gap_tv={fit.get('accuracy_gap_train_val')}"
            ),
        )
    if any(g in gate_failures for g in ("filter_driven_edge", "filter_driven_sparse", "h4_no_edge")):
        add(
            "filter_driven_or_no_edge",
            "auc_vs_sharpe",
            {"auc": cls.get("roc_auc_ovr"), "acc": cls.get("accuracy"), "sharpe": fin.get("sharpe")},
            {"min_auc": cfg.get("min_auc_for_live", 0.515), "min_acc": 0.53},
            "financial edge without discriminative power",
        )
    if dish["flags"]:
        add(
            "metric_honesty",
            "dishonesty_flags",
            dish["flags"],
            "empty",
            f"DSR={dish['dsr']} · MC_p_profit={dish['mc_p_profit']} · uncapped={dish['uncapped']}",
        )
    # Always emit core financial snapshot for the evidence table
    add(
        "snapshot",
        "oos_financial",
        {
            "sharpe": fin.get("sharpe"),
            "sharpe_uncapped": fin.get("sharpe_uncapped"),
            "trade_sharpe_raw": fin.get("trade_sharpe_raw"),
            "n_trades": fin.get("n_trades"),
            "expectancy": fin.get("expectancy"),
            "trade_rate": cls.get("trade_rate_filtered"),
            "auc": cls.get("roc_auc_ovr"),
            "dsr": (adv.get("deflated_sharpe") or {}).get("deflated_sharpe"),
            "pbo": (adv.get("pbo") or {}).get("pbo"),
            "mc_stable": mc.get("stable"),
        },
        "honest trade-level primary",
        "OOS snapshot — bar-path uncapped is diagnostic-only",
    )
    return rows


def infer_primary_root_cause(
    gate_failures: list[str],
    metrics: dict[str, Any],
    *,
    timeframe: str = "",
) -> tuple[str, list[str]]:
    """Pick dominant cause from gates + structural smells."""
    notes: list[str] = []
    candidates: list[tuple[int, str]] = []
    fold_st = metrics.get("fold_stability") or {}
    infl = metrics.get("sharpe_inflation") or (metrics.get("financial_oos") or {}).get("sharpe_inflation") or {}
    fit = metrics.get("fit_diagnosis") or {}
    cls = metrics.get("classification") or {}
    dq = metrics.get("data_quality") or {}
    dish = detect_metric_dishonesty(metrics)
    gates_set = {str(g) for g in gate_failures}
    tf = str(timeframe).upper()

    # Hard overrides for clear structural failures (risk-meeting clarity)
    if "h4_no_edge" in gates_set or (
        tf == "H4" and _f(cls.get("roc_auc_ovr"), 0.5) < 0.52 and _f(cls.get("accuracy"), 0.5) < 0.53
    ):
        return "Labels", notes + ["override:h4_no_edge"]
    if infl.get("inflated") or "inflated_sharpe" in gates_set or (
        dish.get("dishonest") and any(f in (dish.get("flags") or []) for f in ("uncapped_absurd", "path_vs_trade_gap", "uncapped_to_capped_ratio"))
    ):
        return "MetricInflation", notes + ["override:metric_inflation"]
    gap_tv = abs(_f(fit.get("sharpe_gap_train_val")))
    gap_vt = abs(_f(fit.get("sharpe_gap_val_test")))
    if (
        any(g.startswith("overfit") for g in gates_set)
        or fit.get("status") == "overfitting"
    ) and gap_tv >= 3.0:
        return "Model/HP", notes + ["override:severe_overfit_gap"]
    if "high_pbo" in gates_set and (
        fit.get("status") == "overfitting" or gap_vt >= 2.0 or gap_tv >= 1.5
    ):
        return "Model/HP", notes + ["override:pbo_corroborated_overfit"]

    for g in gate_failures:
        cause = _GATE_TO_CAUSE.get(str(g))
        if cause:
            candidates.append((_CAUSE_PRIORITY.get(cause, 9), cause))
            notes.append(f"gate:{g}→{cause}")

    if fold_st.get("trade_rate_pegged"):
        candidates.append((_CAUSE_PRIORITY["TradePolicy"], "TradePolicy"))
        notes.append("smell:trade_rate_pegged")
    if fold_st.get("early_folds_weak"):
        candidates.append((_CAUSE_PRIORITY["ValidationDesign"], "ValidationDesign"))
        notes.append("smell:early_folds_weak")
    if _f(cls.get("roc_auc_ovr"), 0.5) < 0.52 and _f(cls.get("accuracy"), 0.5) < 0.53:
        cause = "Labels" if tf == "H4" else "Features"
        candidates.append((_CAUSE_PRIORITY.get(cause, 5), cause))
        notes.append(f"smell:near_chance→{cause}")
    flags = dq.get("quality_flags") or dq.get("flags") or {}
    if isinstance(flags, dict) and (flags.get("excessive_gaps") or dq.get("gate_pass") is False):
        candidates.append((_CAUSE_PRIORITY["DataQuality"], "DataQuality"))
        notes.append("smell:data_quality")

    if not candidates:
        if abs(_f(fit.get("sharpe_gap_val_test"))) > 2.5:
            return "RegimeShift", notes + ["fallback:val_test_gap"]
        return "ValidationDesign", notes + ["fallback:no_dominant_gate"]

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], notes


def propose_next_actions(
    primary: str,
    gate_failures: list[str],
    metrics: dict[str, Any],
    *,
    timeframe: str = "",
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Ranked falsifiable hypotheses with knobs, expected effect, and risk."""
    cfg = cfg or {}
    tf = str(timeframe).upper()
    rate_cap = _effective_rate_cap(cfg, tf)
    rate_cap = _f((metrics.get("trade_policy") or {}).get("max_fold_trade_rate"), rate_cap)
    rate_cap = _f(metrics.get("max_fold_trade_rate"), rate_cap)
    actions: list[dict[str, Any]] = []
    fold_st = metrics.get("fold_stability") or {}
    infl = metrics.get("sharpe_inflation") or {}
    fit = metrics.get("fit_diagnosis") or {}

    def act(
        code: str,
        hypothesis: str,
        knobs: dict[str, Any],
        expected_effect: str,
        risk: str,
        *,
        ar: str,
        priority: int,
    ) -> None:
        actions.append(
            {
                "priority": priority,
                "code": code,
                "hypothesis": hypothesis,
                "hypothesis_ar": ar,
                "knobs": knobs,
                "expected_effect": expected_effect,
                "risk": risk,
                "falsifiable": True,
            }
        )

    if primary == "TradePolicy" or fold_st.get("trade_rate_pegged") or "trade_rate_saturated" in gate_failures:
        # Healthy by_tf baselines — a prior unscoped next_hypothesis can collapse the
        # cap to 0.05 and create a noop 0.05→0.05 loop (M30 20260803T094032).
        healthy_caps = {
            "M1": 0.06,
            "M5": 0.07,
            "M15": 0.08,
            "M30": 0.07,
            "H1": 0.05,
            "H4": 0.08,
        }
        healthy = healthy_caps.get(tf)
        min_conf = max(_f(cfg.get("min_trade_confidence"), 0.54), 0.58)
        conf_q = min(_f(cfg.get("confidence_quantile"), 0.85) + 0.04, 0.93)
        edge = max(_f(cfg.get("directional_edge"), 0.16), 0.18)
        headroom = _f(cfg.get("quality_first_cap_headroom_frac"), 0.82)
        if rate_cap <= 0.055 and healthy is not None and healthy > rate_cap + 1e-9:
            # Cap was collapsed below the TF baseline — restore + force headroom selection.
            new_cap = float(healthy)
            new_target = round(max(0.02, new_cap * 0.60), 3)
            headroom = min(headroom, 0.78)
            min_conf = max(min_conf, 0.60)
            conf_q = min(max(conf_q, 0.93), 0.95)
            edge = max(edge, 0.20)
            hyp = (
                f"Restore max_fold_trade_rate {rate_cap}→{new_cap} (poisoned floor) and "
                f"keep ops headroom under peg via target={new_target}."
            )
            hyp_ar = "استعادة سقف الإطار الصحي مع هامش تحت منطقة الالتصاق بدل خفض بلا أثر"
        else:
            desat_floor = 0.035
            new_cap = round(max(desat_floor, rate_cap * 0.70), 3)
            if abs(new_cap - rate_cap) < 1e-9:
                new_cap = round(max(0.028, rate_cap * 0.75), 3)
            new_target = round(max(0.02, new_cap * 0.70), 3)
            if rate_cap <= 0.055:
                min_conf = max(min_conf, 0.60)
                conf_q = min(max(conf_q, 0.93), 0.95)
                edge = max(edge, 0.20)
                headroom = min(headroom, 0.78)
            hyp = (
                f"Lower max_fold_trade_rate {rate_cap}→{new_cap} and raise confidence floor "
                "so selection is edge-based, not fill-to-cap."
            )
            hyp_ar = "خفض ضغط التداول وإجبار الاختيار حسب الثقة بدل ملء السقف"
        act(
            "desaturate_trade_policy",
            hyp,
            {
                "quality_first_trade_policy": True,
                "max_fold_trade_rate": new_cap,
                "target_trade_rate": new_target,
                "min_trade_confidence": min_conf,
                "confidence_quantile": conf_q,
                "directional_edge": edge,
                "quality_first_cap_headroom_frac": round(float(headroom), 3),
                "fail_on_trade_rate_saturated": True,
                "penalize_pegged_trade_rate_in_hpo": True,
            },
            "Reduce pegged-rate folds; trade_rate distribution moves below cap; honesty↑; win_rate↑",
            "May starve liquidity on thin TFs — watch min_trades_oos",
            ar=hyp_ar,
            priority=0 if primary == "TradePolicy" else 2,
        )
    if primary == "MetricInflation" or infl.get("inflated") or "inflated_sharpe" in gate_failures:
        act(
            "demote_uncapped_path_sharpe",
            "Make trade-level Sharpe + Deflated Sharpe primary; hard-fail uncapped only when trade_raw is weak.",
            {
                "fail_on_inflated_sharpe": True,
                "max_sharpe_uncapped": min(_f(cfg.get("max_sharpe_uncapped"), 20.0), 15.0),
                "max_uncapped_to_capped_ratio": min(_f(cfg.get("max_uncapped_to_capped_ratio"), 3.0), 2.5),
                "min_trade_sharpe_raw": max(_f(cfg.get("min_trade_sharpe_raw"), -0.05), 0.05),
                "uncapped_rescue_min_trade_sharpe": max(
                    _f(cfg.get("uncapped_rescue_min_trade_sharpe"), 0.25), 0.35
                ),
                "max_uncapped_ratio_with_trade_rescue": max(
                    _f(cfg.get("max_uncapped_ratio_with_trade_rescue"), 3.5), 4.0
                ),
                "rank_by_trade_sharpe": True,
            },
            "Reject weak-trade inflation; allow honest HF edges with strong trade_raw",
            "May still fail if trade_raw collapses under costs",
            ar="إخضاع شارب المسار غير المحدود للتشخيص واعتماد شارب الصفقات مع إنقاذ الحافة الصادقة",
            priority=0 if primary == "MetricInflation" else 2,
        )
    if primary == "Model/HP" or any(g.startswith("overfit") for g in gate_failures):
        # M5/H1/M1 capacity overfit: regularize LGB+RF and prefer Nested when Zoo is heavier.
        force_reg = tf in {"M5", "H1", "M1"} or primary == "Model/HP"
        zoo_win = str((metrics.get("model_zoo") or {}).get("winner") or "").lower()
        rf_overfit = zoo_win in {"random_forest", "rf", "extra_trees"} or tf == "M1"
        act(
            "regularize_capacity",
            "Penalize Train→Val Sharpe gap in nested HP; deepen regularization; prefer simpler zoo within ε.",
            {
                "lgb_max_depth": 3,
                "lgb_min_child_samples": 200,
                "lgb_reg_lambda": max(_f(cfg.get("lgb_reg_lambda"), 3.0), 5.5),
                "lgb_colsample": 0.42,
                "rf_max_depth": 4 if rf_overfit else int(cfg.get("rf_max_depth", 6) or 6),
                "rf_min_samples_leaf": 25 if rf_overfit else int(cfg.get("rf_min_samples_leaf", 12) or 12),
                "top_features": min(int(cfg.get("top_features", 56) or 56), 40),
                "nested_hp_train_val_gap_penalty": True,
                "prefer_simpler_within_epsilon": True,
                "prefer_nested_on_capacity_conflict": True,
                "force_regularize_hp": bool(force_reg),
                "regularize_capacity": True,
            },
            "Shrink Train≫Val gap; overfit_sharpe_gap clears; Nested beats high-capacity Zoo on conflict",
            "May underfit if signal is genuinely high-capacity",
            ar="تشديد التنظيم وتقليل السعة مع تفضيل العائلة الأبسط عند صراع Zoo/Nested",
            priority=0 if primary == "Model/HP" and "high_pbo" not in gate_failures else 2,
        )
    if "high_pbo" in gate_failures or (
        primary == "Model/HP" and _f((metrics.get("advanced") or metrics.get("pbo") or {}).get("pbo"), 0.0) >= 0.55
    ):
        # Prefer this as #1 when PBO is the hard gate — selection-bias / overfit, not early Acc.
        act(
            "reduce_selection_bias",
            "Cut capacity + tighten trade policy so IS→OOS CPCV paths stop flipping; keep max_pbo gate honest.",
            {
                "top_features": min(int((cfg.get("top_features_by_tf") or {}).get(tf, cfg.get("top_features", 56)) or 56), 40),
                "max_fold_trade_rate": min(_f(cfg.get("max_fold_trade_rate"), 0.09), 0.07),
                "confidence_quantile": min(_f(cfg.get("confidence_quantile"), 0.88) + 0.02, 0.93),
                "nested_hp_train_val_gap_penalty": True,
                "prefer_simpler_within_epsilon": True,
                "quality_first_trade_policy": True,
                "pbo_require_corroboration": True,
            },
            "Lower PBO below max_pbo; shrink Train→Val→Test cascade; keep Deploy liquid",
            "May reduce trade count / Val Sharpe if over-regularized",
            ar="تقليل انحياز الاختيار عبر تنظيم أقوى وسياسة صفقات أضيق دون رفع عتبة PBO",
            priority=0 if "high_pbo" in gate_failures else 2,
        )
    if primary == "ValidationDesign" or "early_folds_weak" in gate_failures:
        act(
            "stabilize_early_fold_signal",
            "Improve early-path generalization: milder time-decay, more stable features, AUC-aware early gate, honest Val.",
            {
                "time_decay_half_life": max(_f(cfg.get("time_decay_half_life"), 0.35), 0.45),
                "stable_feature_min_frac": max(_f(cfg.get("stable_feature_min_frac"), 0.6), 0.65),
                "early_fold_min_auc": max(_f(cfg.get("early_fold_min_auc"), 0.58), 0.60),
                "honest_val_sharpe_from_folds": True,
                "barrier_sweep_enabled": True,
                "label_cleaning_enabled": True,
                "stabilize_early_fold_signal": True,
            },
            "Early Acc/AUC↑ · Val≪optimistic path · fewer false early_folds_weak / val_test_gap_hard",
            "Milder decay may slightly reduce recent-regime fit",
            ar="تحسين تعميم الطيات المبكرة وصدق Val بدل تشديد عتبة الدقة فقط",
            priority=0 if primary == "ValidationDesign" else 3,
        )
    if primary in {"Labels", "Features"} or "h4_no_edge" in gate_failures or tf == "H4":
        if tf == "H4" or primary == "Labels":
            act(
                "quarantine_h4_relabel",
                "Quarantine H4 from live confirm until discriminative AUC≥0.55; rebuild barriers + longer horizon + denser features.",
                {
                    "quarantine_h4_confirm": True,
                    "barrier_atr_multiplier": 1.75,
                    "horizon_bars": 6,
                    "use_meta_labeling": True,
                    "fail_h4_near_chance": True,
                    "barrier_sweep_enabled": True,
                    "stable_feature_min_frac": 0.45,
                    "top_features": max(24, int((cfg.get("top_features_by_tf") or {}).get("H4", 28) or 28)),
                },
                "Stop false H4 confirmations; force label/feature redesign toward AUC≫0.5",
                "Loses H4 as confirm TF until rebuilt; may still fail liquidity if edge absent",
                ar="عزل H4 وإعادة بناء التسميات/الأفق/الميزات — الإبقاء للتأكيد فقط حتى AUC≥0.55",
                priority=0 if primary in {"Labels", "Features"} else 2,
            )
        else:
            act(
                "feature_label_redesign",
                "Drop unstable features; retune barriers; enable meta-labeling for filter-driven edges.",
                {
                    "use_meta_labeling": True,
                    "auto_drop_unstable_features": True,
                    "barrier_sweep_enabled": True,
                    "fail_on_filter_driven_edge": True,
                },
                "Raise AUC above chance while preserving post-cost expectancy",
                "Barrier sweep can change trade frequency",
                ar="إعادة تصميم الميزات/الحواجز مع تسمية فوقية",
                priority=0 if primary == "Features" else 3,
            )
    if primary == "DataQuality":
        act(
            "quarantine_tf_data",
            "Quarantine TF until gap/leakage audit passes; raise DQ hard gate.",
            {"data_intel_hard": True, "dq_gate_hard": True, "dq_gate_min_score": 70},
            "Block training on structurally broken series",
            "May pause TF entirely",
            ar="عزل الإطار حتى اجتياز تدقيق الفجوات/التسريب",
            priority=0,
        )
    if primary == "RegimeShift" or any("val_test_gap" in g for g in gate_failures):
        # Do NOT tighten val_test_sharpe_gap_hard_max here — that creates a closed-loop
        # reject spiral (M30 20260803: gap 3.08 failed only because pending override set 2.75).
        # Cap tune-slice Val by fold Test so CPCV freeze cannot invent Val≫Test (M30 095842).
        already_consensus = int(cfg.get("policy_min_agree_folds", 0) or 0) >= 3
        gap_vt = _f((metrics.get("fit_diagnosis") or {}).get("sharpe_gap_val_test"), 0.0)
        slack = 2.0 if gap_vt >= 3.5 else 2.5
        knobs_rs: dict[str, Any] = {
            "policy_min_agree_folds": max(int(cfg.get("policy_min_agree_folds", 3) or 3), 3),
            "prefer_simpler_within_epsilon": True,
            "nested_hp_train_val_gap_penalty": True,
            "lgb_max_depth": min(int(cfg.get("lgb_max_depth", 4) or 4), 3),
            "lgb_reg_lambda": max(_f(cfg.get("lgb_reg_lambda"), 3.0), 5.0),
            "fail_on_crisis_holdout_weak": True,
            "fail_on_unstable_generalization": True,
            "regime_balanced_holdouts": True,
            "fail_on_regime_balanced_holdouts": True,
            "honest_val_sharpe_from_folds": True,
            "honest_val_cap_by_fold_test": True,
            "honest_val_fold_test_slack": slack,
        }
        if already_consensus and gap_vt >= 3.2:
            # Consensus already on — still Val-optimistic: retune each fold + tighter slack.
            knobs_rs["tune_policy_mode"] = "each"
            knobs_rs["honest_val_fold_test_slack"] = min(slack, 2.0)
        act(
            "regime_stable_policy",
            "Cap Val by fold-Test+slack and freeze only on ≥3 liquid-fold consensus; do not move the gap gate.",
            knobs_rs,
            "Shrink Val≫Test optimism without moving the gap gate; clearer OOS policy",
            "May under-trade if folds disagree on thresholds",
            ar="تقييد شارب التحقق بطيات الاختبار + إجماع السياسة بدل تشديد عتبة الفجوة",
            priority=0 if primary == "RegimeShift" else 3,
        )
    # Explicit Phase-A knob when consistency / regime coverage is the smell
    consistency = metrics.get("consistency") or {}
    regime_unstable = (metrics.get("regime_validation") or {}).get("stable") is False
    low_consistency = consistency.get("enabled") and float(consistency.get("score", 100) or 100) < 55
    if low_consistency or "regime_balanced_holdouts_weak" in gate_failures or regime_unstable:
        if not any(a.get("code") in {"regime_stable_policy", "regime_balanced_holdouts"} for a in actions):
            act(
                "regime_balanced_holdouts",
                "Require edge across ≥3 regimes (WR/expectancy); raise consistency before live.",
                {
                    "regime_balanced_holdouts": True,
                    "fail_on_regime_balanced_holdouts": True,
                    "regime_balanced_min_regimes": 3,
                    "prefer_simpler_within_epsilon": True,
                    "quality_first_trade_policy": True,
                },
                "Consistency↑ · fewer single-regime lucky paths",
                "May fail TFs that only work in one regime",
                ar="فرض اتساق عبر أنظمة السوق قبل الترقية الحية",
                priority=1 if primary == "RegimeShift" else 4,
            )

    if not actions:
        act(
            "monitor_champion",
            "No dominant failure — hold champion; monitor PSI/drift only.",
            {},
            "Stability",
            "Stagnation if latent overfit",
            ar="مراقبة البطل دون تغيير",
            priority=9,
        )

    actions.sort(key=lambda a: int(a["priority"]))
    # One dominant hypothesis per iteration (keep top 3 as ranked backlog)
    return actions[:5]


def safe_for_live_verdict(
    *,
    passed_gates: bool,
    honesty: dict[str, Any],
    generalization: dict[str, Any],
    live: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Explicit live verdict with hard constraints — never live if material smells."""
    fold_st = metrics.get("fold_stability") or {}
    infl = metrics.get("sharpe_inflation") or (metrics.get("financial_oos") or {}).get("sharpe_inflation") or {}
    constraints_failed: list[str] = []

    if not passed_gates:
        constraints_failed.append("failed_promotion_gates")
    if fold_st.get("trade_rate_pegged"):
        constraints_failed.append("trade_rate_saturated")
    if infl.get("inflated"):
        constraints_failed.append("inflated_sharpe")
    if fold_st.get("early_folds_weak"):
        constraints_failed.append("early_folds_weak")
    if _f(honesty.get("score")) < 60:
        constraints_failed.append("metric_honesty_below_60")
    if _f(generalization.get("score")) < 60:
        constraints_failed.append("generalization_below_60")
    if abs(_f((metrics.get("fit_diagnosis") or {}).get("sharpe_gap_train_val"))) > 2.0:
        constraints_failed.append("train_val_gap_out_of_bounds")
    if abs(_f((metrics.get("fit_diagnosis") or {}).get("sharpe_gap_val_test"))) > 3.0:
        constraints_failed.append("val_test_gap_out_of_bounds")
    consistency = metrics.get("consistency") or {}
    if consistency.get("enabled") and _f(consistency.get("score"), 100) < float(
        (metrics.get("cfg_hint") or {}).get("min_consistency_for_live", 55)
    ):
        constraints_failed.append("consistency_below_min")

    ok = not constraints_failed and _f(live.get("score")) >= 70
    if ok:
        verdict = "safe_for_live"
        verdict_ar = "آمن للتداول الحي (بحدود مخاطر موثّقة)"
    elif passed_gates and not infl.get("inflated") and not fold_st.get("trade_rate_pegged"):
        verdict = "paper_only"
        verdict_ar = "ورق فقط — قيود سلامة غير مكتملة"
    else:
        verdict = "not_safe"
        verdict_ar = "غير آمن للنشر الحي"

    return {
        "safe_for_live": bool(ok),
        "verdict": verdict,
        "verdict_ar": verdict_ar,
        "constraints_failed": constraints_failed,
        "constraints": [
            "trade_rate not saturated",
            "Sharpe non-inflated (trade-level primary)",
            "early→late folds stable",
            "Train/Val/Test gaps within bounds",
            "metric honesty ≥ 60",
            "generalization ≥ 60",
            "consistency ≥ min (when enabled)",
            "passed promotion gates",
        ],
    }


def build_self_diagnosis(
    metrics: dict[str, Any],
    *,
    timeframe: str = "",
    passed_gates: bool = False,
    cfg: dict[str, Any] | None = None,
    gate_failures: list[str] | None = None,
) -> dict[str, Any]:
    """Full causal diagnosis object for one TF run."""
    cfg = cfg or {}
    gates = list(gate_failures if gate_failures is not None else (metrics.get("gate_failures") or []))
    primary, cause_notes = infer_primary_root_cause(gates, metrics, timeframe=timeframe)
    honesty = compute_metric_honesty_score(metrics, cfg=cfg)
    generalization = compute_generalization_score(metrics)
    live = compute_live_tradability_score(metrics, passed_gates=passed_gates, cfg=cfg)
    evidence = _evidence_rows(metrics, gates, cfg=cfg)
    actions = propose_next_actions(primary, gates, metrics, timeframe=timeframe, cfg=cfg)
    safe = safe_for_live_verdict(
        passed_gates=passed_gates,
        honesty=honesty,
        generalization=generalization,
        live=live,
        metrics=metrics,
    )
    dominant = actions[0] if actions else {}
    config_diff = dict(dominant.get("knobs") or {})
    consistency = metrics.get("consistency") or {}
    consistency_score = float(consistency.get("score", 0) or 0) if consistency else None

    narrative_en = (
        f"{timeframe}: primary_root_cause={primary}. "
        f"honesty={honesty['score']}/100 · generalization={generalization['score']}/100 · "
        f"live_tradability={live['score']}/100"
        + (f" · consistency={consistency_score}/100" if consistency_score is not None else "")
        + f". Gates={gates[:6] or ['(none)']}. "
        f"Next: {dominant.get('code', 'monitor')} — {dominant.get('hypothesis', '')}"
    )
    narrative_ar = (
        f"{timeframe}: السبب الجذري={_CAUSE_AR.get(primary, primary)}. "
        f"صدق المقاييس={honesty['score']}/100 · التعميم={generalization['score']}/100 · "
        f"قابلية التداول الحي={live['score']}/100"
        + (f" · الاتساق={consistency_score}/100" if consistency_score is not None else "")
        + f". البوابات={', '.join(gates[:4]) if gates else 'لا يوجد'}. "
        f"الخطوة التالية: {dominant.get('hypothesis_ar') or dominant.get('code')}"
    )

    return {
        "version": 2,
        "at": _utc(),
        "timeframe": str(timeframe).upper(),
        "passed_gates": bool(passed_gates),
        "primary_root_cause": primary,
        "primary_root_cause_ar": _CAUSE_AR.get(primary, primary),
        "cause_notes": cause_notes,
        "gate_failures": gates,
        "evidence": evidence,
        "metric_honesty_score": honesty["score"],
        "metric_honesty": honesty,
        "generalization_score": generalization["score"],
        "generalization": generalization,
        "live_tradability_score": live["score"],
        "live_tradability": live,
        "consistency_score": consistency_score,
        "consistency": consistency or None,
        "next_actions": actions,
        "suggested_config_diff": config_diff,
        "unified_hypothesis": {
            "code": dominant.get("code"),
            "single_change": ",".join(f"{k}={v}" for k, v in list(config_diff.items())[:4])
            if config_diff
            else str(dominant.get("code") or "monitor"),
            "ar": dominant.get("hypothesis_ar") or dominant.get("code"),
            "knobs": config_diff,
            "expected_effect": dominant.get("expected_effect"),
            "risk": dominant.get("risk"),
            "primary_root_cause": primary,
            "acceptance": {
                "must_not_weaken_gates": True,
                "prefer_consistency_and_wr_over_trade_rate": True,
                "reject_if_trade_rate_saturated": True,
                "reject_if_inflated_sharpe": True,
            },
        },
        "safe_for_live": safe,
        "narrative_en": narrative_en,
        "narrative_ar": narrative_ar,
        "champion_challenger_hint": (
            "reject_challenger"
            if not safe.get("safe_for_live")
            else ("promote_if_beats_champion" if passed_gates else "keep_champion")
        ),
    }


def write_diagnosis_json(path: Path, diagnosis: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diagnosis, indent=2, ensure_ascii=False), encoding="utf-8")


def format_diagnosis_markdown(diagnosis: dict[str, Any]) -> list[str]:
    """Markdown section for evaluation_report / enterprise dossier."""
    safe = diagnosis.get("safe_for_live") or {}
    lines = [
        "",
        "## Self-Diagnostic (Causal)",
        f"- Primary root cause: **{diagnosis.get('primary_root_cause')}** — {diagnosis.get('primary_root_cause_ar')}",
        (
            f"- Scores: honesty={diagnosis.get('metric_honesty_score')}/100 · "
            f"generalization={diagnosis.get('generalization_score')}/100 · "
            f"live_tradability={diagnosis.get('live_tradability_score')}/100 · "
            f"consistency={diagnosis.get('consistency_score')}/100"
        ),
        f"- Safe for live: **{safe.get('verdict')}** — {safe.get('verdict_ar')}",
        f"- Constraints failed: {safe.get('constraints_failed') or '(none)'}",
        f"- Unified hypothesis: `{(diagnosis.get('unified_hypothesis') or {}).get('code')}` — "
        f"{(diagnosis.get('unified_hypothesis') or {}).get('ar')}",
        f"- Narrative (EN): {diagnosis.get('narrative_en')}",
        f"- Narrative (AR): {diagnosis.get('narrative_ar')}",
        "",
        "### Evidence",
    ]
    for row in (diagnosis.get("evidence") or [])[:8]:
        lines.append(
            f"- `{row.get('gate')}` · {row.get('metric')}={row.get('value')} "
            f"(thr={row.get('threshold')}) — {row.get('note')}"
        )
    lines.extend(["", "### Ranked Next Actions"])
    for act in (diagnosis.get("next_actions") or [])[:4]:
        lines.append(
            f"- P{act.get('priority')}: `{act.get('code')}` — {act.get('hypothesis_ar') or act.get('hypothesis')}"
        )
        lines.append(
            f"  - expected: {act.get('expected_effect')} · risk: {act.get('risk')} · knobs: `{act.get('knobs')}`"
        )
    lines.append(f"- Suggested config diff: `{diagnosis.get('suggested_config_diff')}`")
    return lines
