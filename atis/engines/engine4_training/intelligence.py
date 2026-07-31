"""Critical Intelligence + operational Awareness for Engine 4 runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT_CAUSES = (
    "Data",
    "Labeling",
    "Features",
    "Model/HP",
    "TradePolicy",
    "Liquidity/Sample",
    "RegimeShift",
)

# Prior weights (Dirichlet-style) updated from experiment outcomes.
_DEFAULT_CAUSE_WEIGHTS: dict[str, float] = {c: 1.0 for c in ROOT_CAUSES}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_cause_weights(path: Path | None) -> dict[str, float]:
    weights = dict(_DEFAULT_CAUSE_WEIGHTS)
    if path is None or not path.exists():
        return weights
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        for k, v in (raw.get("weights") or {}).items():
            if k in weights:
                weights[k] = float(v)
    except Exception:
        pass
    return weights


def save_cause_weights(path: Path, weights: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"weights": weights, "updated_at": _utc()}, indent=2),
        encoding="utf-8",
    )


def update_cause_weights(
    weights: dict[str, float],
    *,
    cause: str,
    success: bool,
    lr: float = 0.35,
) -> dict[str, float]:
    """Simple Bayesian / bandit update: boost causes that explained failures fixed later."""
    out = dict(weights)
    if cause not in out:
        return out
    if success:
        out[cause] = max(0.4, float(out[cause]) - lr * 0.5)
    else:
        out[cause] = float(out[cause]) + lr
    return out


def classify_root_cause(result: dict[str, Any]) -> tuple[str, list[str]]:
    """Map gate failures / fit diagnosis to a primary root cause."""
    metrics = result.get("metrics") or {}
    gates = list(metrics.get("gate_failures") or [])
    fit = (metrics.get("fit_diagnosis") or {}).get("status") or ""
    dq = metrics.get("data_quality") or {}
    cls = metrics.get("classification") or {}
    fin = metrics.get("financial_oos") or {}
    deploy = metrics.get("financial_deploy_holdout") or {}
    notes: list[str] = []

    error = str(result.get("error") or "")
    if error.startswith("dq_") or error.startswith("insufficient_rows") or dq.get("gate_pass") is False:
        notes.append("Data-quality gate or insufficient sample.")
        return "Data", notes

    auc = float(cls.get("roc_auc_ovr", 0.5) or 0.5)
    acc = float(cls.get("accuracy", 0.5) or 0.5)
    if auc < 0.52 and acc < 0.53:
        notes.append("Near-chance discriminative power — labeling/features/signal collapse.")
        if str(result.get("timeframe", "")).upper() == "H4":
            return "Labeling", notes
        return "Features", notes

    liq_gates = {
        "inactive_folds",
        "oos_trade_rate",
        "deploy_holdout_trades",
        "deploy_holdout_too_few_trades",
        "sparse_deploy_unreliable",
        "min_trades_oos",
        "val_fold_liquidity",
    }
    if any(g in liq_gates for g in gates) or float(deploy.get("n_trades", 0) or 0) < 8:
        notes.append("Starved Val/Deploy trades or inactive folds.")
        return "Liquidity/Sample", notes

    if fit == "overfitting" or any(g.startswith("overfit") for g in gates):
        notes.append("Train≫Val financial or accuracy gap.")
        return "Model/HP", notes

    if fit == "unstable_generalization" or "val_test_gap" in "".join(gates):
        notes.append("Val optimistic vs Test — policy or regime shift.")
        if abs(float((metrics.get("fit_diagnosis") or {}).get("sharpe_gap_val_test", 0) or 0)) > 2.0:
            return "RegimeShift", notes
        return "TradePolicy", notes

    if "filter_driven" in "".join(gates):
        notes.append("Filter-driven edge with weak classifier.")
        return "Features", notes

    if gates:
        notes.append(f"Gate failures: {gates[:4]}")
        return "TradePolicy", notes

    notes.append("Passed or soft issues only.")
    return "Features", notes


def propose_next_experiment(
    results: list[dict[str, Any]],
    *,
    cause_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """One controlled A/B change ranked by Expected Impact × compute cost."""
    weights = cause_weights or dict(_DEFAULT_CAUSE_WEIGHTS)
    failed = [r for r in results if (not r.get("passed_gates")) and not r.get("error")]
    errored = [r for r in results if r.get("error")]
    passed = [r for r in results if r.get("passed_gates")]

    focus = None
    # Priority: H4 failure/skip, then H1 overfit, then weakest passed gap
    for pref in ("H4", "H1", "M30", "M15", "M5", "M1"):
        for r in errored + failed:
            if str(r.get("timeframe", "")).upper() == pref:
                focus = r
                break
        if focus:
            break
    if focus is None and failed:
        focus = failed[0]
    if focus is None and passed:
        # Improve fragile passed (overfitting / large VT gap)
        fragile = []
        for r in passed:
            fit = ((r.get("metrics") or {}).get("fit_diagnosis") or {})
            if fit.get("status") in {"overfitting", "unstable_generalization"}:
                fragile.append(r)
            elif float(fit.get("sharpe_gap_val_test", 0) or 0) > 1.5:
                fragile.append(r)
        focus = fragile[0] if fragile else passed[0]

    if focus is None:
        return {
            "hypothesis": "No results to critique.",
            "change": None,
            "root_cause": "Data",
            "expected_impact": 0.0,
            "compute_cost": 0.0,
            "priority_score": 0.0,
        }

    cause, notes = classify_root_cause(focus)
    tf = str(focus.get("timeframe") or "?")
    w = float(weights.get(cause, 1.0))

    # Single-knob proposals by cause
    catalog = {
        "Data": {
            "hypothesis": f"{tf}: تحسين جودة/سيولة العينة يرفع n_val_trades وAUC فوق العشوائية.",
            "change": {
                "knob": "dq_and_sample",
                "actions": [
                    "raise max_train_bars / extend history",
                    "relax barrier_atr_multiplier slightly for slower TF",
                    "enforce DQ gate remediation before train",
                ],
            },
            "expected_impact": 0.85,
            "compute_cost": 0.4,
        },
        "Labeling": {
            "hypothesis": f"{tf}: إعادة ضبط أفق/حواجز Triple-Barrier تعيد إشارة AUC≫0.5.",
            "change": {
                "knob": "labeling",
                "actions": [
                    f"horizon_by_timeframe.{tf} ±1 bar",
                    f"barrier_atr_multiplier_by_tf.{tf} toward 1.6–2.0",
                    "enable meta-labeling if off",
                ],
            },
            "expected_impact": 0.9,
            "compute_cost": 0.55,
        },
        "Features": {
            "hypothesis": f"{tf}: ميزات نسبية + تفاعل أنظمة + ثبات عبر الطيات ترفع التعميم.",
            "change": {
                "knob": "features",
                "actions": [
                    "prefer_relative_features=true",
                    "stable_feature_min_frac += 0.05",
                    "drop absolute price columns",
                ],
            },
            "expected_impact": 0.65,
            "compute_cost": 0.45,
        },
        "Model/HP": {
            "hypothesis": f"{tf}: تنظيم أقوى + nested HP على Val Sharpe يقلّص overfitting.",
            "change": {
                "knob": "model_hp",
                "actions": [
                    "nested_hp_search=true with regularize bias",
                    "lgb_max_depth-=1, lgb_reg_lambda+=1.5, early_stopping+=20",
                    "compare logistic/HistGBM baseline",
                ],
            },
            "expected_impact": 0.8,
            "compute_cost": 0.7,
        },
        "TradePolicy": {
            "hypothesis": f"{tf}: تجميد السياسة فقط عند إجماع ≥3/5 طيات سائلة يخفض فجوة Val↔Test.",
            "change": {
                "knob": "trade_policy",
                "actions": [
                    "policy_min_agree_folds=3",
                    "exclude starved folds from freeze",
                    "cost_edge_multiple mild relax if starved",
                ],
            },
            "expected_impact": 0.7,
            "compute_cost": 0.35,
        },
        "Liquidity/Sample": {
            "hypothesis": f"{tf}: حد أدنى لصفقات Val + استبعاد الطيات الجائعة يمنع شارب وهمي.",
            "change": {
                "knob": "fold_liquidity",
                "actions": [
                    "min_val_trades_by_tf enforced",
                    "fail gate val_fold_liquidity if median n_val_trades too low",
                    "H4: skip with explanation rather than fake champion",
                ],
            },
            "expected_impact": 0.95,
            "compute_cost": 0.3,
        },
        "RegimeShift": {
            "hypothesis": f"{tf}: كشف انجراف (PSI/ADWIN) + إعادة تسمية عند تجاوز العتبة.",
            "change": {
                "knob": "awareness_drift",
                "actions": [
                    "enable drift_monitor",
                    "trigger retrain on PSI>threshold",
                    "anchored WF secondary check on liquid TFs",
                ],
            },
            "expected_impact": 0.6,
            "compute_cost": 0.5,
        },
    }
    pick = catalog.get(cause, catalog["Features"])
    priority = float(pick["expected_impact"]) * w / max(float(pick["compute_cost"]), 0.15)

    return {
        "timeframe": tf,
        "root_cause": cause,
        "cause_weight": round(w, 3),
        "notes": notes,
        "hypothesis": pick["hypothesis"],
        "change": pick["change"],
        "expected_impact": pick["expected_impact"],
        "compute_cost": pick["compute_cost"],
        "priority_score": round(priority, 4),
        "single_change_rule": "Apply only the listed knob in the next run; keep all else fixed for A/B.",
        "champion_context": {
            "passed": [str(r.get("timeframe")) for r in passed],
            "failed": [str(r.get("timeframe")) for r in failed],
            "errors": [
                {"timeframe": r.get("timeframe"), "error": r.get("error")} for r in errored
            ],
        },
    }


def build_awareness_report(
    *,
    timeframe: str,
    data_quality: dict[str, Any] | None = None,
    fit_diagnosis: dict[str, Any] | None = None,
    classification: dict[str, Any] | None = None,
    financial_oos: dict[str, Any] | None = None,
    financial_deploy: dict[str, Any] | None = None,
    gate_failures: list[str] | None = None,
    decisions: list[dict[str, str]] | None = None,
    feature_ref: np.ndarray | None = None,
    feature_cur: np.ndarray | None = None,
) -> dict[str, Any]:
    """Operational awareness: confidence, risk, drift hooks, decision explanations."""
    dq = data_quality or {}
    fit = fit_diagnosis or {}
    cls = classification or {}
    fin = financial_oos or {}
    deploy = financial_deploy or {}
    gates = list(gate_failures or [])

    auc = float(cls.get("roc_auc_ovr", 0.0) or 0.0)
    ci_low = float(fin.get("sharpe_ci_low", fin.get("sharpe", 0.0)) or 0.0)
    n_trades = float(fin.get("n_trades", 0.0) or 0.0)
    dq_score = float(dq.get("score", 70.0) or 70.0)

    # Confidence 0–1 combining signal, CI, liquidity, DQ
    conf = 0.25 * min(1.0, max(0.0, (auc - 0.5) / 0.25))
    conf += 0.25 * min(1.0, max(0.0, (ci_low + 0.5) / 2.0))
    conf += 0.25 * min(1.0, n_trades / 40.0)
    conf += 0.25 * min(1.0, dq_score / 100.0)
    if fit.get("status") == "overfitting":
        conf *= 0.7
    if fit.get("status") == "unstable_generalization":
        conf *= 0.75
    if gates:
        conf *= 0.85

    psi = None
    drift_flag = False
    if feature_ref is not None and feature_cur is not None:
        psi = float(population_stability_index(feature_ref, feature_cur))
        drift_flag = psi >= 0.25

    risk = {
        "overfit": fit.get("status") == "overfitting",
        "unstable": fit.get("status") == "unstable_generalization",
        "sparse_trades": n_trades < 12 or float(deploy.get("n_trades", 0) or 0) < 8,
        "weak_auc": auc > 0 and auc < 0.52,
        "drift": drift_flag,
        "gate_failures": gates,
    }

    explanations = list(decisions or [])
    if not explanations:
        explanations.append(
            {
                "decision": "fit_status",
                "why": str(fit.get("status", "n/a")),
                "ar": f"تشخيص الملاءمة لـ {timeframe}: {fit.get('status', 'n/a')}",
            }
        )
        if gates:
            explanations.append(
                {
                    "decision": "gate_reject" if gates else "gate_pass",
                    "why": ",".join(gates[:6]),
                    "ar": f"بوابات فاشلة: {', '.join(gates[:4])}",
                }
            )
        if dq.get("skip_reason"):
            explanations.append(
                {
                    "decision": "dq_skip",
                    "why": str(dq.get("skip_reason")),
                    "ar": (dq.get("awareness") or {}).get("explanation_ar")
                    or f"تخطي {timeframe} بسبب جودة البيانات",
                }
            )

    return {
        "timeframe": timeframe,
        "prediction_confidence": round(float(conf), 4),
        "data_quality_score": dq_score,
        "model_limits": {
            "fit_status": fit.get("status"),
            "sharpe_ci_low": ci_low,
            "n_trades_oos": n_trades,
            "deploy_trades": float(deploy.get("n_trades", 0) or 0),
            "auc": auc,
        },
        "risk_probability": {
            "publish_risk": round(1.0 - float(conf), 4),
            "flags": {k: bool(v) for k, v in risk.items() if k != "gate_failures"},
        },
        "drift": {"psi": psi, "flag": drift_flag, "method": "psi_topk_features"},
        "decisions": explanations,
        "updated_at": _utc(),
    }


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """PSI between two 1-D distributions (feature or return)."""
    exp = np.asarray(expected, dtype=float).ravel()
    act = np.asarray(actual, dtype=float).ravel()
    exp = exp[np.isfinite(exp)]
    act = act[np.isfinite(act)]
    if len(exp) < 30 or len(act) < 30:
        return 0.0
    qs = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(exp, qs))
    if len(edges) < 3:
        return 0.0
    e_hist, _ = np.histogram(exp, bins=edges)
    a_hist, _ = np.histogram(act, bins=edges)
    e_pct = e_hist / max(e_hist.sum(), 1)
    a_pct = a_hist / max(a_hist.sum(), 1)
    e_pct = np.clip(e_pct, eps, None)
    a_pct = np.clip(a_pct, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def critique_training_run(
    results: list[dict[str, Any]],
    *,
    models_root: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Full Intelligence pass → development_plan payload."""
    weights_path = (models_root / "intelligence" / "cause_weights.json") if models_root else None
    weights = load_cause_weights(weights_path)
    per_tf = []
    for r in results:
        cause, notes = classify_root_cause(r)
        m = r.get("metrics") or {}
        per_tf.append(
            {
                "timeframe": r.get("timeframe"),
                "passed_gates": bool(r.get("passed_gates")),
                "error": r.get("error"),
                "root_cause": cause,
                "notes": notes,
                "fit_status": (m.get("fit_diagnosis") or {}).get("status"),
                "acc": (m.get("classification") or {}).get("accuracy"),
                "auc": (m.get("classification") or {}).get("roc_auc_ovr"),
                "sharpe": (m.get("financial_oos") or {}).get("sharpe"),
                "sharpe_ci_low": (m.get("financial_oos") or {}).get("sharpe_ci_low"),
                "max_dd": (m.get("financial_oos") or {}).get("max_drawdown"),
                "gap_vt": (m.get("fit_diagnosis") or {}).get("sharpe_gap_val_test"),
                "n_trades": (m.get("financial_oos") or {}).get("n_trades"),
                "gate_failures": m.get("gate_failures") or [],
            }
        )
        success = bool(r.get("passed_gates"))
        weights = update_cause_weights(weights, cause=cause, success=success)

    nxt = propose_next_experiment(results, cause_weights=weights)
    summary_ar = _summary_ar(per_tf, nxt)
    plan = {
        "run_id": run_id,
        "created_at": _utc(),
        "pipeline": "engine4_intelligence",
        "kpi_targets": {
            "sharpe_ci_low_oos": 1.5,
            "max_dd_abs": 0.08,
            "val_test_sharpe_gap_max": 1.5,
            "fit_status": "balanced",
        },
        "per_timeframe": per_tf,
        "next_experiment": nxt,
        "cause_weights": weights,
        "summary_ar": summary_ar,
        "rules": [
            "One controlled change per experiment (A/B).",
            "Never promote failed H4 by substituting an old champion.",
            "Do not use uncapped Sharpe for live_ready decisions.",
            "Prefer balanced fit over overfitting when choosing FinalModel.",
        ],
    }
    if models_root is not None:
        intel_dir = models_root / "intelligence"
        intel_dir.mkdir(parents=True, exist_ok=True)
        save_cause_weights(intel_dir / "cause_weights.json", weights)
        out = intel_dir / f"development_plan_{run_id or 'latest'}.json"
        out.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        (intel_dir / "development_plan.json").write_text(
            json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (intel_dir / "development_plan.md").write_text(
            _plan_markdown(plan), encoding="utf-8"
        )
        plan["artifact_path"] = str(out)
    return plan


def _summary_ar(per_tf: list[dict[str, Any]], nxt: dict[str, Any]) -> str:
    lines = ["ملخص نقدي بعد التشغيل:"]
    for row in per_tf:
        st = "اجتاز" if row.get("passed_gates") else ("خطأ" if row.get("error") else "رُفض")
        lines.append(
            f"- {row.get('timeframe')}: {st} · سبب جذري={row.get('root_cause')} · "
            f"fit={row.get('fit_status')} · AUC={row.get('auc')} · Sharpe={row.get('sharpe')}"
        )
    lines.append(
        f"التجربة التالية المقترحة ({nxt.get('timeframe')} / {nxt.get('root_cause')}): {nxt.get('hypothesis')}"
    )
    return "\n".join(lines)


def _plan_markdown(plan: dict[str, Any]) -> str:
    nxt = plan.get("next_experiment") or {}
    lines = [
        "# ATIS Development Plan (Intelligence)",
        "",
        f"- Run: `{plan.get('run_id')}`",
        f"- Created: {plan.get('created_at')}",
        "",
        "## Summary (AR)",
        "",
        plan.get("summary_ar") or "",
        "",
        "## Next Experiment (single change)",
        "",
        f"- **Timeframe**: {nxt.get('timeframe')}",
        f"- **Root cause**: {nxt.get('root_cause')}",
        f"- **Hypothesis**: {nxt.get('hypothesis')}",
        f"- **Change**: `{json.dumps(nxt.get('change'), ensure_ascii=False)}`",
        f"- **Priority**: {nxt.get('priority_score')} (impact={nxt.get('expected_impact')} · cost={nxt.get('compute_cost')})",
        "",
        "## Per timeframe",
        "",
    ]
    for row in plan.get("per_timeframe") or []:
        lines.append(
            f"- `{row.get('timeframe')}` passed={row.get('passed_gates')} cause={row.get('root_cause')} "
            f"fit={row.get('fit_status')} gates={row.get('gate_failures')}"
        )
    lines.append("")
    return "\n".join(lines)
