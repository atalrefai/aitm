"""Self-optimization from cumulative knowledge + enterprise markdown report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def propose_config_overrides(
    *,
    timeframe: str,
    metrics: dict[str, Any],
    passed_gates: bool,
    history_path: Path | None = None,
) -> dict[str, Any]:
    """Suggest next-run knobs from current failure/success patterns (report-driven)."""
    fit = (metrics.get("fit_diagnosis") or {}).get("status") or ""
    gates = list(metrics.get("gate_failures") or [])
    cls = metrics.get("classification") or {}
    fin = metrics.get("financial_oos") or {}
    readiness = metrics.get("live_readiness") or {}
    tf = str(timeframe).upper()
    overrides: dict[str, Any] = {}
    notes: list[str] = []

    if tf == "H4" or "weak_expectancy" in gates or float(cls.get("roc_auc_ovr", 1) or 1) < 0.52:
        overrides["barrier_atr_multiplier"] = 1.85
        overrides["horizon_bars_delta"] = -1
        overrides["use_meta_labeling"] = True
        overrides["lgb_reg_lambda"] = 6.5
        overrides["lgb_max_depth"] = 3
        notes.append("H4/near-chance: tighten model + retune barriers/meta-label")
    if fit == "overfitting" or "overfit" in "".join(gates):
        overrides["lgb_max_depth"] = min(int(overrides.get("lgb_max_depth", 3)), 3)
        overrides["lgb_min_child_samples"] = 200
        overrides["lgb_reg_lambda"] = max(float(overrides.get("lgb_reg_lambda", 5.5)), 5.5)
        overrides["lgb_colsample"] = 0.42
        overrides["top_features"] = 40
        notes.append("Overfit: stronger regularization + fewer features")
    if "val_fold_liquidity" in gates or "deploy_holdout_trades" in gates:
        overrides["min_trade_confidence"] = 0.50
        overrides["cost_edge_multiple"] = 1.15
        overrides["confidence_quantile"] = 0.78
        notes.append("Liquidity starve: soften policy thresholds carefully")
    if float(fin.get("expectancy", 0) or 0) <= 0:
        overrides["latency_bars"] = 1
        overrides["vol_slippage_k"] = 1.4
        notes.append("Nonpositive expectancy: raise execution realism + edge multiple")
    if readiness.get("verdict") in {"live_ready", "paper_ready"} and passed_gates:
        notes.append("Keep champion settings; schedule PSI drift monitor only")

    # History compare — called *before* current episode is recorded, so prev = last episode.
    hist_note = None
    if history_path and history_path.exists():
        try:
            raw = json.loads(history_path.read_text(encoding="utf-8"))
            eps = list(raw.get("episodes") or [])
            if eps:
                prev = eps[-1].get("financial_oos") or {}
                cur_s = float(fin.get("sharpe", 0) or 0)
                prev_s = float(prev.get("sharpe", 0) or 0)
                delta = cur_s - prev_s
                hist_note = {
                    "prev_sharpe": prev_s,
                    "delta_sharpe": round(delta, 4),
                    "n_prior_episodes": len(eps),
                }
                if delta < -1.0:
                    notes.append("Performance regress vs prior episode — prefer last champion HP")
                    overrides.setdefault("lgb_reg_lambda", 5.5)
                    overrides.setdefault("top_features", 40)
        except Exception:
            pass

    return {
        "timeframe": tf,
        "overrides": overrides,
        "notes": notes,
        "history": hist_note,
        "updated_at": _utc(),
    }


_APPLY_OVERRIDE_KEYS = {
    "barrier_atr_multiplier",
    "use_meta_labeling",
    "lgb_reg_lambda",
    "lgb_max_depth",
    "lgb_min_child_samples",
    "lgb_colsample",
    "top_features",
    "min_trade_confidence",
    "cost_edge_multiple",
    "confidence_quantile",
    "latency_bars",
    "vol_slippage_k",
}


def apply_pending_overrides(cfg: dict[str, Any], history_path: Path) -> dict[str, Any]:
    """Merge last-run self-optimize knobs into this run's config (closed-loop learning)."""
    applied: dict[str, Any] = {}
    if not history_path.exists():
        return applied
    try:
        raw = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception:
        return applied
    pending = dict(raw.get("pending_overrides") or {})
    if not pending:
        return applied
    for key, val in pending.items():
        if key == "horizon_bars_delta":
            try:
                delta = int(val)
                base = int(cfg.get("horizon_bars", cfg.get("label_horizon_bars", 6)) or 6)
                cfg["horizon_bars"] = max(2, base + delta)
                applied["horizon_bars"] = cfg["horizon_bars"]
            except Exception:
                pass
            continue
        if key in _APPLY_OVERRIDE_KEYS:
            cfg[key] = val
            applied[key] = val
    return applied


def write_enterprise_report(path: Path, payload: dict[str, Any]) -> None:
    """Executive markdown dossier after training."""
    m = payload
    fin = m.get("financial_oos") or {}
    cls = m.get("classification") or {}
    fit = m.get("fit_diagnosis") or {}
    data = m.get("data_intelligence") or m.get("data_quality") or {}
    feat = m.get("feature_intelligence") or {}
    label_q = m.get("label_quality") or (data.get("label_quality") if isinstance(data, dict) else {}) or {}
    feat_x = m.get("feature_explainability") or {}
    zoo = m.get("model_zoo") or {}
    regime = m.get("regime_validation") or {}
    stress = m.get("stress_testing") or {}
    mc = m.get("monte_carlo") or {}
    ready = m.get("live_readiness") or {}
    critique = m.get("intelligent_critique") or {}
    self_opt = m.get("self_optimize") or {}
    cc = m.get("champion_challenger") or {}
    recs = m.get("smart_recommendations") or {}
    nested = m.get("nested_hp") or {}
    gates = m.get("gate_failures") or []

    lines = [
        f"# ATIS Enterprise Training Dossier — {m.get('symbol')} {m.get('timeframe')}",
        "",
        f"- Pipeline: `{m.get('pipeline_version')}`",
        f"- Version: `{m.get('version')}`",
        f"- Passed gates: **{m.get('passed_gates')}**",
        f"- Live readiness: **{ready.get('score')}/100 · {ready.get('verdict')}** — {ready.get('verdict_ar')}",
        f"- Fit: **{fit.get('status')}**",
        f"- Champion/Challenger: **{cc.get('decision')}** — {cc.get('summary_ar')}",
        "",
        "## 1. Executive Summary",
        critique.get("executive_ar")
        or f"النموذج {'اجتاز' if m.get('passed_gates') else 'رُفض'} بوابات النشر مع Sharpe={fin.get('sharpe')} وExpectancy={fin.get('expectancy')}.",
        "",
        "## 2. Data Quality & Intelligence",
        f"- DQ/Intel score: {data.get('score')} · ready={data.get('ready', data.get('gate_pass'))}",
        f"- Rows/Features: {data.get('n_rows') or m.get('n_rows')} / {data.get('n_features') or m.get('n_features')}",
        f"- Flags: {data.get('flags') or data.get('quality_flags')}",
        f"- Summary: {data.get('summary_ar') or '—'}",
        "",
        "## 2b. Label Quality",
        f"- Score: {label_q.get('score')} · {label_q.get('summary_ar')}",
        f"- Noise rate: {(label_q.get('noise') or {}).get('noise_rate')} · Flags: {label_q.get('flags')}",
        f"- Recommendations: {label_q.get('recommendations')}",
        "",
        "## 3. Feature Intelligence & Explainability",
        f"- Selected: {feat.get('n_selected')} / input {feat.get('n_input')}",
        f"- Weak dropped: {feat.get('n_weak_dropped')} · Corr dropped: {feat.get('n_corr_dropped')}",
        f"- Top combined: {feat.get('top_combined')}",
        f"- SHAP: {(feat_x.get('shap') or {}).get('enabled')} · Permutation: {(feat_x.get('permutation') or {}).get('enabled')}",
        f"- Stability: {(feat_x.get('stability') or {}).get('summary_ar')}",
        f"- Consensus top: {feat_x.get('consensus_top')}",
        f"- Warnings: {feat_x.get('warnings')}",
        "",
        "## 4. Model Zoo & Nested HP",
        f"- Zoo Enabled: {zoo.get('enabled')} · Winner: **{zoo.get('winner')}** · Tried: {zoo.get('n_models_tried')}",
        f"- Nested HP: {nested.get('mode') or nested.get('enabled')} · Family: {nested.get('best_family')} · Score: {nested.get('best_score')}",
        f"- Ranking: {zoo.get('ranking')}",
        "",
        "## 5. Classification & Financial",
        f"- Acc/AUC: {cls.get('accuracy')} / {cls.get('roc_auc_ovr')}",
        f"- Sharpe/CI/Sortino/Exp: {fin.get('sharpe')} / {fin.get('sharpe_ci_low')} / {fin.get('sortino')} / {fin.get('expectancy')}",
        f"- MaxDD / PF / Trades: {fin.get('max_drawdown')} / {fin.get('profit_factor')} / {fin.get('n_trades')}",
        "",
        "## 6. Regime / Session / Stress / Monte Carlo",
        f"- Regime stable: {regime.get('stable')} · spread={regime.get('sharpe_regime_spread')}",
        f"- Stress robust: {stress.get('robust')} · worst_sharpe={stress.get('worst_sharpe')}",
        f"- Monte Carlo: p_profit={mc.get('p_profit')} · p_dd>25%={mc.get('p_dd_gt_25pct')} · stable={mc.get('stable')}",
        "",
        "## 7. Gate Failures",
        *( [f"- `{g}`" for g in gates] if gates else ["- (none)"] ),
        "",
        "## 8. Why this model / critique",
        f"- Strengths: {critique.get('strengths')}",
        f"- Weaknesses: {critique.get('weaknesses')}",
        f"- Root cause: {critique.get('root_cause')}",
        f"- Leakage/Drift flags: {critique.get('risk_flags')}",
        "",
        "## 9. Self-Optimize Suggestions",
        f"- Overrides: {self_opt.get('overrides')}",
        f"- Notes: {self_opt.get('notes')}",
        "",
        "## 10. Champion vs Challenger",
        f"- Decision: {cc.get('decision')} · promote={cc.get('promote')}",
        f"- Scores: challenger={cc.get('challenger_score')} vs champion={cc.get('champion_score')}",
        f"- Reasons: {cc.get('reasons')}",
        f"- Summary: {cc.get('summary_ar')}",
        "",
        "## 11. Smart Recommendations",
        f"- Primary: {recs.get('primary_code')} — {recs.get('executive_ar')}",
        *(
            [f"- P{it.get('priority')}: `{it.get('code')}` — {it.get('ar')}" for it in (recs.get('items') or [])[:8]]
            if recs.get("items")
            else ["- (none)"]
        ),
        "",
        "## 12. Live Readiness Evidence",
        f"- Reasons: {ready.get('reasons')}",
        f"- Deductions: {ready.get('deductions')}",
        "",
        f"_Generated {_utc()}_",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(x) for x in lines), encoding="utf-8")


def build_intelligent_critique(metrics: dict[str, Any], *, timeframe: str, passed: bool) -> dict[str, Any]:
    from atis.engines.engine4_training.intelligence import classify_root_cause

    cause, notes = classify_root_cause(
        {"metrics": metrics, "passed_gates": passed, "timeframe": timeframe}
    )
    fin = metrics.get("financial_oos") or {}
    cls = metrics.get("classification") or {}
    fit = (metrics.get("fit_diagnosis") or {}).get("status")
    strengths: list[str] = []
    weaknesses: list[str] = []
    risk_flags: list[str] = []

    if float(cls.get("roc_auc_ovr", 0) or 0) >= 0.55:
        strengths.append("discriminative_auc")
    if float(fin.get("expectancy", 0) or 0) > 0:
        strengths.append("positive_expectancy")
    if (metrics.get("regime_validation") or {}).get("stable"):
        strengths.append("regime_stable")
    if fit == "balanced":
        strengths.append("balanced_fit")
    if fit == "overfitting":
        weaknesses.append("overfitting")
        risk_flags.append("overfit_risk")
    if float(cls.get("roc_auc_ovr", 1) or 1) < 0.52:
        weaknesses.append("near_chance_classifier")
        risk_flags.append("possible_no_edge")
    if float(fin.get("sharpe", 0) or 0) < 0:
        weaknesses.append("negative_oos_sharpe")
    gap_vt = float((metrics.get("fit_diagnosis") or {}).get("sharpe_gap_val_test", 0) or 0)
    if abs(gap_vt) > 2.0:
        risk_flags.append("val_test_gap")
        weaknesses.append("unstable_val_test")
    psi = float(((metrics.get("knowledge_loop") or {}).get("advisory") or {}).get("feature_psi", 0) or 0)
    if psi >= 0.25:
        risk_flags.append("concept_drift_psi")
    if not passed:
        weaknesses.append("gate_failures")

    exec_ar = (
        f"{timeframe}: {'نجاح مشروط' if passed else 'فشل بوابات'} · السبب الجذري={cause} · "
        f"Fit={fit} · Sharpe={fin.get('sharpe')} · Exp={fin.get('expectancy')} · "
        f"AUC={cls.get('roc_auc_ovr')}."
    )
    return {
        "root_cause": cause,
        "notes": notes,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "risk_flags": risk_flags,
        "executive_ar": exec_ar,
        "recommendations": [
            "Preserve purged expanding WF + nested Val; never random split.",
            "Prefer liquid TFs (M5/M15/M30/H1) for live; quarantine H4 until AUC≫0.5.",
            "If overfitting: deepen regularization and shrink feature set.",
            "Keep latency≥1 in realism checks before scaling size.",
        ],
    }
