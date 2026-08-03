"""Automatic improvement recommendations after each training run."""

from __future__ import annotations

from typing import Any


def build_smart_recommendations(metrics: dict[str, Any], *, timeframe: str) -> dict[str, Any]:
    """Produce prioritized Arabic/English action list for next experiment.

    Phase A: when Self-Diagnostic has a unified hypothesis, that becomes primary;
    other items remain advisory backlog only (do not invent competing knobs).
    """
    gates = list(metrics.get("gate_failures") or [])
    fit = (metrics.get("fit_diagnosis") or {}).get("status") or ""
    fin = metrics.get("financial_oos") or {}
    cls = metrics.get("classification") or {}
    dq = metrics.get("data_quality") or {}
    label_q = metrics.get("label_quality") or dq.get("label_quality") or {}
    feat_x = metrics.get("feature_explainability") or {}
    zoo = metrics.get("model_zoo") or {}
    nested = metrics.get("nested_hp") or {}
    regime = metrics.get("regime_validation") or {}
    stress = metrics.get("stress_testing") or {}
    mc = metrics.get("monte_carlo") or {}
    ready = metrics.get("live_readiness") or {}
    cc = metrics.get("champion_challenger") or {}
    know = (metrics.get("knowledge_loop") or {}).get("advisory") or {}
    diagnosis = metrics.get("self_diagnosis") or {}
    unified = diagnosis.get("unified_hypothesis") or {}

    items: list[dict[str, Any]] = []

    def add(priority: int, code: str, ar: str, en: str, *, knobs: dict[str, Any] | None = None) -> None:
        items.append(
            {
                "priority": priority,
                "code": code,
                "ar": ar,
                "en": en,
                "suggested_knobs": knobs or {},
            }
        )

    # Diagnosis-driven primary (one hypothesis)
    if unified.get("code") or (diagnosis.get("next_actions") or []):
        top = unified if unified.get("code") else (diagnosis.get("next_actions") or [{}])[0]
        code = str(top.get("code") or "diagnosis_patch")
        knobs = dict(top.get("knobs") or diagnosis.get("suggested_config_diff") or {})
        add(
            0,
            code,
            str(top.get("ar") or top.get("hypothesis_ar") or code),
            str(top.get("hypothesis") or top.get("expected_effect") or code),
            knobs=knobs,
        )

    if "data_quality_gate" in gates or label_q.get("fail_hard"):
        add(
            1,
            "fix_labels_or_data",
            "أصلح جودة البيانات/Labels قبل أي تدريب جديد.",
            "Fix data/label quality before retraining.",
            knobs={"barrier_atr_multiplier": 1.85, "dq_gate_hard": True} if not unified else {},
        )
    if fit == "overfitting" or any("overfit" in g for g in gates):
        add(
            1,
            "regularize",
            "قلّل عمق الشجرة وزِد التنظيم وقلّل عدد الميزات.",
            "Strengthen regularization and shrink the feature set.",
            knobs={"lgb_max_depth": 3, "lgb_reg_lambda": 6.0, "top_features": 40} if not unified else {},
        )
    if fit == "underfitting":
        add(
            2,
            "capacity",
            "زد سعة النموذج قليلاً أو وسّع مجموعة الميزات.",
            "Slightly increase model capacity or feature budget.",
            knobs={"lgb_num_leaves": 20, "top_features": 64} if not unified else {},
        )
    if "val_fold_liquidity" in gates or "deploy_holdout_trades" in gates:
        add(
            1,
            "liquidity",
            "خفّف عتبات الثقة بحذر لزيادة سيولة الصفقات.",
            "Soften confidence floors carefully to restore trade liquidity.",
            knobs={"confidence_quantile": 0.78, "cost_edge_multiple": 1.15} if not unified else {},
        )
    if float(cls.get("roc_auc_ovr", 1) or 1) < 0.52:
        add(
            1,
            "no_edge",
            "التصنيف قريب من العشوائية — أعد تصميم Labels/Features أو جرّب Model Zoo فائزاً آخر.",
            "Near-chance classifier — redesign labels/features or try zoo runner-up.",
            knobs={"use_meta_labeling": True, "model_zoo_enabled": True} if not unified else {},
        )
    if float(fin.get("expectancy", 0) or 0) <= 0:
        add(
            2,
            "expectancy",
            "Expectancy غير إيجابي — ارفع تكلفة التنفيذ في المحاكاة وشدّد فلتر الحافة.",
            "Non-positive expectancy — raise realism costs and tighten edge filter.",
            knobs={"latency_bars": 1, "cost_edge_multiple": 1.4} if not unified else {},
        )
    if regime.get("stable") is False or "regime_unstable" in gates:
        add(
            2,
            "regime",
            "الأداء غير مستقر عبر أنظمة السوق — فعّل فلتر النظام واختبر CPCV-lite.",
            "Unstable across regimes — keep regime filter and try cpcv_lite validation.",
            knobs={"regime_filter": True, "validation_mode": "cpcv_lite"} if not unified else {},
        )
    if stress.get("robust") is False or "stress_fragile" in gates:
        add(
            2,
            "stress",
            "هشاشة تحت الضغط — زد السبريد/الانزلاق في التدريب قبل الترقية.",
            "Stress-fragile — inflate spread/slippage assumptions before promotion.",
            knobs={"spread_pips": 40.0, "slippage_pips": 8.0} if not unified else {},
        )
    if mc.get("stable") is False:
        add(
            3,
            "monte_carlo",
            "مسارات مونت كارلو غير مستقرة — قلّل معدل التداول وارفع الحد الأدنى للصفقات.",
            "Monte Carlo unstable — lower trade rate and raise min trades.",
            knobs={"target_trade_rate": 0.045, "min_trades_oos": 30} if not unified else {},
        )
    if feat_x.get("warnings") or []:
        add(
            2,
            "feature_stability",
            "ميزات غير مستقرة عبر الطيات — اعتمد الميزات المستقرة فقط وفعّل SHAP.",
            "Unstable features across folds — keep stable set and enable SHAP.",
            knobs={"stable_feature_selection": True, "shap_enabled": True} if not unified else {},
        )
    if label_q.get("flags", {}).get("high_noise"):
        add(
            1,
            "label_noise",
            "ضوضاء Labels مرتفعة — زد مضاعف الحاجز أو فعّل meta-labeling.",
            "High label noise — widen barriers or enable meta-labeling.",
            knobs={"barrier_atr_multiplier": 2.0, "use_meta_labeling": True} if not unified else {},
        )
    if zoo.get("winner") and nested.get("best_family") and str(zoo["winner"]) != str(nested.get("best_family")):
        add(
            3,
            "model_family_conflict",
            f"Zoo={zoo.get('winner')} يختلف عن Nested HP={nested.get('best_family')} — أعد البحث داخل كل طية.",
            "Zoo vs nested family mismatch — enable per-fold nested HP.",
            knobs={"nested_hp_per_fold": True, "use_ensemble": True} if not unified else {},
        )
    if cc.get("decision") == "keep_champion":
        add(
            3,
            "challenger_gap",
            "المتحدي لم يتفوق على البطل — غيّر فرضية واحدة فقط في التجربة التالية.",
            "Challenger did not beat champion — change one hypothesis only next run.",
        )
    if know.get("retrain_suggested"):
        add(
            2,
            "drift_retrain",
            f"حلقة المعرفة تقترح إعادة تدريب: {know.get('reason')}.",
            f"Knowledge loop suggests retrain: {know.get('reason')}.",
        )
    if metrics.get("passed_gates") and float(ready.get("score") or 0) >= 75:
        add(
            4,
            "monitor",
            "جاهز للمراقبة الحية — راقب PSI وانحراف الأداء مقابل البطل.",
            "Live-ready — monitor PSI and performance drift vs champion.",
        )
    if not items:
        add(
            4,
            "iterate",
            "لا إشارات حرجة — واصل التجارب المتداخلة مع هدف مالي واحد لكل تشغيل.",
            "No critical flags — continue nested experiments with one financial objective.",
        )

    items.sort(key=lambda x: x["priority"])
    top = items[:8]
    primary = top[0] if top else None
    exec_ar = " · ".join(i["ar"] for i in top[:3])
    return {
        "enabled": True,
        "timeframe": str(timeframe).upper(),
        "n_recommendations": len(top),
        "items": top,
        "executive_ar": exec_ar,
        "primary_code": primary["code"] if primary else None,
        "primary_knobs": dict(primary.get("suggested_knobs") or {}) if primary else {},
        "unified_from_diagnosis": bool(unified.get("code") or diagnosis.get("suggested_config_diff")),
        "acceptance": (unified.get("acceptance") if unified else None)
        or {
            "must_not_weaken_gates": True,
            "prefer_consistency_and_wr_over_trade_rate": True,
        },
    }
