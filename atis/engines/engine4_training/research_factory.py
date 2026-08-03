"""Research factory: one-hypothesis experiments + comparison board (v16)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def experiment_board_path(models_root: Path) -> Path:
    return Path(models_root) / "intelligence" / "research_factory.json"


def load_board(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "experiments": [], "created_at": _utc()}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "experiments": [], "corrupt_reload": True, "created_at": _utc()}


def infer_hypothesis(cfg: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Infer the single change under test from diagnosis first, then cfg flags."""
    diagnosis = metrics.get("self_diagnosis") or {}
    unified = diagnosis.get("unified_hypothesis") or {}
    if unified.get("code"):
        return {
            "code": str(unified.get("code")),
            "single_change": str(unified.get("single_change") or unified.get("code")),
            "ar": str(unified.get("ar") or unified.get("code")),
            "primary_root_cause": diagnosis.get("primary_root_cause") or unified.get("primary_root_cause"),
            "knobs": dict(unified.get("knobs") or diagnosis.get("suggested_config_diff") or {}),
            "source": "self_diagnosis",
        }
    if diagnosis.get("next_actions"):
        top = (diagnosis.get("next_actions") or [{}])[0]
        knobs = dict(top.get("knobs") or diagnosis.get("suggested_config_diff") or {})
        if top.get("code"):
            return {
                "code": str(top.get("code")),
                "single_change": ",".join(f"{k}={v}" for k, v in list(knobs.items())[:4])
                or str(top.get("code")),
                "ar": str(top.get("hypothesis_ar") or top.get("code")),
                "primary_root_cause": diagnosis.get("primary_root_cause"),
                "knobs": knobs,
                "source": "self_diagnosis",
            }

    active = cfg.get("research_active_hypothesis") or {}
    if isinstance(active, dict) and active.get("code"):
        return {
            "code": str(active.get("code")),
            "single_change": str(active.get("single_change") or active.get("code")),
            "ar": str(active.get("ar") or active.get("code")),
            "source": "research_active_hypothesis",
        }
    applied = metrics.get("self_optimize_applied") or {}
    barrier = metrics.get("barrier_sweep") or {}
    family = metrics.get("family_resolution") or {}
    fold_st = metrics.get("fold_stability") or {}
    if bool(cfg.get("use_promotion_validation_mode")) and str(
        cfg.get("validation_mode") or ""
    ).lower().startswith("cpcv"):
        return {
            "code": "cpcv_promotion",
            "single_change": "use_promotion_validation_mode=true",
            "ar": "ترقية بوضع CPCV",
            "source": "cfg_fallback",
        }
    if float(cfg.get("max_fold_trade_rate", 0.22) or 0.22) <= 0.16:
        return {
            "code": "lower_trade_rate_cap",
            "single_change": f"max_fold_trade_rate={cfg.get('max_fold_trade_rate')}",
            "ar": "خفض سقف معدل التداول",
            "source": "cfg_fallback",
        }
    if bool(cfg.get("fail_on_high_pbo")):
        return {
            "code": "strict_pbo_gate",
            "single_change": f"max_pbo={cfg.get('max_pbo', 0.55)}",
            "ar": "بوابة PBO صارمة",
            "source": "cfg_fallback",
        }
    if barrier.get("applied"):
        return {
            "code": "barrier_sweep",
            "single_change": f"atr={barrier.get('chosen_atr')},H={barrier.get('chosen_horizon')}",
            "ar": "تجربة مسح حواجز التسمية",
            "source": "cfg_fallback",
        }
    if applied:
        keys = sorted(applied.keys())
        return {
            "code": "self_optimize",
            "single_change": ",".join(keys[:4]),
            "ar": f"تطبيق تحسين ذاتي: {', '.join(keys[:4])}",
            "source": "cfg_fallback",
        }
    if family.get("conflict") and not bool(cfg.get("prefer_ensemble_on_conflict", False)):
        return {
            "code": "single_family_resolution",
            "single_change": family.get("reason"),
            "ar": f"اختيار عائلة واحدة: {family.get('selected_family')}",
            "source": "cfg_fallback",
        }
    if family.get("conflict"):
        return {
            "code": "family_resolution",
            "single_change": family.get("reason"),
            "ar": f"حل تعارض Zoo/Nested: {family.get('selected_family')}",
            "source": "cfg_fallback",
        }
    if fold_st.get("trade_rate_pegged"):
        return {
            "code": "trade_rate_saturation",
            "single_change": "fail_on_trade_rate_saturated",
            "ar": "معالجة تشبع معدل التداول",
            "source": "cfg_fallback",
        }
    if bool(cfg.get("use_ensemble")):
        return {
            "code": "ensemble",
            "single_change": "use_ensemble=true",
            "ar": "تجربة Ensemble soft-vote",
            "source": "cfg_fallback",
        }
    return {
        "code": "baseline_pipeline",
        "single_change": "pipeline_priority_hardening",
        "ar": "تشغيل خط الأنابيب مع hardening الأولويات",
        "source": "cfg_fallback",
    }


def append_experiment(
    models_root: Path,
    *,
    symbol: str,
    timeframe: str,
    version: str,
    metrics: dict[str, Any],
    cfg: dict[str, Any],
    passed_gates: bool,
) -> dict[str, Any]:
    path = experiment_board_path(models_root)
    board = load_board(path)
    hyp = infer_hypothesis(cfg, metrics)
    fin = metrics.get("financial_oos") or {}
    trade_lvl = metrics.get("trade_level_metrics") or {}
    diagnosis = metrics.get("self_diagnosis") or {}
    consistency = metrics.get("consistency") or {}
    quality = metrics.get("quality_compound") or {}
    # Prefer diagnosis-driven hypothesis when present (always, not only on gate fail)
    if diagnosis.get("unified_hypothesis", {}).get("code"):
        uh = diagnosis["unified_hypothesis"]
        hyp = {
            "code": str(uh.get("code")),
            "single_change": str(uh.get("single_change") or uh.get("code")),
            "ar": str(uh.get("ar") or uh.get("code")),
            "primary_root_cause": diagnosis.get("primary_root_cause"),
            "knobs": dict(uh.get("knobs") or diagnosis.get("suggested_config_diff") or {}),
            "source": "self_diagnosis",
        }
    elif diagnosis.get("next_actions"):
        top = (diagnosis.get("next_actions") or [{}])[0]
        if top.get("code"):
            knobs = dict(top.get("knobs") or diagnosis.get("suggested_config_diff") or {})
            hyp = {
                "code": str(top.get("code")),
                "single_change": ",".join(f"{k}={v}" for k, v in list(knobs.items())[:4]),
                "ar": str(top.get("hypothesis_ar") or top.get("code")),
                "primary_root_cause": diagnosis.get("primary_root_cause"),
                "knobs": knobs,
                "source": "self_diagnosis",
            }
    row = {
        "at": _utc(),
        "symbol": symbol,
        "timeframe": timeframe,
        "version": version,
        "hypothesis": hyp,
        "passed_gates": bool(passed_gates),
        "sharpe": fin.get("sharpe"),
        "sharpe_ci_low": fin.get("sharpe_ci_low"),
        "expectancy": fin.get("expectancy"),
        "trade_sharpe_raw": trade_lvl.get("trade_sharpe_raw"),
        "n_trades": fin.get("n_trades"),
        "auc": (metrics.get("classification") or {}).get("roc_auc_ovr"),
        "fit": (metrics.get("fit_diagnosis") or {}).get("status"),
        "pipeline_version": metrics.get("pipeline_version"),
        "primary_root_cause": diagnosis.get("primary_root_cause"),
        "metric_honesty_score": diagnosis.get("metric_honesty_score"),
        "generalization_score": diagnosis.get("generalization_score"),
        "consistency_score": consistency.get("score") if consistency else diagnosis.get("consistency_score"),
        "quality_compound_score": quality.get("score"),
        "safe_for_live": (diagnosis.get("safe_for_live") or {}).get("verdict"),
        "suggested_config_diff": diagnosis.get("suggested_config_diff") or hyp.get("knobs"),
        "self_diagnosis": {
            "primary_root_cause": diagnosis.get("primary_root_cause"),
            "suggested_config_diff": diagnosis.get("suggested_config_diff"),
            "next_actions": (diagnosis.get("next_actions") or [])[:2],
            "metric_honesty_score": diagnosis.get("metric_honesty_score"),
            "generalization_score": diagnosis.get("generalization_score"),
            "consistency_score": diagnosis.get("consistency_score"),
            "unified_hypothesis": diagnosis.get("unified_hypothesis"),
        }
        if diagnosis
        else {},
    }
    board.setdefault("experiments", []).append(row)
    # Keep last 200
    board["experiments"] = list(board["experiments"])[-200:]
    board["updated_at"] = _utc()
    # Stop decision on this TF history
    tf_hist = [e for e in board["experiments"] if e.get("timeframe") == timeframe][-6:]
    stop, stop_reason = _stop_rule(tf_hist, cfg)
    board["last_stop"] = {"timeframe": timeframe, "stop": stop, "reason": stop_reason}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(board, indent=2, ensure_ascii=False), encoding="utf-8")
    next_hyp = suggest_next_hypothesis(
        tf_hist, cfg, stop=stop, stop_reason=stop_reason, diagnosis=diagnosis
    )
    if next_hyp:
        # Always stamp TF — unscoped rate knobs previously poisoned every TF
        # (M30 20260803: desaturate 0.05 applied without timeframe).
        next_hyp = dict(next_hyp)
        next_hyp["timeframe"] = str(timeframe).upper()
        board["next_hypothesis"] = next_hyp
        path.write_text(json.dumps(board, indent=2, ensure_ascii=False), encoding="utf-8")
        next_path = Path(models_root) / "intelligence" / "next_hypothesis.json"
        next_path.write_text(json.dumps(next_hyp, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "path": str(path),
        "hypothesis": hyp,
        "suggested_config_diff": diagnosis.get("suggested_config_diff") or hyp.get("knobs") or {},
        "n_experiments": len(board["experiments"]),
        "stop_suggested": stop,
        "stop_reason": stop_reason,
        "tf_history_len": len(tf_hist),
        "next_hypothesis": next_hyp,
    }


def suggest_next_hypothesis(
    history: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    stop: bool,
    stop_reason: str,
    diagnosis: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """One next single-change experiment for the research factory loop.

    Prefer Self-Diagnostic dominant action when present (one hypothesis / iteration).
    """
    if stop and stop_reason == "kpi_reached":
        return {
            "code": "hold_champion",
            "single_change": "no_change",
            "ar": "إيقاف التجارب — KPI تحقق، حافظ على البطل",
            "knobs": {},
        }
    # Diagnosis-driven single change (closed-loop OS)
    if isinstance(diagnosis, dict) and (diagnosis.get("next_actions") or diagnosis.get("suggested_config_diff")):
        actions = list(diagnosis.get("next_actions") or [])
        top = actions[0] if actions else {}
        knobs = dict(top.get("knobs") or diagnosis.get("suggested_config_diff") or {})
        if knobs or top.get("code"):
            return {
                "code": str(top.get("code") or "diagnosis_patch"),
                "single_change": ",".join(f"{k}={v}" for k, v in list(knobs.items())[:4]) or str(top.get("code")),
                "ar": str(top.get("hypothesis_ar") or top.get("hypothesis") or "تصحيح من التشخيص الذاتي"),
                "knobs": knobs,
                "expected_effect": top.get("expected_effect"),
                "risk": top.get("risk"),
                "primary_root_cause": diagnosis.get("primary_root_cause"),
            }

    last = history[-1] if history else {}
    # Prefer last episode diagnosis stored on the board row
    last_diag = last.get("self_diagnosis") or {}
    if last_diag.get("suggested_config_diff"):
        knobs = dict(last_diag.get("suggested_config_diff") or {})
        top = (last_diag.get("next_actions") or [{}])[0]
        return {
            "code": str(top.get("code") or "diagnosis_patch"),
            "single_change": ",".join(f"{k}={v}" for k, v in list(knobs.items())[:4]),
            "ar": str(top.get("hypothesis_ar") or "تصحيح من التشخيص الذاتي"),
            "knobs": knobs,
            "primary_root_cause": last_diag.get("primary_root_cause"),
        }

    codes = [(h.get("hypothesis") or {}).get("code") for h in history[-3:]]
    if not bool(cfg.get("use_promotion_validation_mode")):
        return {
            "code": "cpcv_promotion",
            "single_change": "use_promotion_validation_mode=true",
            "ar": "التجربة التالية: تفعيل CPCV للترقية",
            "knobs": {"use_promotion_validation_mode": True},
        }
    if float(cfg.get("max_fold_trade_rate", 0.22) or 0.22) > 0.15:
        return {
            "code": "lower_trade_rate_cap",
            "single_change": "max_fold_trade_rate=0.12",
            "ar": "التجربة التالية: خفض سقف التداول إلى 0.12",
            "knobs": {"max_fold_trade_rate": 0.12, "fail_on_trade_rate_saturated": True},
        }
    if "strict_pbo_gate" not in codes:
        return {
            "code": "strict_pbo_gate",
            "single_change": "fail_on_high_pbo=true",
            "ar": "التجربة التالية: تشديد بوابة PBO",
            "knobs": {"fail_on_high_pbo": True, "max_pbo": 0.55},
        }
    if bool(cfg.get("prefer_ensemble_on_conflict", False)):
        return {
            "code": "single_family_resolution",
            "single_change": "prefer_ensemble_on_conflict=false",
            "ar": "التجربة التالية: اختيار عائلة واحدة بدل Ensemble",
            "knobs": {"prefer_ensemble_on_conflict": False, "use_ensemble_on_conflict": False},
        }
    return {
        "code": "monitor_shadow",
        "single_change": "shadow_m5",
        "ar": "مراقبة Shadow للبطل الحالي",
        "knobs": {},
        "note": last.get("version"),
    }


def _stop_rule(history: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[bool, str]:
    if len(history) < int(cfg.get("research_stop_min_runs", 3)):
        return False, "continue"
    cis = [float(h.get("sharpe_ci_low") or -999) for h in history]
    kpi = float(cfg.get("kpi_sharpe_ci_low", 1.5))
    if cis[-1] >= kpi and history[-1].get("passed_gates") and history[-1].get("fit") == "balanced":
        return True, "kpi_reached"
    recent = cis[-3:]
    if max(recent) - min(recent) <= float(cfg.get("iterative_ci_delta", 0.15)) and recent[-1] > 0:
        return True, "ci_stable"
    if len(history) >= int(cfg.get("iterative_max_experiments", 5)):
        return True, "budget_exhausted"
    return False, "continue"


def compare_last_two(models_root: Path, timeframe: str) -> dict[str, Any] | None:
    board = load_board(experiment_board_path(models_root))
    rows = [e for e in board.get("experiments") or [] if e.get("timeframe") == timeframe]
    if len(rows) < 2:
        return None
    a, b = rows[-2], rows[-1]
    return {
        "prev": a,
        "curr": b,
        "delta_sharpe": round(float(b.get("sharpe") or 0) - float(a.get("sharpe") or 0), 4),
        "delta_expectancy": round(
            float(b.get("expectancy") or 0) - float(a.get("expectancy") or 0), 6
        ),
        "same_hypothesis": (a.get("hypothesis") or {}).get("code")
        == (b.get("hypothesis") or {}).get("code"),
    }
