"""Champion / Challenger comparison and promotion decision helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_champion(models_root: Path, symbol: str, timeframe: str) -> dict[str, Any] | None:
    path = Path(models_root) / symbol / timeframe / "champion.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_champion_metrics(models_root: Path, symbol: str, timeframe: str) -> dict[str, Any] | None:
    champ = load_champion(models_root, symbol, timeframe)
    if not champ:
        return None
    version = champ.get("version")
    if not version:
        return None
    metrics_path = Path(models_root) / symbol / timeframe / str(version) / "metrics_report.json"
    if not metrics_path.exists():
        # Try model_path parent
        mp = champ.get("model_path")
        if mp:
            metrics_path = Path(mp).parent / "metrics_report.json"
    if not metrics_path.exists():
        return {"champion": champ, "metrics": {}}
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        metrics = {}
    return {"champion": champ, "metrics": metrics, "metrics_path": str(metrics_path)}


def composite_score(metrics: dict[str, Any]) -> float:
    fin = metrics.get("financial_oos") or {}
    deploy = metrics.get("financial_deploy_holdout") or {}
    cls = metrics.get("classification") or {}
    fit = (metrics.get("fit_diagnosis") or {}).get("status") or ""
    sharpe = _safe_float(deploy.get("sharpe"), _safe_float(fin.get("sharpe")))
    dd = abs(_safe_float(fin.get("max_drawdown"), 1.0))
    exp = _safe_float(fin.get("expectancy"))
    auc = _safe_float(cls.get("roc_auc_ovr"))
    trades = _safe_float(fin.get("n_trades"))
    ready = _safe_float((metrics.get("live_readiness") or {}).get("score"))
    score = (
        1.0 * sharpe
        - 1.4 * dd
        + 40.0 * exp
        + 0.8 * (auc - 0.5)
        + 0.01 * min(trades, 80.0)
        + 0.01 * ready
    )
    if fit == "overfitting":
        score -= 0.75
    elif fit == "balanced":
        score += 0.15
    return float(score)


def compare_challenger_to_champion(
    challenger_metrics: dict[str, Any],
    *,
    models_root: Path,
    symbol: str,
    timeframe: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide whether challenger may replace champion for this TF."""
    cfg = cfg or {}
    prior = load_champion_metrics(models_root, symbol, timeframe)
    challenger_passed = bool(challenger_metrics.get("passed_gates"))
    challenger_score = composite_score(challenger_metrics)

    if prior is None or not (prior.get("champion") or {}).get("version"):
        return {
            "enabled": True,
            "has_champion": False,
            "challenger_score": round(challenger_score, 4),
            "decision": "promote_as_first_champion" if challenger_passed else "no_champion_reject",
            "promote": bool(challenger_passed),
            "reasons": [
                "no_prior_champion" if challenger_passed else "challenger_failed_gates_no_champion"
            ],
            "summary_ar": (
                "لا يوجد بطل سابق — يُرقّى المتحدي عند اجتياز البوابات"
                if challenger_passed
                else "لا يوجد بطل — والمتحدي رُفض بالبوابات"
            ),
        }

    champ_metrics = prior.get("metrics") or {}
    champ_score = composite_score(champ_metrics)
    delta = challenger_score - champ_score
    min_delta = float(cfg.get("challenger_min_score_delta", 0.05))
    min_sharpe_delta = float(cfg.get("challenger_min_sharpe_delta", 0.05))

    c_fin = challenger_metrics.get("financial_oos") or {}
    h_fin = champ_metrics.get("financial_oos") or {}
    sharpe_delta = _safe_float(c_fin.get("sharpe")) - _safe_float(h_fin.get("sharpe"))

    reasons: list[str] = []
    promote = False
    if not challenger_passed:
        decision = "keep_champion_challenger_failed_gates"
        reasons.append("challenger_failed_gates")
    elif delta >= min_delta and sharpe_delta >= min_sharpe_delta:
        decision = "promote_challenger"
        promote = True
        reasons.append(f"score_delta={delta:.3f}")
        reasons.append(f"sharpe_delta={sharpe_delta:.3f}")
    elif delta >= min_delta and _safe_float((challenger_metrics.get("live_readiness") or {}).get("score")) >= _safe_float(
        (champ_metrics.get("live_readiness") or {}).get("score")
    ):
        decision = "promote_challenger_readiness_tiebreak"
        promote = True
        reasons.append("score_up_readiness_not_worse")
    else:
        decision = "keep_champion"
        reasons.append(f"insufficient_delta score={delta:.3f} sharpe={sharpe_delta:.3f}")

    # Never promote overfitting over balanced champion
    c_fit = (challenger_metrics.get("fit_diagnosis") or {}).get("status")
    h_fit = (champ_metrics.get("fit_diagnosis") or {}).get("status")
    if promote and c_fit == "overfitting" and h_fit == "balanced":
        promote = False
        decision = "keep_champion_block_overfit_challenger"
        reasons.append("overfit_challenger_blocked")

    return {
        "enabled": True,
        "has_champion": True,
        "champion_version": (prior.get("champion") or {}).get("version"),
        "champion_score": round(champ_score, 4),
        "challenger_score": round(challenger_score, 4),
        "score_delta": round(delta, 4),
        "sharpe_delta": round(sharpe_delta, 4),
        "decision": decision,
        "promote": promote,
        "reasons": reasons,
        "champion_sharpe": _safe_float(h_fin.get("sharpe")),
        "challenger_sharpe": _safe_float(c_fin.get("sharpe")),
        "summary_ar": _ar(decision, delta, sharpe_delta),
        "compared_at": _utc(),
    }


def _ar(decision: str, delta: float, sharpe_delta: float) -> str:
    if decision.startswith("promote"):
        return f"ترقية المتحدي · Δscore={delta:+.3f} · ΔSharpe={sharpe_delta:+.3f}"
    if "overfit" in decision:
        return "الإبقاء على البطل — منع متحدي overfitting"
    if "failed_gates" in decision:
        return "الإبقاء على البطل — المتحدي لم يجتز البوابات"
    return f"الإبقاء على البطل · Δscore={delta:+.3f} · ΔSharpe={sharpe_delta:+.3f}"


def write_challenger_record(
    out_dir: Path,
    *,
    version: str,
    comparison: dict[str, Any],
    passed_gates: bool,
) -> None:
    payload = {
        "version": version,
        "passed_gates": bool(passed_gates),
        "role": "challenger",
        "comparison": comparison,
        "updated_at": _utc(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "challenger_vs_champion.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Also keep TF-level latest challenger pointer
    tf_dir = out_dir.parent
    (tf_dir / "challenger.json").write_text(
        json.dumps(
            {
                "version": version,
                "passed_gates": bool(passed_gates),
                "decision": comparison.get("decision"),
                "promote": comparison.get("promote"),
                "updated_at": _utc(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def maybe_update_champion(
    models_root: Path,
    *,
    symbol: str,
    timeframe: str,
    version: str,
    model_path: str,
    comparison: dict[str, Any],
    passed_gates: bool,
) -> dict[str, Any]:
    """Write champion.json only when gates pass AND comparison allows promotion."""
    if not passed_gates:
        return {"updated": False, "reason": "failed_gates"}
    if comparison.get("has_champion") and not comparison.get("promote"):
        return {"updated": False, "reason": comparison.get("decision")}
    champ = Path(models_root) / symbol / timeframe / "champion.json"
    payload = {
        "version": version,
        "model_path": model_path,
        "promoted_at": _utc(),
        "via": comparison.get("decision"),
        "challenger_score": comparison.get("challenger_score"),
        "champion_score_prev": comparison.get("champion_score"),
    }
    champ.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"updated": True, "champion": payload}
