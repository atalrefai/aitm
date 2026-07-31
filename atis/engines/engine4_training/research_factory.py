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
    """Infer the single change under test from cfg flags / self-opt / recs."""
    applied = metrics.get("self_optimize_applied") or {}
    barrier = metrics.get("barrier_sweep") or {}
    family = metrics.get("family_resolution") or {}
    if barrier.get("applied"):
        return {
            "code": "barrier_sweep",
            "single_change": f"atr={barrier.get('chosen_atr')},H={barrier.get('chosen_horizon')}",
            "ar": "تجربة مسح حواجز التسمية",
        }
    if applied:
        keys = sorted(applied.keys())
        return {
            "code": "self_optimize",
            "single_change": ",".join(keys[:4]),
            "ar": f"تطبيق تحسين ذاتي: {', '.join(keys[:4])}",
        }
    if family.get("conflict"):
        return {
            "code": "family_resolution",
            "single_change": family.get("reason"),
            "ar": f"حل تعارض Zoo/Nested: {family.get('selected_family')}",
        }
    if bool(cfg.get("use_ensemble")):
        return {
            "code": "ensemble",
            "single_change": "use_ensemble=true",
            "ar": "تجربة Ensemble soft-vote",
        }
    return {
        "code": "baseline_pipeline",
        "single_change": "pipeline_v16_defaults",
        "ar": "تشغيل خط الأنابيب v16 الافتراضي",
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
    return {
        "path": str(path),
        "hypothesis": hyp,
        "n_experiments": len(board["experiments"]),
        "stop_suggested": stop,
        "stop_reason": stop_reason,
        "tf_history_len": len(tf_hist),
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
