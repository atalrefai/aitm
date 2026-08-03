"""Shadow challenger bookkeeping for Engine 5 paper comparison (v16)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def shadow_store_path(models_root: Path, symbol: str, timeframe: str) -> Path:
    return Path(models_root) / symbol / timeframe / "shadow_challenger.json"


def register_shadow_challenger(
    models_root: Path,
    *,
    symbol: str,
    timeframe: str,
    version: str,
    model_path: str,
    metrics: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    """When gates pass but champion kept (or forced), register challenger for paper shadowing."""
    passed = bool(metrics.get("passed_gates"))
    promote = bool(comparison.get("promote"))
    decision = str(comparison.get("decision") or "")
    force = "keep_champion" in decision or not promote
    if not passed:
        return {"registered": False, "reason": "failed_gates"}
    if promote and "keep_champion" not in decision:
        return {"registered": False, "reason": "promoted_or_failed"}

    fin = metrics.get("financial_oos") or {}
    payload = {
        "role": "shadow_challenger",
        "version": version,
        "model_path": model_path,
        "registered_at": _utc(),
        "oos_sharpe": fin.get("sharpe"),
        "oos_expectancy": fin.get("expectancy"),
        "sharpe_conservative": fin.get("sharpe_conservative"),
        "trade_sharpe_raw": fin.get("trade_sharpe_raw"),
        "live_readiness": (metrics.get("live_readiness") or {}).get("score"),
        "vs_champion": comparison,
        "shadow_days_required": 14,
        "status": "pending_paper_shadow",
        "rolling_live": {"n_decisions": 0, "sum_pnl": 0.0, "sharpe_proxy": None},
        "force_keep_path": bool(force),
    }
    path = shadow_store_path(models_root, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"registered": True, "path": str(path), "version": version}


def record_shadow_decision(
    models_root: Path,
    *,
    symbol: str,
    timeframe: str,
    pnl: float,
) -> dict[str, Any] | None:
    """Append a paper PnL observation for the shadow challenger."""
    path = shadow_store_path(models_root, symbol, timeframe)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    roll = dict(data.get("rolling_live") or {})
    n = int(roll.get("n_decisions") or 0) + 1
    s = float(roll.get("sum_pnl") or 0.0) + float(pnl)
    roll["n_decisions"] = n
    roll["sum_pnl"] = round(s, 8)
    roll["mean_pnl"] = round(s / max(n, 1), 8)
    data["rolling_live"] = roll
    data["updated_at"] = _utc()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def shadow_ready_to_challenge_champion(
    models_root: Path,
    *,
    symbol: str,
    timeframe: str,
    min_decisions: int = 40,
    min_mean_pnl: float = 0.0,
) -> tuple[bool, dict[str, Any]]:
    path = shadow_store_path(models_root, symbol, timeframe)
    if not path.exists():
        return False, {"reason": "no_shadow"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False, {"reason": "corrupt"}
    roll = data.get("rolling_live") or {}
    n = int(roll.get("n_decisions") or 0)
    mean = float(roll.get("mean_pnl") or 0.0)
    ok = n >= int(min_decisions) and mean > float(min_mean_pnl)
    return ok, {"n_decisions": n, "mean_pnl": mean, "version": data.get("version"), "ready": ok}


def load_shadow_meta(models_root: Path, symbol: str, timeframe: str) -> dict[str, Any] | None:
    path = shadow_store_path(models_root, symbol, timeframe)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def retrain_advisory_path(models_root: Path) -> Path:
    return Path(models_root) / "intelligence" / "retrain_advisory.json"


def read_retrain_advisory(models_root: Path) -> dict[str, Any]:
    path = retrain_advisory_path(models_root)
    if not path.exists():
        return {"exists": False, "auto_retrain_recommended": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["exists"] = True
        return data
    except Exception as exc:
        return {"exists": True, "error": str(exc), "auto_retrain_recommended": False}


def write_retrain_request(
    models_root: Path,
    *,
    reason: str,
    source: str = "engine5",
    symbol: str | None = None,
) -> dict[str, Any]:
    """Persist a retrain request for the web/scheduler to pick up."""
    intel = Path(models_root) / "intelligence"
    intel.mkdir(parents=True, exist_ok=True)
    payload = {
        "requested_at": _utc(),
        "reason": reason,
        "source": source,
        "symbol": symbol,
        "status": "pending",
    }
    path = intel / "retrain_request.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
