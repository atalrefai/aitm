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
    """When gates pass but champion kept, register challenger for paper shadowing."""
    passed = bool(metrics.get("passed_gates"))
    promote = bool(comparison.get("promote"))
    if not passed or promote:
        return {"registered": False, "reason": "promoted_or_failed"}

    fin = metrics.get("financial_oos") or {}
    payload = {
        "role": "shadow_challenger",
        "version": version,
        "model_path": model_path,
        "registered_at": _utc(),
        "oos_sharpe": fin.get("sharpe"),
        "oos_expectancy": fin.get("expectancy"),
        "live_readiness": (metrics.get("live_readiness") or {}).get("score"),
        "vs_champion": comparison,
        "shadow_days_required": 14,
        "status": "pending_paper_shadow",
        "rolling_live": {"n_decisions": 0, "sum_pnl": 0.0, "sharpe_proxy": None},
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
