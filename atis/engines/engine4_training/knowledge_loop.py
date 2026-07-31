"""Continuous learning / cumulative knowledge feedback loop.

Transforms isolated train runs into a durable knowledge base:
- Append experiment outcomes (gates, Sharpe, root cause).
- Update cause priors (bandit / Dirichlet-style via intelligence.py).
- Detect concept drift proxies (PSI-like feature shift, performance decay).
- Emit retrain recommendations for Engine 5 / scheduler.

References:
- Gama et al. (2014) — concept drift survey.
- López de Prado — online learning / research factories (AFML).
- Institutional MLOps: champion/challenger + scheduled retrain + drift monitors.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def knowledge_store_path(models_root: Path, symbol: str, timeframe: str) -> Path:
    return Path(models_root) / symbol / timeframe / "knowledge_loop.json"


def load_knowledge(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": 1,
            "created_at": _utc(),
            "episodes": [],
            "lessons": [],
            "cause_weights": {},
            "performance_ema": {},
            "updated_at": _utc(),
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "version": 1,
            "created_at": _utc(),
            "episodes": [],
            "lessons": [],
            "cause_weights": {},
            "performance_ema": {},
            "updated_at": _utc(),
            "corrupt_reload": True,
        }


def save_knowledge(path: Path, store: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    store["updated_at"] = _utc()
    path.write_text(json.dumps(store, indent=2), encoding="utf-8")


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """PSI between two 1-D distributions (feature drift proxy)."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 30 or len(actual) < 30:
        return 0.0
    qs = np.linspace(0, 100, bins + 1)
    breaks = np.unique(np.percentile(expected, qs))
    if len(breaks) < 3:
        return 0.0
    e_hist, _ = np.histogram(expected, bins=breaks)
    a_hist, _ = np.histogram(actual, bins=breaks)
    e = e_hist.astype(float) / max(e_hist.sum(), 1)
    a = a_hist.astype(float) / max(a_hist.sum(), 1)
    e = np.clip(e, 1e-4, None)
    a = np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def extract_lesson(result_metrics: dict[str, Any], *, passed: bool) -> dict[str, Any]:
    """Human+machine readable lesson from one train episode."""
    from atis.engines.engine4_training.intelligence import classify_root_cause

    fake = {"metrics": result_metrics, "passed_gates": passed, "timeframe": result_metrics.get("timeframe")}
    cause, notes = classify_root_cause(fake)
    fin = result_metrics.get("financial_oos") or {}
    fit = result_metrics.get("fit_diagnosis") or {}
    regime = result_metrics.get("regime_validation") or {}
    return {
        "at": _utc(),
        "passed": bool(passed),
        "root_cause": cause,
        "notes": notes,
        "fit_status": fit.get("status"),
        "oos_sharpe": fin.get("sharpe"),
        "oos_expectancy": fin.get("expectancy"),
        "regime_stable": regime.get("stable"),
        "gate_failures": list(result_metrics.get("gate_failures") or []),
        "action": (
            "promote_champion_candidate"
            if passed
            else f"remediate_{cause.lower().replace('/', '_')}"
        ),
    }


def update_performance_ema(
    ema: dict[str, float],
    *,
    sharpe: float,
    expectancy: float,
    alpha: float = 0.25,
) -> dict[str, float]:
    out = dict(ema)
    for key, val in (("sharpe", sharpe), ("expectancy", expectancy)):
        prev = float(out.get(key, val))
        out[key] = float(alpha * float(val) + (1.0 - alpha) * prev)
    return out


def record_training_episode(
    models_root: Path,
    *,
    symbol: str,
    timeframe: str,
    version: str,
    metrics: dict[str, Any],
    passed_gates: bool,
    feature_psi: float = 0.0,
    cause_weights_path: Path | None = None,
) -> dict[str, Any]:
    """Append episode + lesson; update cause weights and EMA performance."""
    from atis.engines.engine4_training.intelligence import (
        load_cause_weights,
        save_cause_weights,
        update_cause_weights,
    )

    path = knowledge_store_path(models_root, symbol, timeframe)
    store = load_knowledge(path)
    lesson = extract_lesson(metrics, passed=passed_gates)
    ready = metrics.get("live_readiness") or {}
    zoo = metrics.get("model_zoo") or {}
    self_opt = metrics.get("self_optimize") or {}
    episode = {
        "version": version,
        "at": _utc(),
        "passed_gates": bool(passed_gates),
        "pipeline_version": metrics.get("pipeline_version"),
        "financial_oos": {
            "sharpe": (metrics.get("financial_oos") or {}).get("sharpe"),
            "sortino": (metrics.get("financial_oos") or {}).get("sortino"),
            "expectancy": (metrics.get("financial_oos") or {}).get("expectancy"),
            "max_drawdown": (metrics.get("financial_oos") or {}).get("max_drawdown"),
            "n_trades": (metrics.get("financial_oos") or {}).get("n_trades"),
        },
        "live_readiness": {
            "score": ready.get("score"),
            "verdict": ready.get("verdict"),
        },
        "model_zoo_winner": zoo.get("winner"),
        "feature_psi": float(feature_psi),
        "lesson": lesson,
        "self_optimize": {
            "overrides": self_opt.get("overrides") or {},
            "notes": self_opt.get("notes") or [],
        },
    }
    episodes = list(store.get("episodes") or [])
    episodes.append(episode)
    store["episodes"] = episodes[-80:]  # retain recent history
    lessons = list(store.get("lessons") or [])
    lessons.append(lesson)
    store["lessons"] = lessons[-40:]
    # Closed-loop: next train run can merge these knobs automatically.
    if self_opt.get("overrides"):
        store["pending_overrides"] = dict(self_opt.get("overrides") or {})
        store["pending_overrides_at"] = _utc()
        store["pending_overrides_from_version"] = version

    fin = metrics.get("financial_oos") or {}
    store["performance_ema"] = update_performance_ema(
        dict(store.get("performance_ema") or {}),
        sharpe=float(fin.get("sharpe", 0.0) or 0.0),
        expectancy=float(fin.get("expectancy", 0.0) or 0.0),
    )

    # Cause prior update
    cw_path = cause_weights_path or (path.parent / "cause_weights.json")
    weights = load_cause_weights(cw_path)
    weights = update_cause_weights(weights, cause=str(lesson["root_cause"]), success=passed_gates)
    save_cause_weights(cw_path, weights)
    store["cause_weights"] = weights

    # Drift / decay advisory
    ema_s = float((store.get("performance_ema") or {}).get("sharpe", 0.0) or 0.0)
    cur_s = float(fin.get("sharpe", 0.0) or 0.0)
    decay = ema_s - cur_s
    store["last_advisory"] = {
        "feature_psi": float(feature_psi),
        "performance_decay": float(decay),
        "retrain_suggested": bool(feature_psi >= 0.25 or decay >= 1.0 or not passed_gates),
        "reason": (
            "feature_drift"
            if feature_psi >= 0.25
            else ("performance_decay" if decay >= 1.0 else ("gates_failed" if not passed_gates else "ok"))
        ),
    }
    save_knowledge(path, store)
    return store


def knowledge_rationale() -> dict[str, str]:
    return {
        "episodes": "Chronological train outcomes — cumulative experience memory.",
        "lessons": "Root-cause tagged remediation hints for the next experiment.",
        "psi": "Population Stability Index — feature distribution drift monitor.",
        "ema": "Exponentially smoothed OOS Sharpe/expectancy for decay detection.",
        "retrain": "Suggests continuous learning when drift or decay trips thresholds.",
    }
