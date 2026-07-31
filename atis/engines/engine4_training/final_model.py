"""Publish a single Final Model artifact from the best trained timeframe."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atis.config import get_path


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = -1e9) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def select_best_result(
    results: list[dict[str, Any]],
    *,
    allow_paper_final: bool = False,
    min_deploy_trades: int = 8,
) -> dict[str, Any] | None:
    """Pick the best trained model: gates first, then OOS quality + liquid TF preference."""
    candidates = [r for r in results if r.get("model_path") and not r.get("error")]
    if not candidates:
        return None

    gated = [r for r in candidates if bool(r.get("passed_gates"))]
    if not gated and not allow_paper_final:
        return None
    pool = gated if gated else candidates
    if not gated:
        preferred = [r for r in pool if str(r.get("timeframe")) in {"H1", "H4"}]
        if preferred:
            pool = preferred

    tf_priority = {"H1": 3.0, "H4": 2.0, "M30": 1.5, "M15": 1.0, "M5": 0.5, "M1": 0.25}

    def _fit_status(row: dict[str, Any]) -> str:
        metrics = row.get("metrics") or {}
        return str((metrics.get("fit_diagnosis") or {}).get("status") or "")

    # Prefer balanced fit; never crown overfitting when a balanced gated alternative exists.
    balanced = [r for r in pool if _fit_status(r) == "balanced"]
    if balanced:
        pool = balanced
    else:
        non_overfit = [r for r in pool if _fit_status(r) != "overfitting"]
        if non_overfit:
            pool = non_overfit

    def _deploy_trades(row: dict[str, Any]) -> float:
        metrics = row.get("metrics") or {}
        deploy = metrics.get("financial_deploy_holdout") or {}
        n = _safe_float(deploy.get("n_trades"), default=0.0)
        if n > 0:
            return n
        fin = metrics.get("financial_oos") or {}
        return _safe_float(fin.get("n_trades"), default=0.0)

    # Prefer candidates with enough deploy/OOS trades (report: H4 sharpe=1.63 on 1 trade).
    liquid = [r for r in pool if _deploy_trades(r) >= float(min_deploy_trades)]
    if liquid:
        pool = liquid

    # Reject failed H4/current-run collapse masquerading via prior champion is handled in publish.
    # Soft-ban near-chance AUC on slow TFs even if gates somehow passed.
    def _auc(row: dict[str, Any]) -> float:
        return _safe_float(((row.get("metrics") or {}).get("classification") or {}).get("roc_auc_ovr"), 0.0)

    signal_ok = [r for r in pool if _auc(r) >= 0.515 or str(r.get("timeframe")) not in {"H4", "D1"}]
    if signal_ok:
        pool = signal_ok

    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        metrics = row.get("metrics") or {}
        fin = metrics.get("financial_oos") or {}
        deploy = metrics.get("financial_deploy_holdout") or {}
        deploy_n = _safe_float(deploy.get("n_trades"), default=0.0)
        # Prefer deploy-holdout Sharpe only when sample size is meaningful.
        if deploy_n >= float(min_deploy_trades):
            sharpe = _safe_float(deploy.get("sharpe"), default=_safe_float(fin.get("sharpe")))
        else:
            sharpe = _safe_float(fin.get("sharpe"))
        dd = abs(_safe_float(fin.get("max_drawdown"), default=1.0))
        ret = _safe_float(fin.get("total_return"), default=-1.0)
        trades = _safe_float(fin.get("n_trades"), default=0.0)
        composite = (
            sharpe
            - 1.5 * dd
            + 0.8 * ret
            + 0.15 * tf_priority.get(str(row.get("timeframe")), 0.0)
            + 0.02 * min(trades, 100.0)
        )
        return (composite, sharpe, -dd, trades)

    return max(pool, key=key)


def publish_final_model(
    results: list[dict[str, Any]],
    *,
    symbol: str = "XAUUSD",
    allow_paper_final: bool = False,
) -> dict[str, Any]:
    """
    Copy the best timeframe artifact into models/FinalModel and write pointers.

    Never downgrade an existing live_ready FinalModel with a paper-only candidate.
    When allow_paper_final is False, ungated runs keep the existing Final (if any).
    """
    best = select_best_result(results, allow_paper_final=allow_paper_final)
    models_root = get_path("models")
    final_dir = models_root / "FinalModel"
    final_dir.mkdir(parents=True, exist_ok=True)
    meta_path = final_dir / "FINAL_MODEL.json"
    existing: dict[str, Any] = {}
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    if best is None:
        reason = "no_gated_results" if not allow_paper_final else "no_trainable_results"
        has_any = any(r.get("model_path") and not r.get("error") for r in results)
        if has_any and not allow_paper_final:
            reason = "no_gated_results_paper_final_disabled"
        payload = {
            "exists": bool(existing.get("exists")),
            "updated_at": _utc(),
            "symbol": symbol,
            "reason": reason,
            "updated_this_run": False,
            "current_run_passed_gates": sum(1 for r in results if r.get("passed_gates")),
            "current_run_candidates": [
                {
                    "timeframe": r.get("timeframe"),
                    "version": r.get("version"),
                    "passed_gates": bool(r.get("passed_gates")),
                    "sharpe": ((r.get("metrics") or {}).get("financial_oos") or {}).get("sharpe"),
                }
                for r in results
                if r.get("model_path") and not r.get("error")
            ],
            **({k: existing.get(k) for k in ("timeframe", "version", "mode", "artifact_path", "passed_gates") if existing.get(k) is not None}),
        }
        if existing.get("exists"):
            payload["kept_existing"] = True
            payload["champion_from_prior_run"] = True
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (models_root / "FinalModel.meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    # Anti-downgrade: keep live_ready champion if this run produced only paper models.
    if existing.get("passed_gates") and existing.get("exists") and not best.get("passed_gates"):
        existing = dict(existing)
        existing["updated_at"] = _utc()
        existing["skipped_downgrade"] = True
        existing["updated_this_run"] = False
        existing["champion_from_prior_run"] = True
        existing["current_run_passed_gates"] = sum(1 for r in results if r.get("passed_gates"))
        existing["rejected_candidate"] = {
            "timeframe": best.get("timeframe"),
            "version": best.get("version"),
            "sharpe": ((best.get("metrics") or {}).get("financial_oos") or {}).get("sharpe"),
        }
        meta_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        (models_root / "FinalModel.meta.json").write_text(json.dumps(existing, indent=2), encoding="utf-8")
        return existing

    src = Path(str(best["model_path"]))
    src_dir = src.parent
    dst_model = final_dir / "model.joblib"
    shutil.copy2(src, dst_model)

    for name in (
        "metrics_report.json",
        "backtest_report.json",
        "feature_list.json",
        "metadata.json",
        "training_config.yaml",
        "evaluation_report.md",
    ):
        p = src_dir / name
        if p.exists():
            shutil.copy2(p, final_dir / name)

    metrics = best.get("metrics") or {}
    fin = metrics.get("financial_oos") or {}
    val = metrics.get("financial_validation") or {}
    meta = {
        "exists": True,
        "artifact_type": "FinalModel",
        "symbol": best.get("symbol") or symbol,
        "timeframe": best.get("timeframe"),
        "version": best.get("version"),
        "source_model_path": str(src),
        "source_dir": str(src_dir),
        "artifact_path": str(dst_model),
        "artifact_dir": str(final_dir),
        "passed_gates": bool(best.get("passed_gates")),
        "mode": "live_ready" if best.get("passed_gates") else "paper_only",
        "updated_at": _utc(),
        "updated_this_run": True,
        "champion_from_prior_run": False,
        "current_run_passed_gates": sum(1 for r in results if r.get("passed_gates")),
        "selection_rule": "gates_first_then_composite_sharpe_dd_return_tf",
        "metrics": {
            "test": fin,
            "validation": val,
            "classification": metrics.get("classification") or {},
            "trade_policy": metrics.get("trade_policy") or {},
            "horizon_bars": metrics.get("horizon_bars"),
            "n_features": metrics.get("n_features"),
            "n_rows": metrics.get("n_rows"),
        },
        "candidates": [
            {
                "timeframe": r.get("timeframe"),
                "version": r.get("version"),
                "passed_gates": bool(r.get("passed_gates")),
                "sharpe": ((r.get("metrics") or {}).get("financial_oos") or {}).get("sharpe"),
                "max_drawdown": ((r.get("metrics") or {}).get("financial_oos") or {}).get("max_drawdown"),
                "total_return": ((r.get("metrics") or {}).get("financial_oos") or {}).get("total_return"),
            }
            for r in results
            if r.get("model_path") and not r.get("error")
        ],
    }

    (final_dir / "FINAL_MODEL.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (final_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (models_root / "FinalModel.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (models_root / "FinalModel.metrics.json").write_text(
        json.dumps(
            {
                "test": fin,
                "validation": val,
                "classification": metrics.get("classification") or {},
                "selected_timeframe": best.get("timeframe"),
                "passed_gates": bool(best.get("passed_gates")),
                "updated_at": _utc(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Relative champion for selected timeframe (best available, even if paper-only).
    tf = str(best.get("timeframe") or "H1")
    champ = models_root / symbol / tf / "champion.json"
    champ.parent.mkdir(parents=True, exist_ok=True)
    champ.write_text(
        json.dumps(
            {
                "version": best.get("version"),
                "model_path": str(src),
                "final_model_path": str(dst_model),
                "passed_gates": bool(best.get("passed_gates")),
                "mode": meta["mode"],
                "updated_at": _utc(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return meta
