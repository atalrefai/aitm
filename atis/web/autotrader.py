"""Background auto-trading loop for Gold Desk (paper/demo)."""

from __future__ import annotations

import importlib
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from atis.config import clear_config_caches, load_engine_config
from atis.shared.logging_utils import get_logger

logger = get_logger("atis.autotrader")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_timeframes(
    timeframe: str | None = None,
    timeframes: list[str] | None = None,
) -> list[str]:
    """Resolve one or many TFs into a unique ordered list (respect live TF policy)."""
    out: list[str] = []
    if timeframes:
        for tf in timeframes:
            t = str(tf or "").strip().upper()
            if t and t not in out:
                out.append(t)
    if not out and timeframe:
        t = str(timeframe).strip().upper()
        if t:
            out.append(t)
    if not out:
        out = ["H1"]
    try:
        cfg = load_engine_config().get("engine5_live") or {}
        blocked = {str(t).upper() for t in (cfg.get("blocked_live_timeframes") or [])}
        allowed_raw = cfg.get("allowed_live_timeframes")
        allowed = {str(t).upper() for t in allowed_raw} if allowed_raw else None
        filtered: list[str] = []
        for t in out:
            if t in blocked:
                continue
            if allowed is not None and t not in allowed:
                continue
            filtered.append(t)
        if filtered:
            return filtered
    except Exception:
        pass
    return out


def _reload_live_engine() -> Any:
    """Reload engine5 so autotrade picks up code/config without full process restart."""
    clear_config_caches()
    name = "atis.engines.engine5_live_trading"
    importlib.import_module(name)
    mod = sys.modules.get(name)
    if mod is not None and getattr(mod, "__spec__", None) is not None:
        mod = importlib.reload(mod)
    return mod


def _resolve_multi_tf_mode(
    tfs: list[str],
    cfg: dict[str, Any],
    *,
    multi_tf_independent: bool | None = None,
    fusion_mode: str | None = None,
) -> tuple[str, bool, bool]:
    """Return (label, force_independent, force_fusion)."""
    if len(tfs) <= 1:
        return "single", False, False

    req = str(fusion_mode or "").strip().lower()
    if multi_tf_independent is True or req in ("independent", "per_tf", "separate"):
        return "independent", True, False
    if multi_tf_independent is False or req in (
        "weighted_consensus",
        "majority",
        "hard_unanimous",
        "fusion",
        "fused",
    ):
        label = req if req and req not in ("fusion", "fused") else str(
            cfg.get("multi_tf_fusion_mode", "weighted_consensus")
        )
        return label, False, True

    independent = bool(cfg.get("multi_tf_independent", True)) and not bool(
        cfg.get("multi_tf_fusion", False)
    )
    if independent:
        return "independent", True, False
    return str(cfg.get("multi_tf_fusion_mode", "weighted_consensus")), False, True


@dataclass
class AutoTraderState:
    running: bool = False
    mode: str = "paper"  # paper | demo
    symbol: str = "XAUUSD"
    timeframe: str = "H1"  # primary / first selected (compat)
    timeframes: list[str] = field(default_factory=lambda: ["H1"])
    interval_seconds: int = 60
    started_at: str | None = None
    last_cycle_at: str | None = None
    cycles: int = 0
    signals: int = 0
    orders: int = 0
    last_error: str | None = None
    last_report: dict[str, Any] | None = None
    last_reports_by_tf: dict[str, Any] = field(default_factory=dict)
    fusion_mode: str = "independent"
    force_independent: bool = True
    force_fusion: bool = False
    stop_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AutoTrader:
    def __init__(self) -> None:
        self.state = AutoTraderState()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self.state.to_dict()

    def start(
        self,
        *,
        mode: str = "paper",
        interval_seconds: int = 60,
        symbol: str = "XAUUSD",
        timeframe: str | None = None,
        timeframes: list[str] | None = None,
        multi_tf_independent: bool | None = None,
        fusion_mode: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self.state.running:
                return self.state.to_dict()
            clear_config_caches()
            try:
                _reload_live_engine()
            except Exception as exc:
                logger.warning("engine5_reload_failed", error=str(exc))

            cfg = load_engine_config().get("engine5_live", {})
            if cfg.get("kill_switch"):
                raise RuntimeError("Kill switch is active — deactivate before auto-trading")

            tfs = _normalize_timeframes(timeframe=timeframe, timeframes=timeframes)
            label, force_ind, force_fus = _resolve_multi_tf_mode(
                tfs,
                cfg,
                multi_tf_independent=multi_tf_independent,
                fusion_mode=fusion_mode,
            )
            self.state = AutoTraderState(
                running=True,
                mode=mode if mode in ("paper", "demo") else "paper",
                symbol=symbol,
                timeframe=tfs[0],
                timeframes=tfs,
                interval_seconds=max(15, int(interval_seconds)),
                started_at=_utc(),
                fusion_mode=label,
                force_independent=force_ind,
                force_fusion=force_fus,
                stop_requested=False,
            )
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            logger.info(
                "autotrader_started",
                mode=self.state.mode,
                interval=self.state.interval_seconds,
                timeframes=tfs,
                multi_tf_mode=label,
                force_independent=force_ind,
                force_fusion=force_fus,
            )
            return self.state.to_dict()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self.state.stop_requested = True
            self.state.running = False
        logger.info("autotrader_stop_requested")
        return self.status()

    def _loop(self) -> None:
        while True:
            with self._lock:
                if self.state.stop_requested or not self.state.running:
                    self.state.running = False
                    break
                mode = self.state.mode
                symbol = self.state.symbol
                timeframes = list(self.state.timeframes or [self.state.timeframe])
                interval = self.state.interval_seconds
                force_independent = bool(self.state.force_independent)
                force_fusion = bool(self.state.force_fusion)
                fusion_mode = self.state.fusion_mode

            try:
                if load_engine_config().get("engine5_live", {}).get("kill_switch"):
                    with self._lock:
                        self.state.last_error = "kill_switch"
                        self.state.running = False
                        self.state.stop_requested = True
                    logger.error("autotrader_stopped_kill_switch")
                    break

                e5 = sys.modules.get("atis.engines.engine5_live_trading")
                if e5 is None:
                    e5 = _reload_live_engine()

                # Multi-TF: each selected TF analyzes and trades independently
                # unless the caller/config explicitly requests fusion.
                if len(timeframes) > 1:
                    report = e5.run_live_multi_tf(
                        [symbol],
                        timeframes,
                        dry_run=(mode == "paper"),
                        allow_ungated=True,
                        force_independent=True if force_independent else None,
                        force_fusion=True if force_fusion else None,
                    )
                else:
                    report = e5.run_live_once(
                        [symbol],
                        timeframes[0],
                        dry_run=(mode == "paper"),
                        allow_ungated=True,
                    )

                report_dict = asdict(report)
                by_tf = dict(getattr(report, "by_timeframe", None) or report_dict.get("by_timeframe") or {})
                actual_mode = str(
                    getattr(report, "multi_tf_mode", None) or report_dict.get("multi_tf_mode") or fusion_mode
                )

                orders_sent = int(report.orders_sent)
                with self._lock:
                    self.state.cycles += 1
                    self.state.last_cycle_at = _utc()
                    self.state.signals += int(report.signals)
                    self.state.orders += orders_sent
                    self.state.last_report = report_dict
                    if actual_mode and actual_mode != self.state.fusion_mode:
                        self.state.fusion_mode = actual_mode
                    self.state.last_reports_by_tf = {
                        "timeframes": timeframes,
                        "mode": actual_mode or self.state.fusion_mode,
                        "independent": (actual_mode or self.state.fusion_mode) == "independent",
                        "fusion": (actual_mode or self.state.fusion_mode) not in ("independent", "single")
                        and len(timeframes) > 1,
                        "signals": int(report.signals),
                        "orders": orders_sent,
                        "by_timeframe": by_tf,
                    }
                    # Only surface hard failures — routine pred=0 holds go to report.skips.
                    self.state.last_error = (
                        "; ".join(report.errors[:5]) if report.errors else None
                    )
                # Push open-positions cache immediately after fills (and every cycle for PnL).
                try:
                    from atis.web.position_watcher import position_watcher

                    if orders_sent > 0:
                        position_watcher.refresh_now()
                    else:
                        position_watcher.nudge()
                except Exception:
                    pass
            except Exception as exc:
                logger.exception("autotrader_cycle_failed", error=str(exc))
                with self._lock:
                    self.state.last_error = str(exc)
                    self.state.last_cycle_at = _utc()
                    self.state.cycles += 1

            slept = 0
            while slept < interval:
                with self._lock:
                    if self.state.stop_requested or not self.state.running:
                        self.state.running = False
                        return
                time.sleep(min(2, interval - slept))
                slept += 2

        with self._lock:
            self.state.running = False


autotrader = AutoTrader()
