"""Background auto-trading loop for Gold Desk (paper/demo)."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from atis.config import load_engine_config
from atis.engines.engine5_live_trading import run_live_multi_tf, run_live_once
from atis.shared.logging_utils import get_logger

logger = get_logger("atis.autotrader")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_timeframes(
    timeframe: str | None = None,
    timeframes: list[str] | None = None,
) -> list[str]:
    """Resolve one or many TFs into a unique ordered list."""
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
    return out


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
    fusion_mode: str = "weighted_consensus"
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
    ) -> dict[str, Any]:
        with self._lock:
            if self.state.running:
                return self.state.to_dict()
            cfg = load_engine_config().get("engine5_live", {})
            if cfg.get("kill_switch"):
                raise RuntimeError("Kill switch is active — deactivate before auto-trading")

            tfs = _normalize_timeframes(timeframe=timeframe, timeframes=timeframes)
            independent = bool(cfg.get("multi_tf_independent", True)) and not bool(
                cfg.get("multi_tf_fusion", False)
            )
            fusion_label = (
                "independent"
                if (len(tfs) > 1 and independent)
                else (
                    str(cfg.get("multi_tf_fusion_mode", "weighted_consensus"))
                    if len(tfs) > 1
                    else "single"
                )
            )
            self.state = AutoTraderState(
                running=True,
                mode=mode if mode in ("paper", "demo") else "paper",
                symbol=symbol,
                timeframe=tfs[0],
                timeframes=tfs,
                interval_seconds=max(15, int(interval_seconds)),
                started_at=_utc(),
                fusion_mode=fusion_label,
                stop_requested=False,
            )
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            logger.info(
                "autotrader_started",
                mode=self.state.mode,
                interval=self.state.interval_seconds,
                timeframes=tfs,
                multi_tf_mode=fusion_label,
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

            try:
                if load_engine_config().get("engine5_live", {}).get("kill_switch"):
                    with self._lock:
                        self.state.last_error = "kill_switch"
                        self.state.running = False
                        self.state.stop_requested = True
                    logger.error("autotrader_stopped_kill_switch")
                    break

                # Multi-TF: each selected TF analyzes and trades independently
                # (unless multi_tf_fusion is enabled in config).
                # Single TF: that TF's trained model only.
                if len(timeframes) > 1:
                    report = run_live_multi_tf(
                        [symbol],
                        timeframes,
                        dry_run=(mode == "paper"),
                        allow_ungated=True,
                    )
                else:
                    report = run_live_once(
                        [symbol],
                        timeframes[0],
                        dry_run=(mode == "paper"),
                        allow_ungated=True,
                    )

                with self._lock:
                    self.state.cycles += 1
                    self.state.last_cycle_at = _utc()
                    self.state.signals += int(report.signals)
                    self.state.orders += int(report.orders_sent)
                    self.state.last_report = asdict(report)
                    self.state.last_reports_by_tf = {
                        "timeframes": timeframes,
                        "mode": self.state.fusion_mode,
                        "independent": self.state.fusion_mode == "independent",
                        "fusion": self.state.fusion_mode not in ("independent", "single")
                        and len(timeframes) > 1,
                        "signals": int(report.signals),
                        "orders": int(report.orders_sent),
                    }
                    self.state.last_error = "; ".join(report.errors[:5]) if report.errors else None
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
