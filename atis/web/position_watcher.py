"""Background watcher: keep open MT5 positions fresh for the desk UI."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Callable

from atis.engines.engine5_live_trading import close_position, list_open_positions
from atis.shared.logging_utils import get_logger
from atis.shared.mt5_client import mt5_session

logger = get_logger("atis.position_watcher")

DEFAULT_AUTO_CLOSE_MIN_PROFIT = 5.0
DEFAULT_AUTO_CLOSE_MAX_LOSS = 0.30


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(rows: list[dict[str, Any]]) -> tuple[Any, ...]:
    """Cheap change detector: tickets + rounded live PnL + volume."""
    items: list[tuple[Any, ...]] = []
    for p in rows:
        items.append(
            (
                int(p.get("ticket", 0) or 0),
                round(float(p.get("net_profit", 0) or 0), 2),
                round(float(p.get("volume", 0) or 0), 2),
                round(float(p.get("price_current", 0) or 0), 3),
            )
        )
    items.sort(key=lambda x: x[0])
    return tuple(items)


def _summarize(rows: list[dict[str, Any]], *, symbol: str, atis_only: bool) -> dict[str, Any]:
    winners = sum(1 for p in rows if float(p.get("net_profit", 0) or 0) > 0)
    losers = sum(1 for p in rows if float(p.get("net_profit", 0) or 0) < 0)
    total_pnl = sum(float(p.get("net_profit", 0) or 0) for p in rows)
    return {
        "positions": rows,
        "count": len(rows),
        "winners": winners,
        "losers": losers,
        "total_pnl": total_pnl,
        "symbol": symbol,
        "atis_only": atis_only,
    }


class PositionWatcher:
    """
    Daemon thread that polls MT5 open positions and caches a snapshot.

    API handlers read the cache (instant); trading/close paths call
    ``refresh_now()`` so the UI sees fills/closes without waiting for poll.

    Also reconciles vanished tickets into the winning-trade training store
    (SL/TP / external closes).

    Optional auto-close: when enabled, closes open positions whose live
    net profit exceeds the profit threshold, and/or whose loss exceeds
    the max-loss threshold.
    """

    def __init__(
        self,
        *,
        symbol_provider: Callable[[], str],
        poll_interval_sec: float = 0.5,
        atis_only: bool = True,
    ) -> None:
        self._symbol_provider = symbol_provider
        self._poll_interval_sec = max(0.2, float(poll_interval_sec))
        self._atis_only = bool(atis_only)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._kick = threading.Event()
        self._seq = 0
        self._fp: tuple[Any, ...] | None = None
        self._known_tickets: set[int] = set()
        self._auto_close_enabled = False
        self._auto_close_min_profit = float(DEFAULT_AUTO_CLOSE_MIN_PROFIT)
        self._auto_close_loss_enabled = False
        self._auto_close_max_loss = float(DEFAULT_AUTO_CLOSE_MAX_LOSS)
        self._auto_closing: set[int] = set()
        self._snapshot: dict[str, Any] = {
            "positions": [],
            "count": 0,
            "winners": 0,
            "losers": 0,
            "total_pnl": 0.0,
            "symbol": "",
            "atis_only": self._atis_only,
            "seq": 0,
            "updated_at": None,
            "error": None,
            "source": "watcher",
            "live": True,
            "auto_close": {
                "enabled": False,
                "min_profit": float(DEFAULT_AUTO_CLOSE_MIN_PROFIT),
                "loss_enabled": False,
                "max_loss": float(DEFAULT_AUTO_CLOSE_MAX_LOSS),
            },
        }
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="atis-position-watcher",
            daemon=True,
        )
        self._started = True
        self._thread.start()
        logger.info("position_watcher_started", interval=self._poll_interval_sec)

    def stop(self) -> None:
        self._stop.set()
        self._kick.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=3.0)
        self._thread = None
        self._started = False
        logger.info("position_watcher_stopped")

    def set_symbol_provider(self, provider: Callable[[], str]) -> None:
        self._symbol_provider = provider

    def _auto_close_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self._auto_close_enabled),
            "min_profit": float(self._auto_close_min_profit),
            "loss_enabled": bool(self._auto_close_loss_enabled),
            "max_loss": float(self._auto_close_max_loss),
        }

    def auto_close_settings(self) -> dict[str, Any]:
        with self._lock:
            return self._auto_close_dict()

    def set_auto_close(
        self,
        *,
        enabled: bool | None = None,
        min_profit: float | None = None,
        loss_enabled: bool | None = None,
        max_loss: float | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if enabled is not None:
                self._auto_close_enabled = bool(enabled)
            if min_profit is not None:
                try:
                    value = float(min_profit)
                except (TypeError, ValueError) as exc:
                    raise ValueError("min_profit must be a number") from exc
                if value <= 0:
                    raise ValueError("min_profit must be > 0")
                self._auto_close_min_profit = value
            if loss_enabled is not None:
                self._auto_close_loss_enabled = bool(loss_enabled)
            if max_loss is not None:
                try:
                    loss_value = float(max_loss)
                except (TypeError, ValueError) as exc:
                    raise ValueError("max_loss must be a number") from exc
                if loss_value <= 0:
                    raise ValueError("max_loss must be > 0")
                self._auto_close_max_loss = loss_value
            settings = self._auto_close_dict()
            snap = dict(self._snapshot)
            snap["auto_close"] = settings
            self._snapshot = snap
        self._kick.set()
        return settings

    def nudge(self) -> None:
        """Wake the poll loop without blocking the caller."""
        self._kick.set()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            out = dict(self._snapshot)
            out["auto_close"] = self._auto_close_dict()
            return out

    def refresh_now(self) -> dict[str, Any]:
        """Force an immediate poll (also wakes the background loop)."""
        self._poll_once()
        self._kick.set()
        return self.snapshot()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception as exc:
                logger.warning("position_watcher_tick_failed", error=str(exc))
            self._kick.wait(timeout=self._poll_interval_sec)
            self._kick.clear()

    def _reconcile_closes(self, current_tickets: set[int]) -> None:
        previous = set(self._known_tickets)
        if not previous:
            self._known_tickets = set(current_tickets)
            return
        missing = previous - current_tickets
        if not missing:
            self._known_tickets = set(current_tickets)
            return
        finalized: set[int] = set()
        try:
            from atis.shared.winning_trade_store import reconcile_missing_tickets

            closed = reconcile_missing_tickets(
                previous,
                current_tickets,
                close_reason="broker_exit",
            )
            for c in closed or []:
                try:
                    finalized.add(int(c.get("ticket")))
                except (TypeError, ValueError):
                    pass
            if closed:
                logger.info(
                    "position_watcher_reconciled_closes",
                    count=len(closed),
                    winners=sum(1 for c in closed if c.get("is_winner")),
                )
        except Exception as exc:
            logger.warning("position_watcher_reconcile_failed", error=str(exc))
        # Keep vanished tickets that are not finalized yet so we retry deal history.
        pending = missing - finalized
        self._known_tickets = set(current_tickets) | pending

    def _auto_close_eligible(
        self, rows: list[dict[str, Any]]
    ) -> list[tuple[int, float, str]]:
        with self._lock:
            profit_on = bool(self._auto_close_enabled)
            profit_thr = float(self._auto_close_min_profit)
            loss_on = bool(self._auto_close_loss_enabled)
            loss_thr = float(self._auto_close_max_loss)
            busy = set(self._auto_closing)
        if not profit_on and not loss_on:
            return []
        eligible: list[tuple[int, float, str]] = []
        for p in rows:
            try:
                ticket = int(p.get("ticket", 0) or 0)
                net = float(p.get("net_profit", p.get("profit", 0)) or 0)
            except (TypeError, ValueError):
                continue
            if not ticket or ticket in busy:
                continue
            if profit_on and net > profit_thr:
                eligible.append((ticket, net, "profit"))
            elif loss_on and net < -loss_thr:
                eligible.append((ticket, net, "loss"))
        return eligible

    def _run_auto_close(self, client: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Close positions that crossed profit or loss thresholds. Mutates ``rows``."""
        eligible = self._auto_close_eligible(rows)
        if not eligible:
            return []
        closed: list[dict[str, Any]] = []
        for ticket, net, kind in eligible:
            with self._lock:
                if ticket in self._auto_closing:
                    continue
                self._auto_closing.add(ticket)
            comment = "ATIS auto-close" if kind == "profit" else "ATIS auto-stop"
            try:
                result = close_position(
                    client,
                    ticket,
                    comment=comment,
                )
                if result.get("ok"):
                    closed.append(
                        {
                            "ticket": ticket,
                            "net_profit": float(result.get("net_profit", net) or net),
                            "reason": f"auto_close_{kind}",
                        }
                    )
                    rows[:] = [
                        p for p in rows if int(p.get("ticket", 0) or 0) != ticket
                    ]
                    settings = self.auto_close_settings()
                    logger.info(
                        "position_auto_closed",
                        ticket=ticket,
                        net_profit=net,
                        kind=kind,
                        min_profit=settings["min_profit"],
                        max_loss=settings["max_loss"],
                    )
                else:
                    logger.warning(
                        "position_auto_close_failed",
                        ticket=ticket,
                        kind=kind,
                        retcode=result.get("retcode"),
                        comment=result.get("comment"),
                    )
            except Exception as exc:
                logger.warning(
                    "position_auto_close_error",
                    ticket=ticket,
                    kind=kind,
                    error=str(exc),
                )
            finally:
                with self._lock:
                    self._auto_closing.discard(ticket)
        return closed

    def _poll_once(self) -> None:
        try:
            symbol = str(self._symbol_provider() or "").strip() or "XAUUSD"
        except Exception:
            symbol = "XAUUSD"
        error: str | None = None
        rows: list[dict[str, Any]] = []
        try:
            # Reconcile MUST run inside the same MT5 session. fetch_deal_pnl uses the
            # global MetaTrader5 IPC bridge — after disconnect it returns None and old
            # code paths finalized winners as net_profit=0.
            with mt5_session() as client:
                rows = list_open_positions(client, symbol, atis_only=self._atis_only)
                self._run_auto_close(client, rows)
                current_tickets = {
                    int(p.get("ticket", 0) or 0)
                    for p in rows
                    if int(p.get("ticket", 0) or 0)
                }
                self._reconcile_closes(current_tickets)
        except Exception as exc:
            error = str(exc)
            logger.warning("position_watcher_mt5_failed", error=error, symbol=symbol)

        payload = _summarize(rows, symbol=symbol, atis_only=self._atis_only)
        fp = _fingerprint(rows) if error is None else ("error", error)
        with self._lock:
            changed = fp != self._fp or error != self._snapshot.get("error")
            # Always advance seq while positions are open so the live UI
            # receives price/PnL ticks even when fingerprint rounding matches.
            force_tick = bool(rows) and error is None
            if changed or force_tick:
                self._seq += 1
                self._fp = fp
            payload["seq"] = self._seq
            payload["updated_at"] = _utc()
            payload["error"] = error
            payload["source"] = "watcher"
            payload["live"] = True
            payload["auto_close"] = self._auto_close_dict()
            self._snapshot = payload


position_watcher = PositionWatcher(
    symbol_provider=lambda: "XAUUSD",
    poll_interval_sec=0.35,
)
