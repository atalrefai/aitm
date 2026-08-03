"""MetaTrader 5 connector with retry/backoff and robust session management."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterable

import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from atis.config import load_engine_config, load_mt5_settings, load_timeframes
from atis.shared.logging_utils import get_logger

logger = get_logger("atis.mt5_client")

# Lazy import so unit tests can mock without requiring MT5 terminal
_mt5: Any = None

# Process-wide lock + refcount: concurrent mt5_session() must not shutdown
# while another caller is still using the terminal IPC.
_mt5_lock = threading.RLock()
_mt5_refcount = 0
_mt5_process_connected = False
_symbol_cache: dict[str, str] = {}


class MT5ConnectionError(RuntimeError):
    """Raised when MT5 initialize/login fails."""


class MT5DataError(RuntimeError):
    """Raised when rates/ticks fetch fails or returns empty unexpectedly."""


def _mt5_module() -> Any:
    global _mt5
    if _mt5 is None:
        try:
            import MetaTrader5 as mt5  # type: ignore
        except ImportError as exc:
            raise MT5ConnectionError(
                "MetaTrader5 package is not installed. "
                "Install with: pip install MetaTrader5"
            ) from exc
        _mt5 = mt5
    return _mt5


def _default_terminal_path(settings_path: str | None) -> str | None:
    if settings_path:
        return settings_path
    candidates = [
        r"C:\Program Files\Windsor Brokers MT5 Terminal\terminal64.exe",
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
    ]
    for cand in candidates:
        if Path(cand).is_file():
            return cand
    return None


def _ipc_healthy() -> bool:
    """True when the Python↔terminal bridge is alive and symbols are loaded."""
    mt5 = _mt5_module()
    try:
        terminal = mt5.terminal_info()
        account = mt5.account_info()
        if terminal is None or account is None:
            return False
        if not getattr(terminal, "connected", False):
            return False
        syms = mt5.symbols_get()
        return bool(syms)
    except Exception:
        return False

def timeframe_to_mt5(timeframe: str) -> int:
    """Map config timeframe key (e.g. H1) to MetaTrader5 constant."""
    mt5 = _mt5_module()
    tfs = load_timeframes()
    if timeframe not in tfs:
        raise KeyError(f"Unknown timeframe: {timeframe}")
    const_name = tfs[timeframe]["mt5_constant"]
    value = getattr(mt5, const_name, None)
    if value is None:
        raise AttributeError(f"MT5 has no constant {const_name}")
    return int(value)


def _retry_kwargs() -> dict[str, Any]:
    cfg = load_engine_config().get("engine1_ingestion", {})
    attempts = int(cfg.get("retry_max_attempts", 5))
    backoff = float(cfg.get("retry_backoff_seconds", 2.0))
    multiplier = float(cfg.get("retry_backoff_multiplier", 2.0))
    return {
        "stop": stop_after_attempt(attempts),
        "wait": wait_exponential(multiplier=multiplier, min=backoff, max=60),
        "retry": retry_if_exception_type((MT5ConnectionError, MT5DataError)),
        "reraise": True,
    }


class MT5Client:
    """Thin wrapper around MetaTrader5 with connection lifecycle + retries."""

    def __init__(self) -> None:
        self._held = False
        self._settings = load_mt5_settings(force_reload=True)

    def reload_settings(self) -> None:
        """Refresh credentials from secrets.env / environment."""
        self._settings = load_mt5_settings(force_reload=True)

    @property
    def connected(self) -> bool:
        return self._held and _mt5_process_connected and _ipc_healthy()

    def connect(self) -> None:
        """Acquire shared terminal connection (refcount). Safe under concurrency."""
        global _mt5_refcount, _mt5_process_connected
        self.reload_settings()
        with _mt5_lock:
            if self._held:
                if not _ipc_healthy():
                    self._force_reconnect_unlocked()
                return
            if _mt5_refcount > 0 and _ipc_healthy():
                _mt5_refcount += 1
                self._held = True
                return
            # Dead shared link or first caller
            if _mt5_refcount > 0 and not _ipc_healthy():
                logger.warning("mt5_ipc_dead_reconnecting", refcount=_mt5_refcount)
                self._shutdown_unlocked()
                _mt5_refcount = 0
            self._connect_with_retry()
            _mt5_refcount = 1
            _mt5_process_connected = True
            self._held = True

    def _force_reconnect_unlocked(self) -> None:
        global _mt5_process_connected
        logger.warning("mt5_force_reconnect")
        self._shutdown_unlocked()
        self._connect_with_retry()
        _mt5_process_connected = True
        _symbol_cache.clear()

    def _shutdown_unlocked(self) -> None:
        global _mt5_process_connected
        mt5 = _mt5_module()
        try:
            mt5.shutdown()
        except Exception:
            pass
        _mt5_process_connected = False

    @retry(**{  # type: ignore[misc]
        "stop": stop_after_attempt(5),
        "wait": wait_exponential(multiplier=2, min=2, max=60),
        "retry": retry_if_exception_type(MT5ConnectionError),
        "reraise": True,
    })
    def _connect_with_retry(self) -> None:
        mt5 = _mt5_module()
        settings = self._settings
        path = _default_terminal_path(settings.mt5_path)
        init_kwargs: dict[str, Any] = {
            "login": settings.mt5_login,
            "password": settings.mt5_password,
            "server": settings.mt5_server,
        }
        if path:
            init_kwargs["path"] = path

        logger.info(
            "mt5_connecting",
            server=settings.mt5_server,
            login=settings.mt5_login,
            path=path,
        )
        # Clear any half-dead IPC before initialize
        try:
            mt5.shutdown()
        except Exception:
            pass
        ok = mt5.initialize(**init_kwargs)
        if not ok:
            err = mt5.last_error()
            logger.error("mt5_initialize_failed", error=err)
            raise MT5ConnectionError(f"MT5 initialize failed: {err}")

        account = mt5.account_info()
        if account is None:
            err = mt5.last_error()
            mt5.shutdown()
            raise MT5ConnectionError(f"MT5 account_info failed: {err}")

        if int(account.login) != int(settings.mt5_login):
            authorized = mt5.login(
                settings.mt5_login,
                password=settings.mt5_password,
                server=settings.mt5_server,
            )
            if not authorized:
                err = mt5.last_error()
                mt5.shutdown()
                raise MT5ConnectionError(f"MT5 login failed: {err}")
            account = mt5.account_info()

        # Wait briefly for symbol catalog if terminal just attached
        if not _ipc_healthy():
            import time

            for _ in range(10):
                time.sleep(0.2)
                if _ipc_healthy():
                    break
        if not _ipc_healthy():
            mt5.shutdown()
            raise MT5ConnectionError("MT5 connected but symbol catalog empty")

        logger.info(
            "mt5_connected",
            login=getattr(account, "login", None),
            server=getattr(account, "server", None),
            balance=getattr(account, "balance", None),
            currency=getattr(account, "currency", None),
        )

    def disconnect(self) -> None:
        """Release shared connection; shutdown only when last holder leaves."""
        global _mt5_refcount, _mt5_process_connected
        with _mt5_lock:
            if not self._held:
                return
            self._held = False
            _mt5_refcount = max(0, _mt5_refcount - 1)
            if _mt5_refcount == 0:
                self._shutdown_unlocked()
                _symbol_cache.clear()
                logger.info("mt5_disconnected")
            else:
                logger.info("mt5_release", remaining=_mt5_refcount)

    def ensure_symbol(self, symbol: str) -> None:
        """Select symbol in Market Watch so history can be fetched."""
        resolved = self.resolve_symbol(symbol)
        mt5 = _mt5_module()
        with _mt5_lock:
            info = mt5.symbol_info(resolved)
            if info is None:
                raise MT5DataError(f"Symbol not found in MT5: {resolved}")
            if not info.visible and not mt5.symbol_select(info.name, True):
                raise MT5DataError(f"Failed to select symbol: {info.name}")

    def resolve_symbol(self, symbol: str) -> str:
        """Return the broker-specific symbol name (handles suffixes like XAUUSD@)."""
        self._require_connected()
        key = symbol.strip().upper()
        with _mt5_lock:
            if not _ipc_healthy():
                self._force_reconnect_unlocked()
            cached = _symbol_cache.get(key)
            if cached:
                info = _mt5_module().symbol_info(cached)
                if info is not None:
                    if not info.visible:
                        _mt5_module().symbol_select(info.name, True)
                    return info.name
            try:
                resolved = self._resolve_symbol_unlocked(symbol)
            except MT5DataError:
                # One reconnect retry — covers IPC killed by another process
                self._force_reconnect_unlocked()
                resolved = self._resolve_symbol_unlocked(symbol)
            _symbol_cache[key] = resolved
            return resolved

    def _resolve_symbol_unlocked(self, symbol: str) -> str:
        mt5 = _mt5_module()
        base = (
            symbol.strip()
            .upper()
            .lstrip("@")
            .replace("/", "")
            .replace(".", "")
            .rstrip("@")
        )

        # 0) Explicit map from config (most reliable for known brokers)
        try:
            mapping = load_engine_config().get("trading", {}).get("broker_symbol_map") or {}
            mapped = mapping.get(base) or mapping.get(symbol.strip().upper())
            if mapped:
                info = mt5.symbol_info(mapped)
                if info is None:
                    mt5.symbol_select(mapped, True)
                    info = mt5.symbol_info(mapped)
                if info is not None:
                    if not info.visible:
                        mt5.symbol_select(info.name, True)
                    return info.name
        except Exception:
            pass

        # 1) Try known candidates — prefer broker suffix "@" early (Windsor)
        for candidate in self._symbol_candidates(symbol):
            info = mt5.symbol_info(candidate)
            if info is None:
                mt5.symbol_select(candidate, True)
                info = mt5.symbol_info(candidate)
            if info is not None:
                if not info.visible:
                    mt5.symbol_select(info.name, True)
                return info.name

        # 2) Scan terminal symbol list for exact / prefixed match
        all_symbols = mt5.symbols_get()
        if all_symbols:
            names = [s.name for s in all_symbols]
            preferred = [f"{base}@", base, f"{base}.", f"{base}m", f"@{base}"]
            for cand in preferred:
                if cand in names:
                    mt5.symbol_select(cand, True)
                    return cand
            for name in names:
                n = name.upper().lstrip("@")
                if n == base or n == f"{base}@":
                    mt5.symbol_select(name, True)
                    return name
                if n.startswith(base) and len(name) <= len(base) + 3:
                    mt5.symbol_select(name, True)
                    return name

        raise MT5DataError(
            f"Symbol not found in MT5: {symbol} (tried {base}, {base}@, and market scan)"
        )

    @staticmethod
    def _symbol_candidates(symbol: str) -> list[str]:
        raw = symbol.strip()
        base = raw.upper().lstrip("@").replace("/", "").replace(".", "").rstrip("@")
        # Windsor Brokers uses trailing "@" (e.g. XAUUSD@) — try it first
        return [
            f"{base}@",
            raw,
            base,
            f"@{base}",
            f"{base}.",
            f"{base}m",
            f"{base}.m",
            f"{base}i",
            f"{base}.i",
            f"{base}pro",
            f"{base}#",
        ]

    def copy_rates_range(
        self,
        symbol: str,
        timeframe: str,
        date_from: datetime,
        date_to: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV between two datetimes (inclusive range as MT5 defines)."""
        self._require_connected()
        return self._copy_rates_range_retry(symbol, timeframe, date_from, date_to)

    def copy_rates_from(
        self,
        symbol: str,
        timeframe: str,
        date_from: datetime,
        count: int,
    ) -> pd.DataFrame:
        self._require_connected()
        return self._copy_rates_from_retry(symbol, timeframe, date_from, count)

    def copy_rates_from_pos(
        self,
        symbol: str,
        timeframe: str,
        start_pos: int,
        count: int,
    ) -> pd.DataFrame:
        self._require_connected()
        return self._copy_rates_from_pos_retry(symbol, timeframe, start_pos, count)

    def account_summary(self) -> dict[str, Any]:
        self._require_connected()
        mt5 = _mt5_module()
        info = mt5.account_info()
        if info is None:
            raise MT5ConnectionError(f"account_info failed: {mt5.last_error()}")
        return info._asdict()

    def terminal_info(self) -> dict[str, Any]:
        self._require_connected()
        mt5 = _mt5_module()
        info = mt5.terminal_info()
        if info is None:
            raise MT5ConnectionError(f"terminal_info failed: {mt5.last_error()}")
        return info._asdict()

    def _require_connected(self) -> None:
        if not self._held:
            raise MT5ConnectionError("MT5Client is not connected. Call connect() first.")
        with _mt5_lock:
            if not _ipc_healthy():
                self._force_reconnect_unlocked()

    def _rates_to_df(self, rates: Any, symbol: str, timeframe: str) -> pd.DataFrame:
        if rates is None or len(rates) == 0:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
            )
        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(
            columns={
                "tick_volume": "tick_volume",
                "real_volume": "real_volume",
            }
        )
        df["symbol"] = symbol
        df["timeframe"] = timeframe
        cols = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
            "symbol",
            "timeframe",
        ]
        df = df[cols].sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        df = df.reset_index(drop=True)
        return df

    def _normalize_dt(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @retry(**{  # type: ignore[misc]
        "stop": stop_after_attempt(5),
        "wait": wait_exponential(multiplier=2, min=2, max=60),
        "retry": retry_if_exception_type(MT5DataError),
        "reraise": True,
    })
    def _copy_rates_range_retry(
        self,
        symbol: str,
        timeframe: str,
        date_from: datetime,
        date_to: datetime,
    ) -> pd.DataFrame:
        mt5 = _mt5_module()
        resolved = self.resolve_symbol(symbol)
        tf = timeframe_to_mt5(timeframe)
        date_from = self._normalize_dt(date_from)
        date_to = self._normalize_dt(date_to)
        # MT5 Python API expects naive UTC datetimes for rates range
        rates = mt5.copy_rates_range(
            resolved,
            tf,
            date_from.replace(tzinfo=None),
            date_to.replace(tzinfo=None),
        )
        if rates is None:
            err = mt5.last_error()
            # (-2, 'Terminal: Invalid params') / empty history sometimes returns None
            logger.warning(
                "mt5_copy_rates_range_none",
                symbol=resolved,
                timeframe=timeframe,
                error=err,
                date_from=str(date_from),
                date_to=str(date_to),
            )
            return self._rates_to_df([], symbol, timeframe)
        return self._rates_to_df(rates, symbol, timeframe)

    @retry(**{  # type: ignore[misc]
        "stop": stop_after_attempt(5),
        "wait": wait_exponential(multiplier=2, min=2, max=60),
        "retry": retry_if_exception_type(MT5DataError),
        "reraise": True,
    })
    def _copy_rates_from_retry(
        self,
        symbol: str,
        timeframe: str,
        date_from: datetime,
        count: int,
    ) -> pd.DataFrame:
        mt5 = _mt5_module()
        resolved = self.resolve_symbol(symbol)
        tf = timeframe_to_mt5(timeframe)
        date_from = self._normalize_dt(date_from)
        rates = mt5.copy_rates_from(
            resolved,
            tf,
            date_from.replace(tzinfo=None),
            count,
        )
        if rates is None:
            err = mt5.last_error()
            logger.warning(
                "mt5_copy_rates_from_none",
                symbol=resolved,
                timeframe=timeframe,
                error=err,
            )
            return self._rates_to_df([], symbol, timeframe)
        return self._rates_to_df(rates, symbol, timeframe)

    @retry(**{  # type: ignore[misc]
        "stop": stop_after_attempt(5),
        "wait": wait_exponential(multiplier=2, min=2, max=60),
        "retry": retry_if_exception_type(MT5DataError),
        "reraise": True,
    })
    def _copy_rates_from_pos_retry(
        self,
        symbol: str,
        timeframe: str,
        start_pos: int,
        count: int,
    ) -> pd.DataFrame:
        mt5 = _mt5_module()
        resolved = self.resolve_symbol(symbol)
        tf = timeframe_to_mt5(timeframe)
        rates = mt5.copy_rates_from_pos(resolved, tf, start_pos, count)
        if rates is None:
            err = mt5.last_error()
            logger.warning(
                "mt5_copy_rates_from_pos_none",
                symbol=resolved,
                timeframe=timeframe,
                error=err,
            )
            return self._rates_to_df([], symbol, timeframe)
        return self._rates_to_df(rates, symbol, timeframe)


@contextmanager
def mt5_session() -> Generator[MT5Client, None, None]:
    """Context manager that connects and always disconnects."""
    client = MT5Client()
    client.connect()
    try:
        yield client
    finally:
        client.disconnect()


def ping_mt5() -> dict[str, Any]:
    """Connectivity smoke test — used in Phase 1 verification."""
    with mt5_session() as client:
        account = client.account_summary()
        terminal = client.terminal_info()
        return {
            "ok": True,
            "login": account.get("login"),
            "server": account.get("server"),
            "balance": account.get("balance"),
            "currency": account.get("currency"),
            "trade_allowed": account.get("trade_allowed"),
            "terminal_connected": terminal.get("connected"),
            "terminal_name": terminal.get("name"),
        }
