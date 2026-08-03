"""Async file bridge: Python → MT5 Common Files for chart drawing."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from atis.config import PROJECT_ROOT
from atis.shared.logging_utils import get_logger
from atis.shared.mt5_pattern_overlay.models import LEGEND_ENTRIES, PatternOverlay

logger = get_logger("atis.mt5_overlay.bridge")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def local_overlay_dir() -> Path:
    path = PROJECT_ROOT / "logs" / "live" / "mt5_overlay"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_mt5_common_files_dir(terminal_info: dict[str, Any] | None) -> Path | None:
    """Return MT5 Common\\Files directory when terminal info is available."""
    if not terminal_info:
        return None
    common = terminal_info.get("commondata_path") or terminal_info.get("common_data_path")
    if not common:
        return None
    files = Path(str(common)) / "Files" / "ATIS"
    try:
        files.mkdir(parents=True, exist_ok=True)
        return files
    except OSError as exc:
        logger.warning("mt5_common_files_unavailable", error=str(exc), path=str(files))
        return None


def resolve_mt5_terminal_files_dir(terminal_info: dict[str, Any] | None) -> Path | None:
    """Return terminal MQL5\\Files\\ATIS when available (non-common)."""
    if not terminal_info:
        return None
    data = terminal_info.get("data_path")
    if not data:
        return None
    files = Path(str(data)) / "MQL5" / "Files" / "ATIS"
    try:
        files.mkdir(parents=True, exist_ok=True)
        return files
    except OSError as exc:
        logger.warning("mt5_terminal_files_unavailable", error=str(exc), path=str(files))
        return None


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


@dataclass
class OverlaySnapshot:
    seq: int
    symbol: str
    broker_symbol: str
    timeframe: str
    patterns: list[PatternOverlay]
    extra: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": 1,
            "seq": int(self.seq),
            "updated_at": _utc_now_iso(),
            "symbol": self.symbol,
            "broker_symbol": self.broker_symbol,
            "timeframe": self.timeframe,
            "legend": LEGEND_ENTRIES,
            "patterns": [p.to_dict() for p in self.patterns],
            "extra": self.extra or {},
        }


class AsyncOverlayBridge:
    """
    Non-blocking writer for MT5 overlay state.

    Trading threads call ``publish()``; a daemon worker serializes to disk
    so chart I/O never blocks order latency.
    """

    def __init__(
        self,
        *,
        terminal_info_provider: Callable[[], dict[str, Any] | None] | None = None,
        poll_idle_sec: float = 0.05,
    ) -> None:
        self._q: queue.Queue[OverlaySnapshot | None] = queue.Queue(maxsize=8)
        self._terminal_info_provider = terminal_info_provider
        self._poll_idle_sec = poll_idle_sec
        self._seq = 0
        self._lock = threading.Lock()
        self._last_payload: dict[str, Any] | None = None
        self._thread = threading.Thread(
            target=self._worker,
            name="atis-mt5-overlay-bridge",
            daemon=True,
        )
        self._started = False
        self._stop = threading.Event()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._stop.clear()
            self._thread.start()
            logger.info("overlay_bridge_started")

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def publish(self, snapshot: OverlaySnapshot, *, block: bool = False) -> bool:
        """Enqueue snapshot. Drops oldest if queue is full (prefer freshest state)."""
        self.start()
        try:
            if block:
                self._q.put(snapshot, timeout=0.2)
            else:
                self._q.put_nowait(snapshot)
            return True
        except queue.Full:
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(snapshot)
                return True
            except queue.Full:
                logger.warning("overlay_publish_dropped", symbol=snapshot.symbol)
                return False

    def write_sync(self, snapshot: OverlaySnapshot) -> list[str]:
        """Synchronous write (tests / forced flush)."""
        payload = snapshot.to_payload()
        return self._write_payload(payload)

    def last_payload(self) -> dict[str, Any] | None:
        return self._last_payload

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=self._poll_idle_sec)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                payload = item.to_payload()
                paths = self._write_payload(payload)
                logger.debug(
                    "overlay_written",
                    seq=payload.get("seq"),
                    patterns=len(payload.get("patterns") or []),
                    paths=paths,
                )
            except Exception as exc:
                logger.warning("overlay_write_failed", error=str(exc))
            finally:
                self._q.task_done()

    def _write_payload(self, payload: dict[str, Any]) -> list[str]:
        written: list[str] = []
        local = local_overlay_dir() / "overlay_state.json"
        atomic_write_json(local, payload)
        written.append(str(local))

        # Also keep a per-symbol/tf copy for history UI / debugging
        sym = str(payload.get("symbol") or "UNK")
        tf = str(payload.get("timeframe") or "UNK")
        named = local_overlay_dir() / f"overlay_{sym}_{tf}.json"
        atomic_write_json(named, payload)
        written.append(str(named))

        info = None
        if self._terminal_info_provider:
            try:
                info = self._terminal_info_provider()
            except Exception as exc:
                logger.debug("terminal_info_provider_failed", error=str(exc))

        for folder in (
            resolve_mt5_common_files_dir(info),
            resolve_mt5_terminal_files_dir(info),
        ):
            if folder is None:
                continue
            target = folder / "overlay_state.json"
            try:
                atomic_write_json(target, payload)
                written.append(str(target))
            except OSError as exc:
                logger.warning("overlay_mt5_write_failed", path=str(target), error=str(exc))

        self._last_payload = payload
        # Small yield so MT5 FILE_READ can open between rapid publishes
        time.sleep(0.001)
        return written
