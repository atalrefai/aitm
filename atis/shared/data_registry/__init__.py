"""Data state registry — JSON metadata for incremental updates (Principle 1.1).

Each timeframe is stored in its own file under the registry root:

    data/registry/M5.json
    data/registry/H1.json
    ...

Audit events without a timeframe go to ``audit.json``.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from atis.config import get_path

_LOCK = threading.RLock()


@dataclass(frozen=True)
class RegistryRow:
    symbol: str
    timeframe: str
    layer: str
    first_available_ts: str | None
    last_updated_ts: str | None
    last_run_status: str | None
    last_run_at: str | None
    row_count: int
    checksum: str | None
    version_hash: str | None
    extra_json: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_checksum(values: Iterable[Any]) -> str:
    """Stable checksum over an iterable of stringifiable values."""
    h = hashlib.sha256()
    for v in values:
        h.update(str(v).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:32]


def _iso(v: datetime | str | None) -> str | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.isoformat()
    return str(v)


def _safe_tf_name(timeframe: str) -> str:
    """Sanitize timeframe for use as a filename stem."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in timeframe.strip())
    return cleaned or "_unknown"


class DataStateRegistry:
    """JSON-backed registry: one file per timeframe for incremental pipeline state."""

    def __init__(self, root_dir: Path | None = None) -> None:
        # Accept legacy ``.db`` paths from config/tests and use their parent directory.
        path = Path(root_dir) if root_dir is not None else get_path("data_registry")
        legacy_db: Path | None = None
        if path.suffix.lower() == ".db":
            legacy_db = path
            path = path.parent
        self.root_dir = path
        self.root_dir.mkdir(parents=True, exist_ok=True)
        # One-time import from previous SQLite registry if JSON state is still empty.
        candidate = legacy_db or (self.root_dir / "data_state_registry.db")
        if candidate.exists() and not any(
            p.name.lower() != "audit.json" for p in self.root_dir.glob("*.json")
        ):
            self._migrate_from_sqlite(candidate)

    def _migrate_from_sqlite(self, db_path: Path) -> None:
        """Import rows/audit from the old SQLite registry into per-timeframe JSON."""
        import sqlite3

        try:
            con = sqlite3.connect(str(db_path))
            con.row_factory = sqlite3.Row
        except Exception:
            return
        try:
            try:
                state_rows = con.execute("SELECT * FROM data_state_registry").fetchall()
            except sqlite3.Error:
                state_rows = []
            by_tf: dict[str, dict[str, Any]] = {}
            for row in state_rows:
                tf = str(row["timeframe"])
                doc = by_tf.setdefault(tf, self._empty_tf_doc(tf))
                symbols = doc.setdefault("symbols", {})
                sym = symbols.setdefault(str(row["symbol"]), {})
                sym[str(row["layer"])] = {
                    "first_available_ts": row["first_available_ts"],
                    "last_updated_ts": row["last_updated_ts"],
                    "last_run_status": row["last_run_status"],
                    "last_run_at": row["last_run_at"],
                    "row_count": int(row["row_count"] or 0),
                    "checksum": row["checksum"],
                    "version_hash": row["version_hash"],
                    "extra_json": row["extra_json"],
                }
            for tf, doc in by_tf.items():
                self._save_tf(tf, doc)

            try:
                audit_rows = con.execute(
                    "SELECT ts, engine, event, symbol, timeframe, detail_json FROM audit_trail ORDER BY id"
                ).fetchall()
            except sqlite3.Error:
                audit_rows = []
            global_audit: list[dict[str, Any]] = []
            for row in audit_rows:
                entry = {
                    "ts": row["ts"],
                    "engine": row["engine"],
                    "event": row["event"],
                    "symbol": row["symbol"],
                    "timeframe": row["timeframe"],
                    "detail_json": row["detail_json"],
                }
                tf = row["timeframe"]
                if tf:
                    doc = self._load_tf(str(tf))
                    audit_list = doc.setdefault("audit", [])
                    if not isinstance(audit_list, list):
                        audit_list = []
                    audit_list.append(entry)
                    doc["audit"] = audit_list[-2000:]
                    self._save_tf(str(tf), doc)
                else:
                    global_audit.append(entry)
            if global_audit:
                self._write_json(self._audit_path(), global_audit[-5000:])
        finally:
            con.close()

    def _tf_path(self, timeframe: str) -> Path:
        return self.root_dir / f"{_safe_tf_name(timeframe)}.json"

    def _audit_path(self) -> Path:
        return self.root_dir / "audit.json"

    def _empty_tf_doc(self, timeframe: str) -> dict[str, Any]:
        return {
            "timeframe": timeframe,
            "updated_at": _utc_now_iso(),
            "symbols": {},
            "audit": [],
        }

    def _read_json(self, path: Path) -> dict[str, Any] | list[Any] | None:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write_json(self, path: Path, payload: dict[str, Any] | list[Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=False)
            fh.write("\n")
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                tmp.replace(path)
                return
            except OSError as exc:
                last_err = exc
                time.sleep(0.05 * (attempt + 1))
                try:
                    if path.exists():
                        path.unlink()
                    tmp.replace(path)
                    return
                except OSError as exc2:
                    last_err = exc2
        # Fallback: write directly if replace remains locked (OneDrive).
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=False)
                fh.write("\n")
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            return
        except OSError:
            if last_err is not None:
                raise last_err
            raise

    def _load_tf(self, timeframe: str) -> dict[str, Any]:
        path = self._tf_path(timeframe)
        raw = self._read_json(path)
        if not isinstance(raw, dict):
            return self._empty_tf_doc(timeframe)
        raw.setdefault("timeframe", timeframe)
        raw.setdefault("symbols", {})
        raw.setdefault("audit", [])
        if not isinstance(raw["symbols"], dict):
            raw["symbols"] = {}
        if not isinstance(raw["audit"], list):
            raw["audit"] = []
        return raw

    def _save_tf(self, timeframe: str, doc: dict[str, Any]) -> None:
        doc["timeframe"] = timeframe
        doc["updated_at"] = _utc_now_iso()
        self._write_json(self._tf_path(timeframe), doc)

    def _row_from_layer(
        self,
        symbol: str,
        timeframe: str,
        layer: str,
        data: dict[str, Any],
    ) -> RegistryRow:
        return RegistryRow(
            symbol=symbol,
            timeframe=timeframe,
            layer=layer,
            first_available_ts=data.get("first_available_ts"),
            last_updated_ts=data.get("last_updated_ts"),
            last_run_status=data.get("last_run_status"),
            last_run_at=data.get("last_run_at"),
            row_count=int(data.get("row_count") or 0),
            checksum=data.get("checksum"),
            version_hash=data.get("version_hash"),
            extra_json=data.get("extra_json"),
        )

    def get(
        self,
        symbol: str,
        timeframe: str,
        layer: str = "raw",
    ) -> RegistryRow | None:
        with _LOCK:
            doc = self._load_tf(timeframe)
            sym = doc["symbols"].get(symbol)
            if not isinstance(sym, dict):
                return None
            layer_data = sym.get(layer)
            if not isinstance(layer_data, dict):
                return None
            return self._row_from_layer(symbol, timeframe, layer, layer_data)

    def last_updated_ts(
        self,
        symbol: str,
        timeframe: str,
        layer: str = "raw",
    ) -> datetime | None:
        row = self.get(symbol, timeframe, layer)
        if row is None or not row.last_updated_ts:
            return None
        return datetime.fromisoformat(row.last_updated_ts)

    def upsert(
        self,
        *,
        symbol: str,
        timeframe: str,
        layer: str = "raw",
        first_available_ts: datetime | str | None = None,
        last_updated_ts: datetime | str | None = None,
        last_run_status: str = "success",
        row_count: int = 0,
        checksum: str | None = None,
        version_hash: str | None = None,
        extra_json: str | None = None,
    ) -> None:
        with _LOCK:
            doc = self._load_tf(timeframe)
            symbols: dict[str, Any] = doc.setdefault("symbols", {})
            sym_entry: dict[str, Any] = symbols.setdefault(symbol, {})
            existing = sym_entry.get(layer) if isinstance(sym_entry.get(layer), dict) else {}

            first_ts = _iso(first_available_ts)
            existing_first = existing.get("first_available_ts")
            if existing_first and first_ts is None:
                first_ts = existing_first
            elif existing_first and first_ts and existing_first < first_ts:
                first_ts = existing_first

            last_ts = _iso(last_updated_ts)
            if last_ts is None:
                last_ts = existing.get("last_updated_ts")

            sym_entry[layer] = {
                "first_available_ts": first_ts,
                "last_updated_ts": last_ts,
                "last_run_status": last_run_status,
                "last_run_at": _utc_now_iso(),
                "row_count": int(row_count),
                "checksum": checksum if checksum is not None else existing.get("checksum"),
                "version_hash": version_hash
                if version_hash is not None
                else existing.get("version_hash"),
                "extra_json": extra_json if extra_json is not None else existing.get("extra_json"),
            }
            symbols[symbol] = sym_entry
            doc["symbols"] = symbols
            self._save_tf(timeframe, doc)

    def audit(
        self,
        engine: str,
        event: str,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        detail_json: str | None = None,
    ) -> None:
        entry = {
            "ts": _utc_now_iso(),
            "engine": engine,
            "event": event,
            "symbol": symbol,
            "timeframe": timeframe,
            "detail_json": detail_json,
        }
        with _LOCK:
            if timeframe:
                doc = self._load_tf(timeframe)
                audit_list = doc.setdefault("audit", [])
                if not isinstance(audit_list, list):
                    audit_list = []
                audit_list.append(entry)
                # Cap per-file audit growth
                doc["audit"] = audit_list[-2000:]
                self._save_tf(timeframe, doc)
                return

            path = self._audit_path()
            raw = self._read_json(path)
            events: list[Any] = raw if isinstance(raw, list) else []
            events.append(entry)
            self._write_json(path, events[-5000:])

    def list_layer(self, layer: str = "raw") -> list[RegistryRow]:
        rows: list[RegistryRow] = []
        with _LOCK:
            for path in sorted(self.root_dir.glob("*.json")):
                if path.name.lower() == "audit.json":
                    continue
                raw = self._read_json(path)
                if not isinstance(raw, dict):
                    continue
                timeframe = str(raw.get("timeframe") or path.stem)
                symbols = raw.get("symbols") or {}
                if not isinstance(symbols, dict):
                    continue
                for symbol, layers in symbols.items():
                    if not isinstance(layers, dict):
                        continue
                    layer_data = layers.get(layer)
                    if isinstance(layer_data, dict):
                        rows.append(self._row_from_layer(symbol, timeframe, layer, layer_data))
        rows.sort(key=lambda r: (r.symbol, r.timeframe))
        return rows

    def list_timeframes(self) -> list[str]:
        """Return timeframe stems that have a state JSON file."""
        out: list[str] = []
        for path in sorted(self.root_dir.glob("*.json")):
            if path.name.lower() == "audit.json":
                continue
            raw = self._read_json(path)
            if isinstance(raw, dict) and raw.get("timeframe"):
                out.append(str(raw["timeframe"]))
            else:
                out.append(path.stem)
        return out

    def timeframe_path(self, timeframe: str) -> Path:
        return self._tf_path(timeframe)

    def load_timeframe_doc(self, timeframe: str) -> dict[str, Any] | None:
        """Load the full JSON document for one timeframe, or None if missing."""
        path = self._tf_path(timeframe)
        if not path.exists():
            return None
        with _LOCK:
            doc = self._load_tf(timeframe)
        return doc

    def prune_symbols(self, allowed_symbols: Iterable[str]) -> int:
        """Remove non-allowed symbols from per-timeframe JSON registry files."""
        allowed = {str(s) for s in allowed_symbols if str(s).strip()}
        if not allowed:
            return 0

        changed = 0
        with _LOCK:
            for path in sorted(self.root_dir.glob("*.json")):
                if path.name.lower() == "audit.json":
                    continue
                raw = self._read_json(path)
                if not isinstance(raw, dict):
                    continue
                symbols = raw.get("symbols")
                if not isinstance(symbols, dict):
                    continue
                filtered = {
                    symbol: data
                    for symbol, data in symbols.items()
                    if symbol in allowed
                }
                if filtered == symbols:
                    continue
                raw["symbols"] = filtered
                timeframe = str(raw.get("timeframe") or path.stem)
                self._save_tf(timeframe, raw)
                changed += 1
        return changed

    def list_state_files(self) -> list[dict[str, Any]]:
        """List per-timeframe state JSON files with path metadata for the UI."""
        files: list[dict[str, Any]] = []
        with _LOCK:
            for path in sorted(self.root_dir.glob("*.json")):
                if path.name.lower() == "audit.json":
                    continue
                raw = self._read_json(path)
                timeframe = path.stem
                symbols: list[str] = []
                updated_at = None
                if isinstance(raw, dict):
                    timeframe = str(raw.get("timeframe") or path.stem)
                    updated_at = raw.get("updated_at")
                    syms = raw.get("symbols") or {}
                    if isinstance(syms, dict):
                        symbols = sorted(syms.keys())
                stat = path.stat()
                files.append(
                    {
                        "timeframe": timeframe,
                        "filename": path.name,
                        "path": str(path),
                        "relative_path": f"data/registry/{path.name}",
                        "exists": True,
                        "size_bytes": int(stat.st_size),
                        "updated_at": updated_at,
                        "symbols": symbols,
                        "symbol_count": len(symbols),
                    }
                )
        return files
