"""Pattern knowledge base — persistent catalog, stats, and discovery events."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterable

from atis.config import PROJECT_ROOT, get_path
from atis.shared.feature_engine.patterns import (
    COMPOUND_TEMPLATES,
    PATTERN_CATALOG,
    pattern_labels,
)


_KB_WRITE_LOCK = threading.RLock()
_CATALOG_SYNCED_PATHS: set[str] = set()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pattern_catalog (
    pattern_key   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    category      TEXT NOT NULL,
    bias          TEXT NOT NULL,
    conditions    TEXT,
    source        TEXT DEFAULT 'builtin',
    created_at    TEXT,
    updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS pattern_stats (
    symbol            TEXT NOT NULL,
    timeframe         TEXT NOT NULL,
    pattern_key       TEXT NOT NULL,
    occurrences       INTEGER DEFAULT 0,
    evaluated         INTEGER DEFAULT 0,
    successes         INTEGER DEFAULT 0,
    success_rate      REAL,
    avg_forward_return REAL,
    confidence        REAL,
    last_seen_ts      TEXT,
    conditions        TEXT,
    updated_at        TEXT,
    PRIMARY KEY (symbol, timeframe, pattern_key)
);

CREATE TABLE IF NOT EXISTS pattern_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT NOT NULL,
    timeframe     TEXT NOT NULL,
    pattern_key   TEXT NOT NULL,
    ts            TEXT NOT NULL,
    close         REAL,
    strength      REAL,
    forward_return REAL,
    success       INTEGER,
    meta_json     TEXT
);

CREATE INDEX IF NOT EXISTS idx_pattern_events_lookup
    ON pattern_events(symbol, timeframe, pattern_key, ts);

CREATE TABLE IF NOT EXISTS discovered_compounds (
    compound_key  TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    legs_json     TEXT NOT NULL,
    lift          REAL,
    occurrences   INTEGER,
    success_rate  REAL,
    confidence    REAL,
    conditions    TEXT,
    symbol        TEXT,
    timeframe     TEXT,
    meta_json     TEXT,
    updated_at    TEXT
);
"""

MIGRATE_SQL = """
ALTER TABLE discovered_compounds ADD COLUMN meta_json TEXT;
ALTER TABLE pattern_stats ADD COLUMN quality_score REAL;
ALTER TABLE pattern_stats ADD COLUMN approved INTEGER;
ALTER TABLE pattern_stats ADD COLUMN std_dev REAL;
ALTER TABLE pattern_stats ADD COLUMN strength REAL;
"""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_kb_path() -> Path:
    try:
        root = get_path("data_registry")
        # Legacy config pointed at a .db file; keep pattern KB beside it.
        if root.suffix.lower() == ".db":
            root = root.parent
    except Exception:
        root = PROJECT_ROOT / "data" / "registry"
    root.mkdir(parents=True, exist_ok=True)
    return root / "pattern_knowledge.db"


class PatternKnowledgeBase:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else default_kb_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self) -> None:
        # Serialize schema + catalog bootstrap — parallel discovery threads
        # otherwise collide on the same SQLite file ("database is locked").
        with _KB_WRITE_LOCK:
            with self._conn_unlocked() as con:
                con.executescript(SCHEMA_SQL)
                for stmt in MIGRATE_SQL.strip().split(";"):
                    s = stmt.strip()
                    if not s:
                        continue
                    try:
                        con.execute(s)
                    except sqlite3.OperationalError:
                        # Column already exists on upgraded DBs
                        pass
            key = str(self.db_path.resolve())
            if key not in _CATALOG_SYNCED_PATHS:
                self._sync_builtin_catalog_unlocked()
                _CATALOG_SYNCED_PATHS.add(key)

    @contextmanager
    def _conn_unlocked(self) -> Generator[sqlite3.Connection, None, None]:
        con = sqlite3.connect(str(self.db_path), timeout=120.0, check_same_thread=False)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            con.execute("PRAGMA busy_timeout=120000")
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        with self._conn_unlocked() as con:
            yield con

    def sync_builtin_catalog(self) -> int:
        with _KB_WRITE_LOCK:
            n = self._sync_builtin_catalog_unlocked()
            _CATALOG_SYNCED_PATHS.add(str(self.db_path.resolve()))
            return n

    def _sync_builtin_catalog_unlocked(self) -> int:
        now = _utc()
        n = 0
        with self._conn_unlocked() as con:
            for key, meta in PATTERN_CATALOG.items():
                con.execute(
                    """
                    INSERT INTO pattern_catalog(pattern_key, name, category, bias, conditions, source, created_at, updated_at)
                    VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(pattern_key) DO UPDATE SET
                        name=excluded.name,
                        category=excluded.category,
                        bias=excluded.bias,
                        conditions=excluded.conditions,
                        updated_at=excluded.updated_at
                    """,
                    (
                        key,
                        meta["name"],
                        meta["category"],
                        meta["bias"],
                        meta.get("conditions"),
                        "builtin",
                        now,
                        now,
                    ),
                )
                n += 1
            for key, name, a, b in COMPOUND_TEMPLATES:
                con.execute(
                    """
                    INSERT INTO pattern_catalog(pattern_key, name, category, bias, conditions, source, created_at, updated_at)
                    VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(pattern_key) DO UPDATE SET
                        name=excluded.name,
                        conditions=excluded.conditions,
                        updated_at=excluded.updated_at
                    """,
                    (
                        key,
                        name,
                        "compound",
                        "neutral",
                        f"{a}=1 AND {b}=1",
                        "builtin_compound",
                        now,
                        now,
                    ),
                )
                n += 1
        return n

    def upsert_discovered(self, items: Iterable[dict[str, Any]]) -> int:
        with _KB_WRITE_LOCK:
            return self._upsert_discovered_unlocked(items)

    def _upsert_discovered_unlocked(self, items: Iterable[dict[str, Any]]) -> int:
        now = _utc()
        count = 0
        with self._conn() as con:
            for item in items:
                key = str(item.get("key") or item.get("id") or "")
                if not key:
                    continue
                # Only promote approved New* patterns into training-facing catalog source
                source = "discovered"
                if key.startswith("New"):
                    source = "discovered_approved" if item.get("approved") else "discovered_rejected"
                con.execute(
                    """
                    INSERT INTO pattern_catalog(pattern_key, name, category, bias, conditions, source, created_at, updated_at)
                    VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(pattern_key) DO UPDATE SET
                        name=excluded.name,
                        conditions=excluded.conditions,
                        source=excluded.source,
                        bias=excluded.bias,
                        updated_at=excluded.updated_at
                    """,
                    (
                        key,
                        item.get("name", key),
                        item.get("category", "discovered" if key.startswith("New") else "compound"),
                        item.get("bias", "neutral"),
                        item.get("conditions") or item.get("mathematical_rules"),
                        source,
                        now,
                        now,
                    ),
                )
                meta = {
                    "description": item.get("description"),
                    "mathematical_rules": item.get("mathematical_rules"),
                    "logical_rules": item.get("logical_rules"),
                    "appearance_conditions": item.get("appearance_conditions"),
                    "std_dev": item.get("std_dev"),
                    "risk_ratio": item.get("risk_ratio"),
                    "quality_score": item.get("quality_score"),
                    "strength": item.get("strength"),
                    "approved": item.get("approved"),
                    "soft_promoted": item.get("soft_promoted"),
                    "bias": item.get("bias"),
                    "best_timeframe": item.get("best_timeframe"),
                    "best_market_regime": item.get("best_market_regime"),
                    "validation": item.get("validation"),
                    "avg_move_after": item.get("avg_move_after") or item.get("avg_forward_return"),
                    "htf_confirm": item.get("htf_confirm"),
                }
                con.execute(
                    """
                    INSERT INTO discovered_compounds(
                        compound_key, name, legs_json, lift, occurrences, success_rate,
                        confidence, conditions, symbol, timeframe, meta_json, updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(compound_key) DO UPDATE SET
                        lift=excluded.lift,
                        occurrences=excluded.occurrences,
                        success_rate=COALESCE(excluded.success_rate, discovered_compounds.success_rate),
                        confidence=COALESCE(excluded.confidence, discovered_compounds.confidence),
                        conditions=excluded.conditions,
                        meta_json=excluded.meta_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        key,
                        item.get("name", key),
                        json.dumps(item.get("legs") or []),
                        item.get("lift"),
                        int(item.get("occurrences") or 0),
                        item.get("success_rate"),
                        item.get("confidence"),
                        item.get("conditions") or item.get("mathematical_rules"),
                        item.get("symbol"),
                        item.get("timeframe"),
                        json.dumps(meta, ensure_ascii=False, default=str),
                        now,
                    ),
                )
                count += 1
        return count

    def upsert_stats(self, rows: Iterable[dict[str, Any]]) -> int:
        with _KB_WRITE_LOCK:
            return self._upsert_stats_unlocked(rows)

    def _upsert_stats_unlocked(self, rows: Iterable[dict[str, Any]]) -> int:
        now = _utc()
        n = 0
        with self._conn() as con:
            for r in rows:
                approved = r.get("approved")
                approved_i = None if approved is None else (1 if approved else 0)
                con.execute(
                    """
                    INSERT INTO pattern_stats(
                        symbol, timeframe, pattern_key, occurrences, evaluated, successes,
                        success_rate, avg_forward_return, confidence, last_seen_ts, conditions,
                        quality_score, approved, std_dev, strength, updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(symbol, timeframe, pattern_key) DO UPDATE SET
                        occurrences=excluded.occurrences,
                        evaluated=excluded.evaluated,
                        successes=excluded.successes,
                        success_rate=excluded.success_rate,
                        avg_forward_return=excluded.avg_forward_return,
                        confidence=excluded.confidence,
                        last_seen_ts=excluded.last_seen_ts,
                        conditions=excluded.conditions,
                        quality_score=excluded.quality_score,
                        approved=excluded.approved,
                        std_dev=excluded.std_dev,
                        strength=excluded.strength,
                        updated_at=excluded.updated_at
                    """,
                    (
                        r["symbol"],
                        r["timeframe"],
                        r["pattern_key"],
                        int(r.get("occurrences") or 0),
                        int(r.get("evaluated") or 0),
                        int(r.get("successes") or 0),
                        r.get("success_rate"),
                        r.get("avg_forward_return"),
                        r.get("confidence"),
                        r.get("last_seen_ts"),
                        r.get("conditions"),
                        r.get("quality_score"),
                        approved_i,
                        r.get("std_dev"),
                        r.get("strength"),
                        now,
                    ),
                )
                n += 1
        return n

    def insert_events(self, events: Iterable[dict[str, Any]], *, chunk: int = 2000) -> int:
        with _KB_WRITE_LOCK:
            return self._insert_events_unlocked(events, chunk=chunk)

    def _insert_events_unlocked(self, events: Iterable[dict[str, Any]], *, chunk: int = 2000) -> int:
        rows = list(events)
        if not rows:
            return 0
        with self._conn() as con:
            # Replace prior events for same symbol/tf/pattern keys in this batch scope
            pairs = {(e["symbol"], e["timeframe"]) for e in rows}
            for sym, tf in pairs:
                con.execute(
                    "DELETE FROM pattern_events WHERE symbol=? AND timeframe=?",
                    (sym, tf),
                )
            payload = [
                (
                    e["symbol"],
                    e["timeframe"],
                    e["pattern_key"],
                    e["ts"],
                    e.get("close"),
                    e.get("strength"),
                    e.get("forward_return"),
                    e.get("success"),
                    json.dumps(e.get("meta") or {}),
                )
                for e in rows
            ]
            for i in range(0, len(payload), chunk):
                con.executemany(
                    """
                    INSERT INTO pattern_events(
                        symbol, timeframe, pattern_key, ts, close, strength,
                        forward_return, success, meta_json
                    ) VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    payload[i : i + chunk],
                )
        return len(payload)

    def list_stats(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
        *,
        min_occurrences: int = 0,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        q = """
            SELECT s.*, c.name, c.category, c.bias, c.conditions AS catalog_conditions, c.source
            FROM pattern_stats s
            LEFT JOIN pattern_catalog c ON c.pattern_key = s.pattern_key
            WHERE s.occurrences >= ?
        """
        args: list[Any] = [min_occurrences]
        if symbol:
            q += " AND s.symbol=?"
            args.append(symbol)
        if timeframe:
            q += " AND s.timeframe=?"
            args.append(timeframe)
        q += " ORDER BY s.occurrences DESC, s.success_rate DESC LIMIT ?"
        args.append(limit)
        with self._conn() as con:
            rows = con.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    def list_discovered(
        self,
        limit: int = 100,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        key_prefix: str | None = None,
        include_null_lift: bool = True,
    ) -> list[dict[str, Any]]:
        """List discovered compounds.

        NewN rows often have ``lift IS NULL``; ordering by lift alone previously
        pushed them past the LIMIT and dropped their metadata on JSON export.
        """
        q = "SELECT * FROM discovered_compounds WHERE 1=1"
        args: list[Any] = []
        if symbol:
            q += " AND (symbol IS NULL OR symbol=?)"
            args.append(symbol)
        if timeframe:
            q += " AND (timeframe IS NULL OR timeframe=?)"
            args.append(timeframe)
        if key_prefix:
            q += " AND compound_key LIKE ?"
            args.append(f"{key_prefix}%")
        if include_null_lift:
            # Prefer real lift, but keep null-lift NewN ahead of the cut-off
            q += " ORDER BY (lift IS NULL) ASC, lift DESC, occurrences DESC LIMIT ?"
        else:
            q += " AND lift IS NOT NULL ORDER BY lift DESC, occurrences DESC LIMIT ?"
        args.append(limit)
        with self._conn() as con:
            rows = con.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    def list_new_patterns(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Return NewN discoveries (lift may be null) without being crowded out."""
        return self.list_discovered(
            limit,
            symbol=symbol,
            timeframe=timeframe,
            key_prefix="New",
            include_null_lift=True,
        )

    def catalog_size(self) -> dict[str, int]:
        with self._conn() as con:
            total = con.execute("SELECT COUNT(*) FROM pattern_catalog").fetchone()[0]
            by_cat = con.execute(
                "SELECT category, COUNT(*) AS n FROM pattern_catalog GROUP BY category"
            ).fetchall()
            stats_n = con.execute("SELECT COUNT(*) FROM pattern_stats WHERE occurrences>0").fetchone()[0]
            disc = con.execute("SELECT COUNT(*) FROM discovered_compounds").fetchone()[0]
        return {
            "catalog_total": int(total),
            "with_stats": int(stats_n),
            "discovered_compounds": int(disc),
            "by_category": {str(r[0]): int(r[1]) for r in by_cat},
            "builtin_labels": len(pattern_labels()),
        }

    def summary(self, symbol: str, timeframe: str | None = None) -> dict[str, Any]:
        stats = self.list_stats(symbol, timeframe, min_occurrences=1, limit=1000)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "catalog": self.catalog_size(),
            "patterns_with_hits": len(stats),
            "total_occurrences": int(sum(s.get("occurrences") or 0 for s in stats)),
            "avg_success_rate": (
                float(
                    sum(s["success_rate"] for s in stats if s.get("success_rate") is not None)
                    / max(1, sum(1 for s in stats if s.get("success_rate") is not None))
                )
                if any(s.get("success_rate") is not None for s in stats)
                else None
            ),
            "top": stats[:30],
            "discovered": self.list_discovered(40),
        }
