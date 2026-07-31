"""Engine 3 orchestration tests."""

from __future__ import annotations

import threading
import time

from atis.engines import engine3_features as e3


def test_run_features_parallelizes_symbol_timeframe_tasks(monkeypatch) -> None:
    thread_names: list[str] = []

    def fake_compute_symbol_timeframe(
        registry: object,
        symbol: str,
        timeframe: str,
        *,
        force_rebuild: bool = False,
    ) -> e3.FeatureResult:
        thread_names.append(threading.current_thread().name)
        time.sleep(0.05)
        return e3.FeatureResult(symbol=symbol, timeframe=timeframe, rows_processed=1, rows_total=1)

    monkeypatch.setattr(e3, "compute_symbol_timeframe", fake_compute_symbol_timeframe)

    report = e3.run_features(symbols=["XAUUSD"], timeframes=["M1", "M5", "H1"])

    assert [r.timeframe for r in report.results] == ["M1", "M5", "H1"]
    assert len(set(thread_names)) >= 2
    assert report.status == "success"
