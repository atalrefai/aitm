"""Run Engines 1 → 2 → 3 sequentially for configured (or CLI) symbols/TFs."""

from __future__ import annotations

import argparse
import json
import sys

from atis.config import load_engine_config, load_symbols, load_timeframes
from atis.engines.engine1_ingestion import run_ingestion
from atis.engines.engine2_cleaning import run_cleaning
from atis.engines.engine3_features import run_features
from atis.shared.logging_utils import get_logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ATIS pipeline Engines 1-3")
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--timeframes", nargs="+", default=None)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip Engine 1 (use existing raw data)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = get_logger("atis.pipeline")
    cfg = load_engine_config().get("engine1_ingestion", {})
    symbols = args.symbols or list(cfg.get("default_symbols") or load_symbols())
    timeframes = args.timeframes or list(cfg.get("default_timeframes") or list(load_timeframes()))

    summary: dict = {"symbols": symbols, "timeframes": timeframes}

    try:
        if not args.skip_ingest:
            log.info("pipeline_engine1_start", n=len(symbols) * len(timeframes))
            r1 = run_ingestion(symbols, timeframes, force_rebuild=args.force_rebuild)
            summary["engine1"] = r1.to_dict()["summary"]
            summary["engine1_status"] = r1.status
        else:
            summary["engine1_status"] = "skipped"

        log.info("pipeline_engine2_start")
        r2 = run_cleaning(symbols, timeframes, force_rebuild=args.force_rebuild)
        summary["engine2"] = r2.to_dict()["summary"]
        summary["engine2_status"] = r2.status

        log.info("pipeline_engine3_start")
        r3 = run_features(symbols, timeframes, force_rebuild=args.force_rebuild)
        summary["engine3"] = r3.to_dict()["summary"]
        summary["engine3_status"] = r3.status

        print(json.dumps(summary, indent=2))
        bad = [
            summary.get("engine1_status"),
            summary.get("engine2_status"),
            summary.get("engine3_status"),
        ]
        if any(s not in ("success", "skipped") for s in bad):
            return 2
        return 0
    except Exception as exc:
        log.exception("pipeline_failed", error=str(exc))
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
