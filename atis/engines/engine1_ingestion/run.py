"""CLI entrypoint for Engine 1 — Data Ingestion."""

from __future__ import annotations

import argparse
import json
import sys

from atis.engines.engine1_ingestion import run_ingestion
from atis.shared.logging_utils import get_logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ATIS Engine 1 — MT5 Data Ingestion")
    p.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to ingest (default: engine_config.default_symbols)",
    )
    p.add_argument(
        "--timeframes",
        nargs="+",
        default=None,
        help="Timeframes to ingest (default: engine_config.default_timeframes)",
    )
    p.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete existing parquet and re-backfill from scratch",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = get_logger("atis.engine1.cli")
    try:
        report = run_ingestion(
            symbols=args.symbols,
            timeframes=args.timeframes,
            force_rebuild=args.force_rebuild,
        )
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.status == "success" else 2
    except Exception as exc:
        log.exception("cli_failed", error=str(exc))
        print(f"Engine 1 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
