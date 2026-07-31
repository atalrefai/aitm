"""CLI entrypoint for Engine 2 — Data Cleaning."""

from __future__ import annotations

import argparse
import json
import sys

from atis.engines.engine2_cleaning import run_cleaning
from atis.shared.logging_utils import get_logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ATIS Engine 2 — Data Cleaning")
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--timeframes", nargs="+", default=None)
    p.add_argument("--force-rebuild", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = get_logger("atis.engine2.cli")
    try:
        report = run_cleaning(
            symbols=args.symbols,
            timeframes=args.timeframes,
            force_rebuild=args.force_rebuild,
        )
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.status == "success" else 2
    except Exception as exc:
        log.exception("cli_failed", error=str(exc))
        print(f"Engine 2 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
