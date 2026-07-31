"""CLI for Engine 4 — Training & Backtesting."""

from __future__ import annotations

import argparse
import json
import sys

from atis.engines.engine4_training import run_training
from atis.shared.logging_utils import get_logger


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ATIS Engine 4 — Training")
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--timeframes", nargs="+", default=None)
    args = p.parse_args(argv)
    log = get_logger("atis.engine4.cli")
    try:
        report = run_training(symbols=args.symbols, timeframes=args.timeframes)
        print(json.dumps(report, indent=2))
        return 0 if report["summary"]["errors"] == 0 else 2
    except Exception as exc:
        log.exception("cli_failed", error=str(exc))
        print(f"Engine 4 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
