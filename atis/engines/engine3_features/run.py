"""CLI for Engine 3 — Feature & Pattern Engine."""

from __future__ import annotations

import argparse
import json
import sys

from atis.engines.engine3_features import run_features
from atis.shared.logging_utils import get_logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ATIS Engine 3 — Features")
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--timeframes", nargs="+", default=None)
    p.add_argument("--force-rebuild", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = get_logger("atis.engine3.cli")
    try:
        report = run_features(
            symbols=args.symbols,
            timeframes=args.timeframes,
            force_rebuild=args.force_rebuild,
        )
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.status == "success" else 2
    except Exception as exc:
        log.exception("cli_failed", error=str(exc))
        print(f"Engine 3 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
