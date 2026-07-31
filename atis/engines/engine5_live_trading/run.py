"""CLI for Engine 5 — Live / Paper trading."""

from __future__ import annotations

import argparse
import json
import sys

from atis.engines.engine5_live_trading import run_live_once, run_live_loop
from atis.shared.logging_utils import get_logger


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ATIS Engine 5 — Paper/Demo Trading")
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--loop", action="store_true", help="Run continuous loop")
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument(
        "--execute-demo",
        action="store_true",
        help="Send real demo orders (still blocked for live/real money)",
    )
    p.add_argument(
        "--require-champion",
        action="store_true",
        help="Refuse ungated models",
    )
    args = p.parse_args(argv)
    log = get_logger("atis.engine5.cli")
    dry_run = not args.execute_demo
    allow_ungated = not args.require_champion
    from atis.config import load_engine_config

    cfg = load_engine_config().get("engine5_live", {})
    trade_cfg = load_engine_config().get("trading", {})
    symbols = args.symbols or list(cfg.get("symbols") or [trade_cfg.get("primary_symbol", "XAUUSD")])
    timeframe = args.timeframe or str(cfg.get("timeframe") or trade_cfg.get("primary_timeframe", "M5"))
    try:
        if args.loop:
            run_live_loop(
                symbols,
                timeframe,
                interval_seconds=args.interval,
                max_iterations=args.max_iterations,
                dry_run=dry_run,
                allow_ungated=allow_ungated,
            )
            return 0
        report = run_live_once(
            symbols,
            timeframe,
            dry_run=dry_run,
            allow_ungated=allow_ungated,
        )
        print(json.dumps(__import__("dataclasses").asdict(report), indent=2))
        return 0
    except Exception as exc:
        log.exception("cli_failed", error=str(exc))
        print(f"Engine 5 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
