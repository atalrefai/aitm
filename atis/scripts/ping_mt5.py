"""Smoke-test MT5 connectivity using config/secrets.env."""

from __future__ import annotations

import json
import sys

from atis.config import ensure_project_dirs, set_global_seed
from atis.shared.logging_utils import get_logger
from atis.shared.mt5_client import ping_mt5


def main() -> int:
    ensure_project_dirs()
    set_global_seed()
    log = get_logger("atis.scripts.ping_mt5")
    try:
        result = ping_mt5()
        log.info("ping_success", **{k: v for k, v in result.items() if k != "ok"})
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        log.exception("ping_failed", error=str(exc))
        print(f"MT5 ping failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
