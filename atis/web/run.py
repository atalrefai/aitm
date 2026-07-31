"""Launch ATIS Gold Desk web UI."""

from __future__ import annotations

import argparse

import uvicorn

from atis.config import load_engine_config


def main() -> None:
    cfg = load_engine_config().get("web", {})
    p = argparse.ArgumentParser(description="ATIS Gold Desk")
    p.add_argument("--host", default=str(cfg.get("host", "127.0.0.1")))
    p.add_argument("--port", type=int, default=int(cfg.get("port", 8787)))
    args = p.parse_args()
    uvicorn.run("atis.web.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
