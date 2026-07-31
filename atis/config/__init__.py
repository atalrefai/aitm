"""Configuration loading for ATIS."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# project_root/atis/config/loader.py -> parents[2] = project_root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class MT5Settings(BaseSettings):
    """MT5 credentials from environment / secrets.env only — never hardcode."""

    model_config = SettingsConfigDict(
        env_file=str(CONFIG_DIR / "secrets.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mt5_login: int = Field(..., alias="MT5_LOGIN")
    mt5_password: str = Field(..., alias="MT5_PASSWORD")
    mt5_server: str = Field(..., alias="MT5_SERVER")
    mt5_path: str | None = Field(default=None, alias="MT5_PATH")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data)}")
    return data


@lru_cache(maxsize=1)
def load_engine_config() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "engine_config.yaml")


def clear_config_caches() -> None:
    """Drop cached YAML configs so training/UI pick up file edits without process restart."""
    load_engine_config.cache_clear()
    load_symbols.cache_clear()
    load_timeframes.cache_clear()


@lru_cache(maxsize=1)
def load_symbols() -> list[str]:
    data = _load_yaml(CONFIG_DIR / "symbols.yaml")
    symbols = data.get("symbols", [])
    if not symbols:
        raise ValueError("symbols.yaml has an empty symbols list")
    return list(symbols)


@lru_cache(maxsize=1)
def load_timeframes() -> dict[str, Any]:
    data = _load_yaml(CONFIG_DIR / "timeframes.yaml")
    tfs = data.get("timeframes", {})
    if not tfs:
        raise ValueError("timeframes.yaml has no timeframes")
    return dict(tfs)


def load_mt5_settings() -> MT5Settings:
    """Load MT5 secrets; ensures dotenv is applied once."""
    secrets_path = CONFIG_DIR / "secrets.env"
    if secrets_path.exists():
        load_dotenv(secrets_path, override=False)
    return MT5Settings()  # type: ignore[call-arg]


def resolve_path(relative: str | Path) -> Path:
    """Resolve a project-relative path to an absolute Path."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def get_path(key: str) -> Path:
    """Return an absolute path from engine_config.paths[key]."""
    cfg = load_engine_config()
    paths = cfg.get("paths", {})
    if key not in paths:
        raise KeyError(f"Unknown path key: {key}")
    return resolve_path(paths[key])


def ensure_project_dirs() -> None:
    """Create required data/log directories if missing."""
    cfg = load_engine_config()
    for key, rel in cfg.get("paths", {}).items():
        path = resolve_path(rel)
        # data_registry is a directory of per-timeframe JSON state files
        if key == "data_registry" and path.suffix.lower() == ".db":
            path = path.parent
        path.mkdir(parents=True, exist_ok=True)
    # Always ensure standard leaves exist
    for sub in (
        "data/raw",
        "data/clean",
        "data/features",
        "data/registry",
        "data/patterns",
        "models",
        "logs",
    ):
        (PROJECT_ROOT / sub).mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int | None = None) -> int:
    """Fix RNG seed for determinism (Principle 1.3)."""
    import random

    import numpy as np

    cfg = load_engine_config()
    if seed is None:
        seed = int(cfg.get("project", {}).get("random_seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed
