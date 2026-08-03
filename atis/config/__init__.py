"""Configuration loading for ATIS."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

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


def secrets_env_path() -> Path:
    return CONFIG_DIR / "secrets.env"


def read_secrets_env() -> dict[str, str]:
    """Parse config/secrets.env into a flat key→value map (no expansion)."""
    path = secrets_env_path()
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def write_secrets_env(updates: dict[str, str | None], *, unset: Iterable[str] | None = None) -> Path:
    """
    Merge updates into secrets.env, preserving unrelated keys and comment lines.
    Pass None (or omit) to leave a key unchanged; use unset to remove keys.
    """
    path = secrets_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    remove = {k.strip() for k in (unset or []) if k and str(k).strip()}
    clean_updates = {
        str(k).strip(): ("" if v is None else str(v))
        for k, v in updates.items()
        if k and str(k).strip() and k.strip() not in remove
    }

    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()

    seen: set[str] = set()
    new_lines: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(raw)
            continue
        key, _, _ = stripped.partition("=")
        key = key.strip()
        if key in remove:
            continue
        if key in clean_updates:
            new_lines.append(f"{key}={clean_updates[key]}")
            seen.add(key)
        else:
            new_lines.append(raw)
            seen.add(key)

    for key, value in clean_updates.items():
        if key not in seen:
            new_lines.append(f"{key}={value}")

    text = "\n".join(new_lines).rstrip() + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def apply_secrets_to_environ(values: dict[str, str] | None = None) -> None:
    """Push secrets into os.environ so subsequent MT5Settings() picks them up."""
    data = values if values is not None else read_secrets_env()
    for key, value in data.items():
        os.environ[key] = value
    path = secrets_env_path()
    if path.exists():
        load_dotenv(path, override=True)


def load_mt5_settings(*, force_reload: bool = False) -> MT5Settings:
    """Load MT5 secrets; ensures dotenv is applied."""
    secrets_path = secrets_env_path()
    if secrets_path.exists():
        load_dotenv(secrets_path, override=force_reload)
    return MT5Settings()  # type: ignore[call-arg]


def save_mt5_credentials(
    *,
    login: int | str,
    password: str | None = None,
    server: str,
    path: str | None = None,
    keep_existing_password: bool = False,
) -> MT5Settings:
    """Persist MT5 credentials to secrets.env and refresh process env."""
    existing = read_secrets_env()
    login_s = str(int(str(login).strip()))
    server_s = str(server).strip()
    if not server_s:
        raise ValueError("MT5_SERVER is required")

    if keep_existing_password or password is None or password == "":
        pwd = existing.get("MT5_PASSWORD", "")
        if not pwd:
            raise ValueError("MT5_PASSWORD is required")
    else:
        pwd = str(password)

    updates: dict[str, str | None] = {
        "MT5_LOGIN": login_s,
        "MT5_PASSWORD": pwd,
        "MT5_SERVER": server_s,
    }
    unset: list[str] = []
    if path is not None:
        path_s = str(path).strip()
        if path_s:
            updates["MT5_PATH"] = path_s
        else:
            unset.append("MT5_PATH")

    write_secrets_env(updates, unset=unset or None)
    apply_secrets_to_environ()
    return load_mt5_settings(force_reload=True)


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
