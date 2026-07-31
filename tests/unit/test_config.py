"""Config loading tests."""

from atis.config import load_engine_config, load_symbols, load_timeframes, set_global_seed


def test_load_configs() -> None:
    cfg = load_engine_config()
    assert cfg["project"]["name"] == "ATIS"
    symbols = load_symbols()
    assert symbols == ["XAUUSD"]
    tfs = load_timeframes()
    assert "H1" in tfs
    assert tfs["H1"]["minutes"] == 60


def test_seed() -> None:
    assert set_global_seed(42) == 42
