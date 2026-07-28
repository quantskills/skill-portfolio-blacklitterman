"""Unit tests for scripts/data — loaders, column self-check, error handling.

We install a stub `panda_data` module via monkeypatch so no real network is touched.
"""
import sys
import types

import pandas as pd
import pytest

from scripts import data


def _install_fake_panda_data(monkeypatch, *, weights_df=None, prices_df=None, turnover_df=None,
                              init_token_impl=None):
    fake = types.ModuleType("panda_data")
    fake.init_token = init_token_impl or (lambda **kw: None)
    fake.get_index_weights = lambda **kw: weights_df
    fake.get_stock_daily = lambda **kw: prices_df
    fake.get_factor = lambda **kw: turnover_df

    exceptions_mod = types.ModuleType("panda_data.exceptions")

    class ServiceError(Exception):
        pass

    exceptions_mod.ServiceError = ServiceError
    fake.exceptions = exceptions_mod

    monkeypatch.setitem(sys.modules, "panda_data", fake)
    monkeypatch.setitem(sys.modules, "panda_data.exceptions", exceptions_mod)
    return ServiceError


def test_init_panda_data_raises_when_env_missing(monkeypatch):
    monkeypatch.delenv("PANDA_DATA_USERNAME", raising=False)
    monkeypatch.delenv("PANDA_DATA_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="PANDA_DATA_USERNAME"):
        data.init_panda_data()


def test_load_prior_returns_expected_columns(monkeypatch):
    weights = pd.DataFrame({
        "index_symbol": ["000300.SH"] * 3,
        "date": ["20260721"] * 3,
        "stock_symbol": ["600000.SH", "600519.SH", "000001.SZ"],
        "weight": [0.5, 1.5, 2.0],
    })
    _install_fake_panda_data(monkeypatch, weights_df=weights)
    df = data.load_prior("20260721")
    assert set(df.columns) >= {"symbol", "date", "weight"}
    assert len(df) == 3
    # symbol should be pulled from stock_symbol
    assert set(df["symbol"]) == {"600000.SH", "600519.SH", "000001.SZ"}


def test_load_prior_falls_back_to_equal_weights_when_weight_missing(monkeypatch, capsys):
    weights = pd.DataFrame({
        "index_symbol": ["000300.SH"] * 4,
        "date": ["20260721"] * 4,
        "stock_symbol": ["A", "B", "C", "D"],
        # note: no `weight` column
    })
    _install_fake_panda_data(monkeypatch, weights_df=weights)
    df = data.load_prior("20260721")
    assert "weight" in df.columns
    # Equal weights summing to 1
    assert df["weight"].nunique() == 1
    assert abs(df["weight"].sum() - 1.0) < 1e-9
    captured = capsys.readouterr()
    assert "weight" in captured.err.lower() and "equal" in captured.err.lower()


def test_load_prior_raises_on_empty(monkeypatch):
    _install_fake_panda_data(monkeypatch, weights_df=pd.DataFrame())
    with pytest.raises(ValueError, match="no get_index_weights data"):
        data.load_prior("20260721")


def test_load_prices_returns_expected_columns(monkeypatch):
    prices = pd.DataFrame({
        "symbol": ["600519.SH", "600519.SH"],
        "date": ["20260720", "20260721"],
        "close": [1500.0, 1510.0],
    })
    _install_fake_panda_data(monkeypatch, prices_df=prices)
    df = data.load_prices("20260101", "20260721", ["600519.SH"])
    assert set(df.columns) >= {"symbol", "date", "close"}


def test_load_turnover_returns_expected_columns(monkeypatch):
    tv = pd.DataFrame({
        "symbol": ["600519.SH"],
        "date": ["20260721"],
        "turnover": [0.012],
    })
    _install_fake_panda_data(monkeypatch, turnover_df=tv)
    df = data.load_turnover("20260101", "20260721", ["600519.SH"])
    assert set(df.columns) >= {"symbol", "date", "turnover"}


def test_assert_columns_raises_on_missing_column():
    df = pd.DataFrame({"symbol": ["A"], "date": ["20260721"]})  # no `close`
    with pytest.raises(ValueError, match="missing columns"):
        data._assert_columns(df, "prices")


def test_main_returns_1_on_missing_credentials(monkeypatch, capsys):
    monkeypatch.delenv("PANDA_DATA_USERNAME", raising=False)
    monkeypatch.delenv("PANDA_DATA_PASSWORD", raising=False)
    monkeypatch.setattr(sys, "argv", ["data.py", "--self-check", "--date", "20260721"])
    rc = data._main()
    assert rc == 1
    assert "PANDA_DATA_USERNAME" in capsys.readouterr().err


def test_main_returns_1_on_service_error(monkeypatch, capsys):
    ServiceError = _install_fake_panda_data(
        monkeypatch,
        init_token_impl=lambda **kw: (_ for _ in ()).throw(
            sys.modules["panda_data.exceptions"].ServiceError("HTTP 503")
        ),
    )
    monkeypatch.setenv("PANDA_DATA_USERNAME", "u")
    monkeypatch.setenv("PANDA_DATA_PASSWORD", "p")
    monkeypatch.setattr(sys, "argv", ["data.py", "--self-check", "--date", "20260721"])
    rc = data._main()
    assert rc == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "503" in err or "panda_data" in err.lower()
