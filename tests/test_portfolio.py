"""End-to-end test: portfolio.main() with a fully mocked panda_data."""
import sys
import types

import numpy as np
import pandas as pd
import pytest

from scripts import portfolio


def _make_fake_panda_data(monkeypatch, *, n_stocks: int = 30, n_days: int = 260,
                          scan_date: str = "20260721"):
    from datetime import datetime, timedelta

    rng = np.random.default_rng(0)
    symbols = [f"S{i:03d}.SH" for i in range(n_stocks)]

    # Date list ending on scan_date (inclusive), stepping back day-by-day.
    dt = datetime.strptime(scan_date, "%Y%m%d")
    all_dates = [(dt - timedelta(days=n_days - i)).strftime("%Y%m%d")
                 for i in range(n_days + 1)]

    # Random-walk closes + uniform turnover per (symbol, date)
    price_rows = []
    turnover_rows = []
    for s in symbols:
        p = 100.0
        for d in all_dates:
            p *= float(np.exp(rng.normal(0.0005, 0.02)))
            price_rows.append({"symbol": s, "date": d, "close": p})
            turnover_rows.append({"symbol": s, "date": d,
                                  "turnover": float(rng.uniform(0.005, 0.03))})
    prices_df = pd.DataFrame(price_rows)
    turnover_df = pd.DataFrame(turnover_rows)

    # Prior weights on scan_date, normalized
    weights = rng.uniform(1.0, 3.0, size=n_stocks)
    weights = weights / weights.sum()
    prior_df = pd.DataFrame({
        "index_symbol": ["000300.SH"] * n_stocks,
        "date": [scan_date] * n_stocks,
        "stock_symbol": symbols,
        "weight": weights,
    })

    fake = types.ModuleType("panda_data")
    fake.init_token = lambda **kw: None
    fake.get_index_weights = lambda **kw: prior_df.copy()
    fake.get_stock_daily = lambda **kw: prices_df[
        prices_df["symbol"].isin(kw.get("symbol") or symbols)
    ][["symbol", "date", "close"]].copy()
    fake.get_factor = lambda **kw: turnover_df[
        turnover_df["symbol"].isin(kw.get("symbol") or symbols)
    ][["symbol", "date", "turnover"]].copy()

    exceptions_mod = types.ModuleType("panda_data.exceptions")

    class ServiceError(Exception):
        pass

    exceptions_mod.ServiceError = ServiceError
    fake.exceptions = exceptions_mod
    monkeypatch.setitem(sys.modules, "panda_data", fake)
    monkeypatch.setitem(sys.modules, "panda_data.exceptions", exceptions_mod)


def test_portfolio_main_end_to_end(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PANDA_DATA_USERNAME", "u")
    monkeypatch.setenv("PANDA_DATA_PASSWORD", "p")
    _make_fake_panda_data(monkeypatch)
    out = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "portfolio.py",
        "--date", "20260721",
        "--min_valid_days", "100",
        "--cov_lookback", "150",
        "--output_dir", str(out),
    ])

    rc = portfolio.main()
    assert rc == 0, capsys.readouterr()

    csv_path = out / "portfolio_20260721.csv"
    md_path = out / "portfolio_20260721.md"
    assert csv_path.exists()
    assert md_path.exists()

    df = pd.read_csv(csv_path)
    assert len(df) >= 20  # some symbols may drop for NaN in cov window; most survive
    assert abs(df["w_bl"].sum() - 1.0) < 1e-6
    assert (df["w_bl"] >= 0).all()
    # Non-degenerate: some Δw should be nonzero
    assert (df["delta_w"].abs() > 0).any()
