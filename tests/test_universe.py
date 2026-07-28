"""Unit tests for scripts/universe.filter_universe."""
import pandas as pd
import pytest

from scripts import universe


def _make_prior(symbols_weights: list[tuple[str, float]], date: str = "20260721") -> pd.DataFrame:
    return pd.DataFrame([
        {"symbol": s, "date": date, "weight": w} for s, w in symbols_weights
    ])


def _make_prices(symbol_days: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for sym, days in symbol_days.items():
        for i, d in enumerate(days):
            rows.append({"symbol": sym, "date": d, "close": 10.0 + i * 0.1})
    return pd.DataFrame(rows)


def _dates(n: int, end: str = "20260720") -> list[str]:
    # produce n distinct YYYYMMDD strings ending at `end`; monotonic decreasing
    from datetime import datetime, timedelta
    dt = datetime.strptime(end, "%Y%m%d")
    return [(dt - timedelta(days=i)).strftime("%Y%m%d") for i in range(n)][::-1]


def test_filter_universe_keeps_symbols_with_enough_valid_closes():
    prior = _make_prior([("A", 0.6), ("B", 0.4)])
    prices = _make_prices({
        "A": _dates(250),
        "B": _dates(250),
    })
    syms, w = universe.filter_universe(prior, prices, "20260721", min_valid_days=200)
    assert syms == ["A", "B"]
    assert abs(w.sum() - 1.0) < 1e-9
    assert w["A"] == pytest.approx(0.6)
    assert w["B"] == pytest.approx(0.4)


def test_filter_universe_drops_symbols_with_too_few_valid_closes():
    prior = _make_prior([("A", 0.6), ("B", 0.4)])
    prices = _make_prices({
        "A": _dates(250),
        "B": _dates(50),  # too few
    })
    syms, w = universe.filter_universe(prior, prices, "20260721", min_valid_days=200)
    assert syms == ["A"]
    assert w["A"] == pytest.approx(1.0)


def test_filter_universe_drops_zero_and_nan_weight():
    prior = pd.DataFrame([
        {"symbol": "A", "date": "20260721", "weight": 0.5},
        {"symbol": "B", "date": "20260721", "weight": 0.0},
        {"symbol": "C", "date": "20260721", "weight": float("nan")},
    ])
    prices = _make_prices({s: _dates(250) for s in ("A", "B", "C")})
    syms, w = universe.filter_universe(prior, prices, "20260721")
    assert syms == ["A"]
    assert w["A"] == pytest.approx(1.0)


def test_filter_universe_only_considers_scan_date_prior():
    prior = pd.DataFrame([
        {"symbol": "A", "date": "20260721", "weight": 0.6},
        {"symbol": "B", "date": "20260721", "weight": 0.4},
        {"symbol": "C", "date": "20260720", "weight": 1.0},  # different date, ignored
    ])
    prices = _make_prices({s: _dates(250) for s in ("A", "B", "C")})
    syms, _ = universe.filter_universe(prior, prices, "20260721")
    assert syms == ["A", "B"]


def test_filter_universe_ignores_prices_on_or_after_scan_date():
    """min_valid_days is measured over the window ENDING T-1 — T-day close should not count."""
    prior = _make_prior([("A", 1.0)])
    # only day is scan_date itself → 0 valid days pre-T
    prices = _make_prices({"A": ["20260721"]})
    syms, _ = universe.filter_universe(prior, prices, "20260721", min_valid_days=1)
    assert syms == []


def test_filter_universe_empty_returns_empty_series():
    prior = _make_prior([])
    prices = _make_prices({})
    syms, w = universe.filter_universe(prior, prices, "20260721")
    assert syms == []
    assert len(w) == 0
