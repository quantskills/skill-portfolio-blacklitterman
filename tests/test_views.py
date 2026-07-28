"""Unit tests for scripts/views.build_views."""
import numpy as np
import pandas as pd
import pytest

from scripts import views


def _prices_with_momentum(symbols: list[str], mom_ranks: dict[str, float], n_days: int = 30) -> pd.DataFrame:
    """Construct closes such that the 20-day return of each symbol is proportional to mom_ranks[symbol]."""
    from datetime import datetime, timedelta
    dt = datetime.strptime("20260721", "%Y%m%d")
    dates = [(dt - timedelta(days=n_days - i)).strftime("%Y%m%d") for i in range(n_days)]
    rows = []
    for s in symbols:
        r = mom_ranks[s]
        # linear ramp so 20-day cumulative return ~ r
        closes = np.linspace(100.0, 100.0 * (1.0 + r), n_days)
        for d, c in zip(dates, closes):
            rows.append({"symbol": s, "date": d, "close": float(c)})
    return pd.DataFrame(rows)


def _flat_turnover(symbols: list[str], turnover_by_sym: dict[str, float], n_days: int = 30) -> pd.DataFrame:
    from datetime import datetime, timedelta
    dt = datetime.strptime("20260721", "%Y%m%d")
    dates = [(dt - timedelta(days=n_days - i)).strftime("%Y%m%d") for i in range(n_days)]
    rows = []
    for s in symbols:
        for d in dates:
            rows.append({"symbol": s, "date": d, "turnover": turnover_by_sym[s]})
    return pd.DataFrame(rows)


def test_build_views_p_shape_and_row_sum_zero():
    N = 20
    syms = [f"S{i:02d}" for i in range(N)]
    mom = {s: i * 0.01 for i, s in enumerate(syms)}
    prices = _prices_with_momentum(syms, mom)
    turnover = _flat_turnover(syms, {s: 0.01 + i * 0.001 for i, s in enumerate(syms)})
    P, Q, mem = views.build_views(prices, turnover, syms, "20260721", q_hat=0.05)
    assert P.shape == (3, N)
    assert Q.shape == (3,)
    # Each P row sums to zero
    for k in range(3):
        assert abs(P[k].sum()) < 1e-9


def test_build_views_q_equals_q_hat_for_all_rows():
    syms = [f"S{i:02d}" for i in range(20)]
    mom = {s: i * 0.01 for i, s in enumerate(syms)}
    prices = _prices_with_momentum(syms, mom)
    turnover = _flat_turnover(syms, {s: 0.01 for s in syms})
    _, Q, _ = views.build_views(prices, turnover, syms, "20260721", q_hat=0.07)
    assert np.allclose(Q, 0.07)


def test_build_views_momentum_top_decile_is_long():
    """Highest 20d return should end up on the long side of the momentum row."""
    syms = [f"S{i:02d}" for i in range(20)]
    mom = {s: i * 0.01 for i, s in enumerate(syms)}  # S19 is the biggest winner
    prices = _prices_with_momentum(syms, mom)
    turnover = _flat_turnover(syms, {s: 0.01 for s in syms})
    P, _, mem = views.build_views(prices, turnover, syms, "20260721", q_hat=0.05)
    # Column index for S19
    j = syms.index("S19")
    assert P[0, j] > 0  # momentum row, long side
    assert mem.loc["S19", "in_view_mom"] == 1
    # S00 should be on the short side
    j0 = syms.index("S00")
    assert P[0, j0] < 0
    assert mem.loc["S00", "in_view_mom"] == -1


def test_build_views_reversal_top_decile_is_short_by_default():
    """5d cumulative return: highest = bearish (short side of P row)."""
    syms = [f"S{i:02d}" for i in range(20)]
    # symbols already have monotone ramp; last 5 days of ramp share the same slope,
    # so mom & rev orderings coincide for this fixture.
    mom = {s: i * 0.01 for i, s in enumerate(syms)}
    prices = _prices_with_momentum(syms, mom)
    turnover = _flat_turnover(syms, {s: 0.01 for s in syms})
    P, _, mem = views.build_views(prices, turnover, syms, "20260721", q_hat=0.05)
    j = syms.index("S19")
    assert P[1, j] < 0  # reversal row, S19 (biggest 5d winner) is SHORT
    assert mem.loc["S19", "in_view_rev"] == -1


def test_build_views_turnover_top_decile_is_short_by_default():
    syms = [f"S{i:02d}" for i in range(20)]
    mom = {s: 0.0 for s in syms}
    prices = _prices_with_momentum(syms, mom)
    turnover = _flat_turnover(syms, {s: 0.001 * i for i, s in enumerate(syms)})  # S19 highest
    P, _, mem = views.build_views(prices, turnover, syms, "20260721", q_hat=0.05)
    j = syms.index("S19")
    assert P[2, j] < 0  # turnover row, S19 (highest turnover) is SHORT
    assert mem.loc["S19", "in_view_turnover"] == -1


def test_build_views_flip_mom_inverts_that_row_only():
    syms = [f"S{i:02d}" for i in range(20)]
    mom = {s: i * 0.01 for i, s in enumerate(syms)}
    prices = _prices_with_momentum(syms, mom)
    turnover = _flat_turnover(syms, {s: 0.001 for s in syms})
    P_base, _, mem_base = views.build_views(prices, turnover, syms, "20260721")
    P_flip, _, mem_flip = views.build_views(prices, turnover, syms, "20260721", flip_mom=True)
    # Mom row (0) is negated
    assert np.allclose(P_flip[0], -P_base[0])
    # Other rows unchanged
    assert np.allclose(P_flip[1], P_base[1])
    assert np.allclose(P_flip[2], P_base[2])
    # Membership sign flipped for mom column only
    j = syms.index("S19")
    assert mem_flip.loc["S19", "in_view_mom"] == -mem_base.loc["S19", "in_view_mom"]


def test_build_views_no_lookahead_ignores_dates_ge_scan_date():
    """Injecting a huge T-day return must NOT influence the momentum ranking."""
    syms = [f"S{i:02d}" for i in range(20)]
    mom = {s: i * 0.01 for i, s in enumerate(syms)}
    prices = _prices_with_momentum(syms, mom)
    # Add a HUGE close on scan_date for S00 (would become #1 if lookahead leaked)
    inject = pd.DataFrame([{"symbol": "S00", "date": "20260721", "close": 1e6}])
    prices = pd.concat([prices, inject], ignore_index=True)
    turnover = _flat_turnover(syms, {s: 0.01 for s in syms})
    P, _, _ = views.build_views(prices, turnover, syms, "20260721", q_hat=0.05)
    # S00 should still be on the SHORT side of mom (its pre-T returns were the lowest).
    j0 = syms.index("S00")
    assert P[0, j0] < 0
