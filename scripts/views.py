"""Build BL views P (K×N) and Q (K,) from three cross-sectional factors.

Sign conventions (defaults; overridable via flip_* flags — spec §6.2):
  - Momentum (20d cumulative return):     high = bullish  →  P row: +
  - Reversal (5d cumulative return):       high = bearish  →  sign-flipped
  - Turnover (20d avg turnover):           high = bearish  →  sign-flipped

Each view is a unit-notional decile spread:
  P_k[i] = +1/|L| if i ∈ top decile L
         = -1/|S| if i ∈ bottom decile S
         = 0      otherwise

All Q_k are set to `q_hat` (a single global magnitude, spec §6.2). Sign
flips are applied to the P row (not Q), so Q_k stays positive and
CSV-readable.

Membership DataFrame records +1/0/-1 per (symbol, view) AFTER sign flip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _factor_score_return(
    prices_df: pd.DataFrame,
    symbols: list[str],
    scan_date: str,
    lookback: int,
) -> pd.Series:
    """Cumulative return over the `lookback` trading days ending T-1."""
    df = prices_df[prices_df["date"] < scan_date]
    wide = (
        df.pivot(index="date", columns="symbol", values="close")
          .sort_index()
    )
    wide = wide[[s for s in symbols if s in wide.columns]]
    wide = wide.tail(lookback + 1)
    if len(wide) < 2:
        return pd.Series(index=symbols, dtype=float)
    ret = wide.iloc[-1] / wide.iloc[0] - 1.0
    return ret.reindex(symbols)


def _factor_score_turnover(
    turnover_df: pd.DataFrame,
    symbols: list[str],
    scan_date: str,
    lookback: int,
) -> pd.Series:
    """Mean turnover over the `lookback` trading days ending T-1."""
    df = turnover_df[turnover_df["date"] < scan_date]
    wide = (
        df.pivot(index="date", columns="symbol", values="turnover")
          .sort_index()
    )
    wide = wide[[s for s in symbols if s in wide.columns]]
    wide = wide.tail(lookback)
    return wide.mean(axis=0).reindex(symbols)


def _decile_row(score: pd.Series, symbols: list[str], sign: int) -> tuple[np.ndarray, np.ndarray]:
    """Build one P row and its membership vector from a factor score.

    Args:
        score: pd.Series indexed by symbol, higher = "raw bullish".
        symbols: full universe order (P row has this column order).
        sign: +1 if raw high = long-side; -1 if raw high = short-side.

    Returns:
        (row, membership) where row.shape == (N,), membership.shape == (N,) in {+1, 0, -1}.
    """
    N = len(symbols)
    row = np.zeros(N, dtype=float)
    membership = np.zeros(N, dtype=int)
    valid = score.dropna()
    if len(valid) < 10:
        return row, membership
    n = len(valid)
    k = max(1, n // 10)
    ranked = valid.sort_values(ascending=False)
    top = set(ranked.head(k).index)  # highest raw score
    bot = set(ranked.tail(k).index)  # lowest raw score

    long_side = top if sign == +1 else bot
    short_side = bot if sign == +1 else top

    L, S = len(long_side), len(short_side)
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    for s in long_side:
        i = sym_to_idx.get(s)
        if i is not None:
            row[i] = 1.0 / L
            membership[i] = +1
    for s in short_side:
        i = sym_to_idx.get(s)
        if i is not None:
            row[i] = -1.0 / S
            membership[i] = -1
    return row, membership


def build_views(
    prices_df: pd.DataFrame,
    turnover_df: pd.DataFrame,
    symbols: list[str],
    scan_date: str,
    *,
    mom_lookback: int = 20,
    rev_lookback: int = 5,
    turnover_lookback: int = 20,
    q_hat: float = 0.05,
    flip_mom: bool = False,
    flip_rev: bool = False,
    flip_turnover: bool = False,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    mom_score = _factor_score_return(prices_df, symbols, scan_date, mom_lookback)
    rev_score = _factor_score_return(prices_df, symbols, scan_date, rev_lookback)
    tv_score  = _factor_score_turnover(turnover_df, symbols, scan_date, turnover_lookback)

    # Base signs before user flips: mom → +1 (high = long); rev → -1 (high = short);
    # turnover → -1 (high = short).
    mom_sign = -1 if flip_mom else +1
    rev_sign = +1 if flip_rev else -1
    tv_sign  = +1 if flip_turnover else -1

    row_mom, mem_mom = _decile_row(mom_score, symbols, mom_sign)
    row_rev, mem_rev = _decile_row(rev_score, symbols, rev_sign)
    row_tv,  mem_tv  = _decile_row(tv_score,  symbols, tv_sign)

    P = np.vstack([row_mom, row_rev, row_tv])
    Q = np.array([q_hat, q_hat, q_hat], dtype=float)
    mem = pd.DataFrame(
        {"in_view_mom": mem_mom, "in_view_rev": mem_rev, "in_view_turnover": mem_tv},
        index=symbols,
    )
    return P, Q, mem
