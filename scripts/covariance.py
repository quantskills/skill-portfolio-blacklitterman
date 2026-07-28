"""Ledoit-Wolf covariance estimate on daily-return matrix.

Window: `cov_lookback` most recent trading days strictly before scan_date.
Output is annualized by multiplying by 252. Symbols with any NaN in their
return series are dropped from the returned matrix (design §9 test spec).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


TRADING_DAYS_PER_YEAR = 252


def ledoit_wolf_cov(
    prices_df: pd.DataFrame,
    symbols: list[str],
    scan_date: str,
    cov_lookback: int = 252,
) -> pd.DataFrame:
    """Return annualized LW-shrunk covariance for `symbols` on `scan_date`.

    Args:
        prices_df: long-form DataFrame with columns [symbol, date, close].
        symbols: ordered universe list. Output cov's row/col order matches.
        scan_date: 'YYYYMMDD'. Only dates strictly less than scan_date are used.
        cov_lookback: number of most-recent valid trading days per symbol.

    Returns:
        DataFrame with index/columns = kept symbols (in the input order,
        possibly a subset if some had NaNs).
    """
    # Pivot to date × symbol
    df = prices_df[prices_df["date"] < scan_date]
    wide = (
        df.pivot(index="date", columns="symbol", values="close")
          .sort_index()
    )
    # Take intersection with `symbols`, preserve caller order
    present = [s for s in symbols if s in wide.columns]
    wide = wide[present]
    # Keep only the last cov_lookback+1 rows (need +1 to compute cov_lookback returns)
    wide = wide.tail(cov_lookback + 1)
    # Daily simple returns
    rets = wide.pct_change(fill_method=None).dropna(how="all").iloc[-cov_lookback:]
    # Drop any symbol with any NaN in the return window
    rets = rets.dropna(axis=1, how="any")
    kept = list(rets.columns)
    if not kept:
        return pd.DataFrame()
    lw = LedoitWolf().fit(rets.values)
    cov = lw.covariance_ * TRADING_DAYS_PER_YEAR
    return pd.DataFrame(cov, index=kept, columns=kept)
