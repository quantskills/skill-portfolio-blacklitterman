"""沪深300 ∩ data-available universe filter.

Rules (spec §5):
  1. Start from prior_df[date == scan_date].symbol.
  2. Drop symbols with fewer than min_valid_days non-null closes in the
     1-year window ending T-1 (strictly before scan_date).
  3. Drop symbols with weight == 0 or NaN.
  4. Return sorted symbol list + renormalized weight Series (sums to 1).
"""
from __future__ import annotations

import pandas as pd


def filter_universe(
    prior_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    scan_date: str,
    min_valid_days: int = 200,
) -> tuple[list[str], pd.Series]:
    if prior_df.empty:
        return [], pd.Series(dtype=float, name="weight")
    prior_t = prior_df[prior_df["date"] == scan_date]
    # Drop zero / NaN weights
    prior_t = prior_t[prior_t["weight"].notna() & (prior_t["weight"] > 0)]

    if prior_t.empty:
        return [], pd.Series(dtype=float, name="weight")

    # Count valid closes strictly before scan_date
    pre_t = prices_df[prices_df["date"] < scan_date]
    valid_counts = (
        pre_t.dropna(subset=["close"])
             .groupby("symbol", sort=False)
             .size()
    )

    kept = []
    weights = []
    for _, row in prior_t.iterrows():
        sym = row["symbol"]
        if valid_counts.get(sym, 0) >= min_valid_days:
            kept.append(sym)
            weights.append(row["weight"])

    if not kept:
        return [], pd.Series(dtype=float, name="weight")

    total = float(sum(weights))
    w = pd.Series(
        [x / total for x in weights],
        index=kept,
        name="weight",
    )
    # Sort by symbol for determinism (matches ETF-radar's sorted-universe convention).
    w = w.sort_index()
    return list(w.index), w
