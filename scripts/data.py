"""panda_data thin wrappers for skill-portfolio-blacklitterman.

Three interfaces (see references/need_used_api.md):
  - get_index_weights  → load_prior      (先验权重)
  - get_stock_daily    → load_prices     (协方差 + 动量/反转视图)
  - get_factor         → load_turnover   (换手率视图)

panda_data is imported lazily inside each function so this module can be
imported without panda_data installed — useful for unit-testing callers
that mock the loaders.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

EXPECTED_COLUMNS: dict[str, set[str]] = {
    # `weight` is technically required, but load_prior falls back to equal
    # weights + WARN if it's missing (see spec §4.2 caveat), so we keep it
    # out of the assertion set for the prior. `stock_symbol` is remapped to
    # `symbol` inside load_prior.
    "prior":    {"stock_symbol", "date"},
    "prices":   {"symbol", "date", "close"},
    "turnover": {"symbol", "date", "turnover"},
}


def init_panda_data() -> None:
    """Authenticate with panda_data via env vars. Raises RuntimeError if unset."""
    user = os.environ.get("PANDA_DATA_USERNAME")
    pwd = os.environ.get("PANDA_DATA_PASSWORD")
    if not user or not pwd:
        raise RuntimeError(
            "Missing env vars PANDA_DATA_USERNAME / PANDA_DATA_PASSWORD. "
            "Export them before running the portfolio."
        )
    import panda_data
    panda_data.init_token(username=user, password=pwd)


def _assert_columns(df: pd.DataFrame, kind: str) -> None:
    expected = EXPECTED_COLUMNS[kind]
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"panda_data {kind} response missing columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}."
        )


def load_prior(date: str, index_symbol: str = "000300.SH") -> pd.DataFrame:
    """Fetch index constituent weights for a single date.

    Returns DataFrame with columns [symbol, date, weight]. If panda_data omits
    `weight`, falls back to equal weights over the returned constituents and
    emits a one-line WARN on stderr (spec §4.2 caveat).
    """
    import panda_data
    df = panda_data.get_index_weights(
        index_symbol=index_symbol,
        stock_symbol="",
        start_date=date,
        end_date=date,
        fields=None,  # request all columns, especially `weight`
    )
    if df is None or (hasattr(df, "empty") and df.empty):
        raise ValueError(f"no get_index_weights data for {index_symbol} on {date}")
    _assert_columns(df, "prior")
    df = df.copy()
    df["date"] = df["date"].astype(str)
    df["symbol"] = df["stock_symbol"].astype(str)
    if "weight" not in df.columns:
        print(
            f"[warn] get_index_weights returned no `weight` column — falling back to "
            f"equal weights across {len(df)} constituents",
            file=sys.stderr,
        )
        df["weight"] = 1.0 / len(df)
    return df[["symbol", "date", "weight"]]


def load_prices(start: str, end: str, symbols: list[str]) -> pd.DataFrame:
    """Fetch daily closes for a symbol list over [start, end]."""
    import panda_data
    df = panda_data.get_stock_daily(
        symbol=symbols,
        start_date=start,
        end_date=end,
        fields=["symbol", "date", "close"],
    )
    if df is None or (hasattr(df, "empty") and df.empty):
        return pd.DataFrame(columns=["symbol", "date", "close"])
    _assert_columns(df, "prices")
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str)
    df["date"] = df["date"].astype(str)
    return df[["symbol", "date", "close"]]


def load_turnover(start: str, end: str, symbols: list[str]) -> pd.DataFrame:
    """Fetch daily turnover for a symbol list over [start, end]."""
    import panda_data
    df = panda_data.get_factor(
        symbol=symbols,
        start_date=start,
        end_date=end,
        factors=["turnover"],
        type="stock",
    )
    if df is None or (hasattr(df, "empty") and df.empty):
        return pd.DataFrame(columns=["symbol", "date", "turnover"])
    _assert_columns(df, "turnover")
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str)
    df["date"] = df["date"].astype(str)
    return df[["symbol", "date", "turnover"]]


def self_check(date: str, index_symbol: str = "000300.SH") -> int:
    """Manually invoke each loader for `date` and print column diagnostics.

    Returns 0 on success, 4 on any missing required column.
    """
    init_panda_data()
    import panda_data
    exit_code = 0
    probes = (
        ("prior",    lambda: panda_data.get_index_weights(
            index_symbol=index_symbol, stock_symbol="",
            start_date=date, end_date=date, fields=None)),
        ("prices",   lambda: panda_data.get_stock_daily(
            symbol=None, start_date=date, end_date=date,
            fields=["symbol", "date", "close"])),
        ("turnover", lambda: panda_data.get_factor(
            symbol=None, start_date=date, end_date=date,
            factors=["turnover"], type="stock")),
    )
    for kind, loader in probes:
        print(f"--- {kind} ---")
        try:
            df = loader()
        except Exception as e:
            print(f"[ERROR] {kind} raised: {e}")
            exit_code = 4
            continue
        if df is None or (hasattr(df, "empty") and df.empty):
            print(f"[WARN] {kind} returned empty on {date}")
            continue
        got = set(df.columns)
        expected = EXPECTED_COLUMNS[kind]
        missing = expected - got
        extra = got - expected
        print(f"got columns:      {sorted(got)}")
        print(f"missing required: {sorted(missing)}")
        print(f"extra (ignored):  {sorted(extra)}")
        if missing:
            exit_code = 4
    return exit_code


def _main() -> int:
    p = argparse.ArgumentParser(
        description="panda_data field self-check for skill-portfolio-blacklitterman"
    )
    p.add_argument("--self-check", action="store_true", required=True)
    p.add_argument("--date", required=True, help="YYYYMMDD")
    p.add_argument("--index_symbol", default="000300.SH")
    args = p.parse_args()

    try:
        from panda_data.exceptions import ServiceError as _ServiceError
        service_error_cls: tuple = (_ServiceError,)
    except ImportError:
        service_error_cls = ()

    try:
        return self_check(args.date, index_symbol=args.index_symbol)
    except RuntimeError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    except service_error_cls as e:  # type: ignore[misc]
        print(f"[error] panda_data service error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(_main())
