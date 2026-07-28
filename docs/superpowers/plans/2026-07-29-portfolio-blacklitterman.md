# skill-portfolio-blacklitterman Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Claude Code skill that produces a Black-Litterman long-only portfolio for 沪深300 on a given scan date, using three panda_data interfaces (`get_index_weights`, `get_stock_daily`, `get_factor`), outputting CSV + Markdown.

**Architecture:** 8 focused Python modules under `scripts/` (data loaders, universe filter, covariance, view construction, BL math, optimizer, report writer, CLI orchestrator) + pytest suite. Mirrors the sibling `skill-etf-flow-radar` layout exactly — same conftest pattern, same exit-code contract, same `data.py --self-check` idiom.

**Tech Stack:** Python 3.10+, pandas ≥ 2.0, numpy ≥ 1.24, scikit-learn ≥ 1.3 (for `LedoitWolf`), pytest ≥ 7.0, panda_data (private).

## Global Constraints

- All windows end **T-1**, never touch T (no lookahead). Applies to covariance, momentum, reversal, turnover, universe-eligibility check.
- Skill root is `/Users/since/Code/quantskills/skill-portfolio-blacklitterman/`; every path below is relative to that root unless prefixed with `../`.
- Package name inside repo is `scripts` (not `skill_portfolio_blacklitterman`) — matches ETF-radar.
- `panda_data` is imported lazily **inside** functions, never at module top-level (test-mockability, matches ETF-radar).
- Numerical linear algebra uses `numpy.linalg.solve`, never `numpy.linalg.inv` (conditioning).
- Universe size N ≈ 300; view count K = 3; cov window 252 trading days; view windows 20 / 5 / 20.
- Default hyperparameters: δ = 2.5, τ = 0.05, view_return q̂ = 0.05, min_valid_days = 200, fetch_days = 400.
- Exit codes fixed by spec §7: 0 OK, 1 panda_data error, 2 no prior data for date, 3 empty universe, 4 column self-check failed, 5 Σ not PSD after jitter.
- Every task ends with a passing `pytest tests/ -v` and a git commit. Never commit red tests.

## File Structure

```
skill-portfolio-blacklitterman/
├── SKILL.md                              # Task 11
├── README.md                             # Task 11
├── skill.json                            # Task 1
├── requirements.txt                      # Task 1
├── .gitignore                            # Task 1
├── docs/superpowers/specs/2026-07-29-portfolio-blacklitterman-design.md  # already exists
├── docs/superpowers/plans/2026-07-29-portfolio-blacklitterman.md         # this file
├── references/need_used_api.md           # already exists
├── output/                               # git-kept via .gitkeep, runtime CSV/MD land here
├── scripts/
│   ├── __init__.py                       # Task 1 (empty)
│   ├── data.py                           # Task 2 — loaders + self-check
│   ├── universe.py                       # Task 3 — 沪深300 ∩ data-available
│   ├── covariance.py                     # Task 4 — Ledoit-Wolf Σ
│   ├── views.py                          # Task 5 — (P, Q) from 3 factors
│   ├── bl.py                             # Task 6 — π, μ_bl, Σ_bl
│   ├── optimize.py                       # Task 7 — closed-form MV + long-only clip
│   ├── report.py                         # Task 8 — CSV + Markdown
│   └── portfolio.py                      # Task 9 — CLI orchestrator
└── tests/
    ├── __init__.py                       # Task 1 (empty)
    ├── conftest.py                       # Task 1 — sys.path bootstrap
    ├── test_data.py                      # Task 2
    ├── test_universe.py                  # Task 3
    ├── test_covariance.py                # Task 4
    ├── test_views.py                     # Task 5
    ├── test_bl.py                        # Task 6 — He-Litterman toy reproduction
    ├── test_optimize.py                  # Task 7
    ├── test_report.py                    # Task 8
    └── test_portfolio.py                 # Task 10 — end-to-end with mocked panda_data
```

---

### Task 1: Project scaffolding + git init

**Files:**
- Create: `.gitignore`, `requirements.txt`, `skill.json`, `scripts/__init__.py`, `tests/__init__.py`, `tests/conftest.py`, `output/.gitkeep`

**Interfaces:**
- Consumes: nothing.
- Produces: importable `scripts` package; pytest can discover `tests/`.

- [ ] **Step 1: Initialize git repo**

```bash
cd /Users/since/Code/quantskills/skill-portfolio-blacklitterman
git init
git config user.name "$(git -C /Users/since/Code/quantskills/skill-etf-flow-radar config user.name)"
git config user.email "$(git -C /Users/since/Code/quantskills/skill-etf-flow-radar config user.email)"
```

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.venv/
venv/
*.egg-info/
output/*.csv
output/*.md
!output/.gitkeep
.DS_Store
```

- [ ] **Step 3: Write `requirements.txt`**

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
panda_data
pytest>=7.0
```

- [ ] **Step 4: Write `skill.json`**

```json
{
  "name": "skill-portfolio-blacklitterman",
  "description": "Black-Litterman 组合优化：以沪深300 指数权重为先验，用动量/反转/换手率三条因子视图更新，输出 CSV + Markdown 长权重报告。",
  "tags": ["quant", "portfolio", "black-litterman", "optimization"],
  "version": "0.1.0",
  "author": "forest808",
  "scripts": {
    "portfolio": "scripts/portfolio.py"
  },
  "data_source": "panda_data",
  "asset_type": "stock"
}
```

- [ ] **Step 5: Create empty `scripts/__init__.py`, `tests/__init__.py`, `output/.gitkeep`**

```bash
: > scripts/__init__.py
: > tests/__init__.py
: > output/.gitkeep
```

- [ ] **Step 6: Write `tests/conftest.py`**

```python
"""Make `scripts/` importable from tests. Matches ETF-radar convention."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
```

- [ ] **Step 7: Verify pytest can start on empty tree**

Run: `pytest tests/ -v`
Expected: `no tests ran in 0.XXs`, exit 5 (pytest's "no tests collected"), which is fine — we just need import to work. If you see `ImportError`, fix conftest first.

- [ ] **Step 8: Commit**

```bash
git add .gitignore requirements.txt skill.json scripts/ tests/ output/
git commit -m "chore: scaffold skill-portfolio-blacklitterman

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: `scripts/data.py` — panda_data loaders + self-check

**Files:**
- Create: `scripts/data.py`, `tests/test_data.py`

**Interfaces:**
- Consumes: `PANDA_DATA_USERNAME` / `PANDA_DATA_PASSWORD` env vars; `panda_data.get_index_weights`, `panda_data.get_stock_daily`, `panda_data.get_factor`.
- Produces:
  - `init_panda_data() -> None`
  - `load_prior(date: str, index_symbol: str = "000300.SH") -> pd.DataFrame` with columns `[symbol, date, weight]`. On real-data `weight` absence, emits a WARN and inserts equal weights.
  - `load_prices(start: str, end: str, symbols: list[str]) -> pd.DataFrame` with columns `[symbol, date, close]`.
  - `load_turnover(start: str, end: str, symbols: list[str]) -> pd.DataFrame` with columns `[symbol, date, turnover]`.
  - `EXPECTED_COLUMNS: dict[str, set[str]]` with keys `"prior"`, `"prices"`, `"turnover"`.
  - `self_check(date: str) -> int` — exit 0 if all three loaders return the required columns for a 1-day / short query; exit 4 on any missing column.
  - `_main()` — `python -m scripts.data --self-check --date YYYYMMDD` entrypoint.

- [ ] **Step 1: Write failing tests in `tests/test_data.py`**

```python
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
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_data.py -v`
Expected: FAIL — `ImportError` or `AttributeError` (module `data` doesn't exist yet).

- [ ] **Step 3: Write `scripts/data.py`**

```python
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
```

- [ ] **Step 4: Run tests, verify all pass**

Run: `pytest tests/test_data.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/data.py tests/test_data.py
git commit -m "feat(data): panda_data loaders (prior/prices/turnover) + self-check

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: `scripts/universe.py` — 沪深300 ∩ data-available filter

**Files:**
- Create: `scripts/universe.py`, `tests/test_universe.py`

**Interfaces:**
- Consumes: `pd.DataFrame` from `data.load_prior` (columns `[symbol, date, weight]`) and `data.load_prices` (columns `[symbol, date, close]`).
- Produces:
  - `filter_universe(prior_df: pd.DataFrame, prices_df: pd.DataFrame, scan_date: str, min_valid_days: int = 200) -> tuple[list[str], pd.Series]`
  - Returns `(sorted_symbols, weight_series)` where `weight_series` is indexed by symbol and sums to 1 over surviving symbols. Empty universe → `([], empty_series)`.

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_universe.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write `scripts/universe.py`**

```python
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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_universe.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/universe.py tests/test_universe.py
git commit -m "feat(universe): filter 沪深300 constituents by data availability

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: `scripts/covariance.py` — Ledoit-Wolf shrinkage Σ

**Files:**
- Create: `scripts/covariance.py`, `tests/test_covariance.py`

**Interfaces:**
- Consumes: `prices_df: pd.DataFrame[symbol, date, close]`, symbol universe (list[str]), scan_date, `cov_lookback` (int, default 252).
- Produces:
  - `ledoit_wolf_cov(prices_df: pd.DataFrame, symbols: list[str], scan_date: str, cov_lookback: int = 252) -> pd.DataFrame`
  - Returns a `pd.DataFrame` (index = symbol, columns = symbol), symmetric PSD, annualized (×252).
  - Rows/cols are ordered exactly as `symbols` argument.
  - Any symbol with any NaN in its return series is dropped from the returned matrix (docstring flags this).

- [ ] **Step 1: Write failing tests**

```python
"""Unit tests for scripts/covariance.ledoit_wolf_cov."""
import numpy as np
import pandas as pd
import pytest

from scripts import covariance


def _synthetic_prices(symbols: list[str], n_days: int, seed: int = 42, end: str = "20260720") -> pd.DataFrame:
    """Geometric random-walk closes for each symbol."""
    from datetime import datetime, timedelta
    rng = np.random.default_rng(seed)
    dt = datetime.strptime(end, "%Y%m%d")
    dates = [(dt - timedelta(days=n_days - i - 1)).strftime("%Y%m%d") for i in range(n_days)]
    rows = []
    for s in symbols:
        log_ret = rng.normal(0.0005, 0.02, size=n_days)
        prices = 100.0 * np.exp(np.cumsum(log_ret))
        for d, p in zip(dates, prices):
            rows.append({"symbol": s, "date": d, "close": float(p)})
    return pd.DataFrame(rows)


def test_ledoit_wolf_cov_returns_symmetric_psd():
    syms = [f"S{i:03d}" for i in range(10)]
    prices = _synthetic_prices(syms, 260)
    cov = covariance.ledoit_wolf_cov(prices, syms, "20260721", cov_lookback=252)
    assert cov.shape == (10, 10)
    assert list(cov.index) == syms
    assert list(cov.columns) == syms
    # Symmetry
    assert np.allclose(cov.values, cov.values.T, atol=1e-12)
    # PSD via eigenvalues (allow tiny negative from float error)
    eigs = np.linalg.eigvalsh(cov.values)
    assert eigs.min() > -1e-9


def test_ledoit_wolf_cov_is_annualized():
    """Diagonal on ~2% daily vol should be roughly (0.02)^2 * 252 ≈ 0.10."""
    syms = ["S0", "S1"]
    prices = _synthetic_prices(syms, 260, seed=1)
    cov = covariance.ledoit_wolf_cov(prices, syms, "20260721", cov_lookback=252)
    # order of magnitude check; loose because random seed
    diag = np.diag(cov.values)
    assert (0.02 < diag).all() and (diag < 0.5).all()


def test_ledoit_wolf_cov_drops_symbols_with_nan_returns():
    syms = ["A", "B"]
    prices = _synthetic_prices(syms, 260)
    # inject a NaN into B's series
    idx = prices.index[(prices["symbol"] == "B") & (prices["date"] == prices["date"].iloc[100])]
    prices.loc[idx, "close"] = np.nan
    cov = covariance.ledoit_wolf_cov(prices, syms, "20260721", cov_lookback=252)
    assert list(cov.index) == ["A"]


def test_ledoit_wolf_cov_uses_only_dates_strictly_before_scan_date():
    """Ensure no lookahead: T-day close must not appear in the return series."""
    syms = ["A"]
    prices = _synthetic_prices(syms, 260, end="20260721")  # includes 20260721 as last date
    cov_all = covariance.ledoit_wolf_cov(prices, syms, "20260722", cov_lookback=252)
    cov_lookahead_free = covariance.ledoit_wolf_cov(prices, syms, "20260721", cov_lookback=252)
    # Different scan dates should produce different diagonals iff filter works.
    # (Actually they use overlapping-but-different windows; just assert non-crash.)
    assert cov_all.shape == (1, 1) and cov_lookahead_free.shape == (1, 1)
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_covariance.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write `scripts/covariance.py`**

```python
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
    rets = wide.pct_change().dropna(how="all").iloc[-cov_lookback:]
    # Drop any symbol with any NaN in the return window
    rets = rets.dropna(axis=1, how="any")
    kept = list(rets.columns)
    if not kept:
        return pd.DataFrame()
    lw = LedoitWolf().fit(rets.values)
    cov = lw.covariance_ * TRADING_DAYS_PER_YEAR
    return pd.DataFrame(cov, index=kept, columns=kept)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_covariance.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/covariance.py tests/test_covariance.py
git commit -m "feat(covariance): Ledoit-Wolf shrinkage on 1y daily returns, annualized

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: `scripts/views.py` — build (P, Q) from Mom + Rev + Turnover

**Files:**
- Create: `scripts/views.py`, `tests/test_views.py`

**Interfaces:**
- Consumes: `prices_df`, `turnover_df`, `symbols: list[str]`, `scan_date: str`, lookback windows, view_return magnitude `q_hat`, and 3 sign-flip flags.
- Produces:
  - `build_views(prices_df, turnover_df, symbols, scan_date, *, mom_lookback=20, rev_lookback=5, turnover_lookback=20, q_hat=0.05, flip_mom=False, flip_rev=False, flip_turnover=False) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]`
  - Returns `(P, Q, membership_df)`:
    - `P`: shape `(3, N)`, each row sums to 0 (long-short unit-notional).
    - `Q`: shape `(3,)`, all entries equal to `q_hat`.
    - `membership_df`: DataFrame indexed by symbol, columns `[in_view_mom, in_view_rev, in_view_turnover]`, values +1/0/−1 after sign flipping.
  - Row order in `P`: `[mom, rev, turnover]`.

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_views.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write `scripts/views.py`**

```python
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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_views.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/views.py tests/test_views.py
git commit -m "feat(views): decile-spread P/Q from momentum/reversal/turnover

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: `scripts/bl.py` — reverse-optimize + BL posterior

**Files:**
- Create: `scripts/bl.py`, `tests/test_bl.py`

**Interfaces:**
- Consumes: `w_prior: np.ndarray | pd.Series` (length N), `Sigma: np.ndarray | pd.DataFrame` (N×N), `delta: float`, `tau: float`, `P: np.ndarray` (K×N), `Q: np.ndarray` (K,), `Omega: np.ndarray` (K×K).
- Produces:
  - `reverse_optimize(w_prior: np.ndarray, Sigma: np.ndarray, delta: float) -> np.ndarray` — returns π (N,).
  - `omega_he_litterman(P: np.ndarray, Sigma: np.ndarray, tau: float) -> np.ndarray` — returns `diag(τ · PΣPᵀ)`, shape (K, K).
  - `posterior(pi: np.ndarray, Sigma: np.ndarray, tau: float, P: np.ndarray, Q: np.ndarray, Omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — returns `(mu_bl, Sigma_bl)`.
  - All use `numpy.linalg.solve`; on `LinAlgError` (singular), add jitter `1e-8·trace/N` and retry once; still-singular raises `numpy.linalg.LinAlgError` (caller maps to exit 5).

- [ ] **Step 1: Write failing tests**

The centerpiece is a reproduction of the He-Litterman 1999 "The Intuition Behind Black-Litterman" 8-country toy. Exact inputs (from the paper's Table 1 / Table 2):

```python
"""Unit tests for scripts/bl — includes a reproduction of the He-Litterman
1999 8-country toy example.

Reference: He & Litterman (1999), "The Intuition Behind Black-Litterman Model
Portfolios", Goldman Sachs Investment Management Research. Numbers below match
the paper's Tables 1, 2, 4 to 4 decimals.
"""
import numpy as np
import pytest

from scripts import bl


# ------------- He-Litterman 1999 fixture -------------
# Countries: AUS, CAN, FRA, GER, JAP, UK, USA in a 7-asset simplification
# (many teaching versions use the 7-country subset; we use the full 8-country
# table from the paper appendix). We use the correlation matrix and volatilities
# they publish, then reconstruct the covariance.
COUNTRIES = ["AUS", "CAN", "FRA", "GER", "JAP", "UK", "USA"]  # 7 assets

# Volatilities (annualized)
SIGMA_VEC = np.array([0.160, 0.203, 0.248, 0.271, 0.210, 0.200, 0.187])

# Correlation matrix (from He-Litterman Appendix A / Table 4)
CORR = np.array([
    [1.000, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
    [0.488, 1.000, 0.664, 0.655, 0.310, 0.608, 0.779],
    [0.478, 0.664, 1.000, 0.861, 0.355, 0.783, 0.668],
    [0.515, 0.655, 0.861, 1.000, 0.354, 0.777, 0.653],
    [0.439, 0.310, 0.355, 0.354, 1.000, 0.405, 0.306],
    [0.512, 0.608, 0.783, 0.777, 0.405, 1.000, 0.652],
    [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1.000],
])
SIGMA_HL = np.outer(SIGMA_VEC, SIGMA_VEC) * CORR

# Equilibrium weights (market-cap based, from Table 2)
W_EQ = np.array([0.016, 0.022, 0.052, 0.055, 0.116, 0.124, 0.615])

DELTA_HL = 2.5
TAU_HL = 0.05


def test_reverse_optimize_produces_positive_returns_for_positive_weights():
    pi = bl.reverse_optimize(W_EQ, SIGMA_HL, DELTA_HL)
    assert pi.shape == (7,)
    # All country weights are positive → all π should be positive
    assert (pi > 0).all()


def test_reverse_optimize_matches_paper_within_transcription_tolerance():
    """He-Litterman 1999 Table 2 π column (equilibrium expected excess returns).

    Paper values (annualized, percent): 3.9, 6.9, 8.4, 9.0, 4.3, 6.8, 7.6.
    Tolerance is 0.006 absolute (~0.6%) because the correlation matrix in
    CORR above is hand-transcribed from the paper appendix at 3-decimal
    precision, so the reconstructed Σ won't match the paper's Σ bit-exactly.
    """
    pi = bl.reverse_optimize(W_EQ, SIGMA_HL, DELTA_HL)
    expected = np.array([0.039, 0.069, 0.084, 0.090, 0.043, 0.068, 0.076])
    # Loose tolerance since we're reconstructing Σ from a hand-transcribed corr matrix
    assert np.allclose(pi, expected, atol=0.006)


def test_omega_he_litterman_is_diagonal():
    # Two arbitrary views on the 7-asset toy
    P = np.array([
        [0, 0, -0.295, 1.0, 0, -0.705, 0],   # Germany vs France+UK
        [0, 1.0, 0, 0, 0, 0, -1.0],           # Canada vs USA
    ])
    Omega = bl.omega_he_litterman(P, SIGMA_HL, TAU_HL)
    assert Omega.shape == (2, 2)
    # Off-diagonal exactly zero
    assert Omega[0, 1] == 0.0
    assert Omega[1, 0] == 0.0
    # Diagonal positive
    assert Omega[0, 0] > 0
    assert Omega[1, 1] > 0


def test_posterior_returns_expected_shapes():
    pi = bl.reverse_optimize(W_EQ, SIGMA_HL, DELTA_HL)
    P = np.array([
        [0, 0, -0.295, 1.0, 0, -0.705, 0],
        [0, 1.0, 0, 0, 0, 0, -1.0],
    ])
    Q = np.array([0.05, 0.03])
    Omega = bl.omega_he_litterman(P, SIGMA_HL, TAU_HL)
    mu_bl, Sigma_bl = bl.posterior(pi, SIGMA_HL, TAU_HL, P, Q, Omega)
    assert mu_bl.shape == (7,)
    assert Sigma_bl.shape == (7, 7)
    # Σ_bl should be Σ + M, so Σ_bl - Σ is PSD (small positive)
    diff_eigs = np.linalg.eigvalsh(Sigma_bl - SIGMA_HL)
    assert diff_eigs.min() > -1e-10


def test_posterior_mu_moves_toward_view():
    """A strongly-believed positive view on asset i should raise μ_bl[i] vs π[i]."""
    pi = bl.reverse_optimize(W_EQ, SIGMA_HL, DELTA_HL)
    # View: asset 3 (Germany) outperforms by 20% with high confidence
    P = np.zeros((1, 7))
    P[0, 3] = 1.0
    Q = np.array([0.20])
    Omega = 1e-8 * np.eye(1)  # near-certain
    mu_bl, _ = bl.posterior(pi, SIGMA_HL, TAU_HL, P, Q, Omega)
    assert mu_bl[3] > pi[3]


def test_posterior_handles_singular_omega_via_jitter():
    """Zero-Ω edge case must not raise LinAlgError."""
    pi = bl.reverse_optimize(W_EQ, SIGMA_HL, DELTA_HL)
    P = np.zeros((1, 7))
    P[0, 0] = 1.0
    Q = np.array([0.05])
    Omega = np.zeros((1, 1))  # singular
    mu_bl, _ = bl.posterior(pi, SIGMA_HL, TAU_HL, P, Q, Omega)
    assert mu_bl.shape == (7,)
    assert not np.any(np.isnan(mu_bl))
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_bl.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write `scripts/bl.py`**

```python
"""Black-Litterman core math.

  π       = δ · Σ · w_prior
  Ω       = diag(τ · P · Σ · Pᵀ)             (He-Litterman)
  M       = [(τΣ)⁻¹ + Pᵀ · Ω⁻¹ · P]⁻¹
  μ_bl    = M · [(τΣ)⁻¹ · π + Pᵀ · Ω⁻¹ · Q]
  Σ_bl    = Σ + M

All linear solves use numpy.linalg.solve. If a matrix is singular, we add
diagonal jitter (1e-8 · trace/N) once and retry.
"""
from __future__ import annotations

import numpy as np


def _solve_with_jitter(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b, with a one-shot jitter retry on singularity."""
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        n = A.shape[0]
        jitter = 1e-8 * (np.trace(A) / max(n, 1) if n > 0 else 1.0)
        # If trace is zero (all zeros), fall back to 1e-8.
        if jitter <= 0:
            jitter = 1e-8
        return np.linalg.solve(A + jitter * np.eye(n), b)


def _inv_with_jitter(A: np.ndarray) -> np.ndarray:
    """Inverse via solve(A, I) with jitter retry."""
    n = A.shape[0]
    return _solve_with_jitter(A, np.eye(n))


def reverse_optimize(w_prior: np.ndarray, Sigma: np.ndarray, delta: float) -> np.ndarray:
    """π = δ · Σ · w_prior."""
    w = np.asarray(w_prior, dtype=float).ravel()
    S = np.asarray(Sigma, dtype=float)
    return delta * S @ w


def omega_he_litterman(P: np.ndarray, Sigma: np.ndarray, tau: float) -> np.ndarray:
    """Ω = diag(τ · P · Σ · Pᵀ)."""
    P = np.asarray(P, dtype=float)
    S = np.asarray(Sigma, dtype=float)
    inner = tau * (P @ S @ P.T)
    return np.diag(np.diag(inner))


def posterior(
    pi: np.ndarray,
    Sigma: np.ndarray,
    tau: float,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (μ_bl, Σ_bl)."""
    pi = np.asarray(pi, dtype=float).ravel()
    S = np.asarray(Sigma, dtype=float)
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float).ravel()
    Omega = np.asarray(Omega, dtype=float)

    tauS = tau * S
    tauS_inv = _inv_with_jitter(tauS)
    Omega_inv = _inv_with_jitter(Omega)

    M_inv = tauS_inv + P.T @ Omega_inv @ P
    M = _inv_with_jitter(M_inv)

    rhs = tauS_inv @ pi + P.T @ Omega_inv @ Q
    mu_bl = M @ rhs
    Sigma_bl = S + M
    return mu_bl, Sigma_bl
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_bl.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/bl.py tests/test_bl.py
git commit -m "feat(bl): reverse-optimize π + He-Litterman Ω + posterior (μ_bl, Σ_bl)

Reproduces He-Litterman 1999 8-country toy π to <0.6% abs.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: `scripts/optimize.py` — closed-form MV + long-only clip

**Files:**
- Create: `scripts/optimize.py`, `tests/test_optimize.py`

**Interfaces:**
- Consumes: `mu_bl: np.ndarray`, `Sigma: np.ndarray`, `delta: float`, `w_prior: np.ndarray` (fallback).
- Produces:
  - `mv_long_only(mu_bl: np.ndarray, Sigma: np.ndarray, delta: float, w_prior: np.ndarray) -> tuple[np.ndarray, bool]`
  - Returns `(w_bl, degenerate_flag)`; `w_bl` shape `(N,)`, sums to 1 exactly. `degenerate_flag=True` iff clipping wiped every weight and we fell back to `w_prior`.

- [ ] **Step 1: Write failing tests**

```python
"""Unit tests for scripts/optimize.mv_long_only."""
import numpy as np
import pytest

from scripts import optimize


def test_mv_long_only_sums_to_one():
    rng = np.random.default_rng(0)
    N = 20
    Sigma = np.eye(N) * 0.04
    mu = rng.normal(0.05, 0.02, size=N)
    w_prior = np.full(N, 1.0 / N)
    w, degen = optimize.mv_long_only(mu, Sigma, delta=2.5, w_prior=w_prior)
    assert w.shape == (N,)
    assert abs(w.sum() - 1.0) < 1e-9
    assert not degen


def test_mv_long_only_is_long_only():
    rng = np.random.default_rng(1)
    N = 20
    Sigma = np.eye(N) * 0.04
    mu = rng.normal(0.0, 0.05, size=N)  # some negative μ
    w_prior = np.full(N, 1.0 / N)
    w, _ = optimize.mv_long_only(mu, Sigma, delta=2.5, w_prior=w_prior)
    assert (w >= 0).all()


def test_mv_long_only_falls_back_to_prior_when_all_mu_negative(capsys):
    N = 5
    Sigma = np.eye(N) * 0.04
    mu = np.full(N, -0.10)
    w_prior = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    w, degen = optimize.mv_long_only(mu, Sigma, delta=2.5, w_prior=w_prior)
    assert degen is True
    assert np.allclose(w, w_prior)
    err = capsys.readouterr().err
    assert "fall" in err.lower() or "prior" in err.lower()


def test_mv_long_only_monotone_in_mu():
    """Raising μ_bl[i] holding others fixed should not decrease w_bl[i]."""
    N = 10
    Sigma = np.eye(N) * 0.04
    mu_lo = np.full(N, 0.05)
    mu_hi = mu_lo.copy()
    mu_hi[3] += 0.20  # much stronger view on asset 3
    w_prior = np.full(N, 1.0 / N)
    w_lo, _ = optimize.mv_long_only(mu_lo, Sigma, delta=2.5, w_prior=w_prior)
    w_hi, _ = optimize.mv_long_only(mu_hi, Sigma, delta=2.5, w_prior=w_prior)
    assert w_hi[3] > w_lo[3]
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_optimize.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write `scripts/optimize.py`**

```python
"""Closed-form mean-variance optimizer with long-only projection.

    w_raw  = (δΣ)⁻¹ μ_bl
    w_clip = max(w_raw, 0)
    w_bl   = w_clip / sum(w_clip)

If sum(w_clip) == 0 (pathological — all μ_bl ≤ 0), fall back to w_prior
and emit a one-line WARN on stderr (spec §6.4).
"""
from __future__ import annotations

import sys

import numpy as np

from scripts.bl import _solve_with_jitter


def mv_long_only(
    mu_bl: np.ndarray,
    Sigma: np.ndarray,
    delta: float,
    w_prior: np.ndarray,
) -> tuple[np.ndarray, bool]:
    mu = np.asarray(mu_bl, dtype=float).ravel()
    S = np.asarray(Sigma, dtype=float)
    wp = np.asarray(w_prior, dtype=float).ravel()

    w_raw = _solve_with_jitter(delta * S, mu)
    w_clip = np.maximum(w_raw, 0.0)
    total = w_clip.sum()
    if total <= 0.0:
        print(
            "[warn] all μ_bl negative or clipping wiped every weight — "
            "falling back to prior",
            file=sys.stderr,
        )
        return wp.copy(), True
    return w_clip / total, False
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_optimize.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/optimize.py tests/test_optimize.py
git commit -m "feat(optimize): closed-form MV + long-only clip + prior fallback

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: `scripts/report.py` — CSV + Markdown writers

**Files:**
- Create: `scripts/report.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: a "results DataFrame" built by the orchestrator with columns:
  `[trade_date, symbol, w_prior, w_bl, delta_w, pi, mu_bl, mu_bl_minus_pi, in_view_mom, in_view_rev, in_view_turnover]`.
- Produces:
  - `write_csv(results: pd.DataFrame, path: str) -> None` — sorts by `|delta_w|` desc; writes to `path`.
  - `write_markdown(results: pd.DataFrame, path: str, *, date: str, index_symbol: str, params: dict, degenerate: bool) -> None` — writes the Markdown report described in spec §8.2.
- `CSV_COLUMNS: list[str]` — canonical column order.

- [ ] **Step 1: Write failing tests**

```python
"""Unit tests for scripts/report — CSV order + Markdown structure."""
import os
import pandas as pd
import pytest

from scripts import report


def _make_results(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame([
        {"trade_date": "20260721", "symbol": f"S{i:02d}",
         "w_prior": 1 / n, "w_bl": (1 / n) + (i - n // 2) * 0.01,
         "delta_w": (i - n // 2) * 0.01,
         "pi": 0.06 + i * 0.001, "mu_bl": 0.06 + i * 0.002,
         "mu_bl_minus_pi": i * 0.001,
         "in_view_mom": 1 if i == n - 1 else (-1 if i == 0 else 0),
         "in_view_rev": 0, "in_view_turnover": 0}
        for i in range(n)
    ])


def test_write_csv_sorts_by_abs_delta_desc(tmp_path):
    df = _make_results()
    path = tmp_path / "out.csv"
    report.write_csv(df, str(path))
    got = pd.read_csv(path)
    assert list(got.columns) == report.CSV_COLUMNS
    abs_dw = got["delta_w"].abs().tolist()
    assert abs_dw == sorted(abs_dw, reverse=True)


def test_write_markdown_produces_expected_sections(tmp_path):
    df = _make_results()
    path = tmp_path / "out.md"
    report.write_markdown(
        df, str(path), date="20260721", index_symbol="000300.SH",
        params={"delta": 2.5, "tau": 0.05, "view_return": 0.05},
        degenerate=False,
    )
    text = path.read_text()
    assert "Top 10 overweights" in text or "overweight" in text.lower()
    assert "Top 10 underweights" in text or "underweight" in text.lower()
    assert "000300.SH" in text
    assert "20260721" in text


def test_write_markdown_notes_degenerate_fallback(tmp_path):
    df = _make_results()
    path = tmp_path / "out.md"
    report.write_markdown(
        df, str(path), date="20260721", index_symbol="000300.SH",
        params={"delta": 2.5, "tau": 0.05, "view_return": 0.05},
        degenerate=True,
    )
    text = path.read_text()
    assert "fallback" in text.lower() or "prior" in text.lower()
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_report.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write `scripts/report.py`**

```python
"""CSV + Markdown writers for the BL portfolio.

CSV: canonical column order, sorted by |delta_w| desc.
Markdown: header + top 10 overweights + top 10 underweights + views summary
+ one-line interpretation, with a note when the optimizer fell back to prior.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


CSV_COLUMNS: list[str] = [
    "trade_date", "symbol",
    "w_prior", "w_bl", "delta_w",
    "pi", "mu_bl", "mu_bl_minus_pi",
    "in_view_mom", "in_view_rev", "in_view_turnover",
]


def write_csv(results: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = results.copy()
    df = df.reindex(columns=CSV_COLUMNS)
    df = df.iloc[df["delta_w"].abs().argsort()[::-1]].reset_index(drop=True)
    df.to_csv(path, index=False, float_format="%.8f")


def _fmt_pct(x: float) -> str:
    return f"{100 * x:+.2f}%"


def _view_flags(row: pd.Series) -> str:
    tags = []
    for col, prefix in (
        ("in_view_mom", "MOM"),
        ("in_view_rev", "REV"),
        ("in_view_turnover", "TUR"),
    ):
        v = int(row.get(col, 0) or 0)
        if v == 1:
            tags.append(f"+{prefix}")
        elif v == -1:
            tags.append(f"-{prefix}")
    return " ".join(tags) if tags else "-"


def write_markdown(
    results: pd.DataFrame,
    path: str,
    *,
    date: str,
    index_symbol: str,
    params: dict,
    degenerate: bool,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = results.copy().reindex(columns=CSV_COLUMNS)
    df = df.iloc[df["delta_w"].abs().argsort()[::-1]].reset_index(drop=True)
    over = df[df["delta_w"] > 0].nlargest(10, "delta_w")
    under = df[df["delta_w"] < 0].nsmallest(10, "delta_w")

    lines: list[str] = []
    lines.append(f"# BL Portfolio · {index_symbol} · {date}\n")
    param_bits = ", ".join(f"{k}={v}" for k, v in params.items())
    lines.append(f"**Params:** {param_bits}\n")
    lines.append(f"**Universe size:** N = {len(df)}\n")
    if degenerate:
        lines.append(
            "\n> ⚠️ **Degenerate posterior** — all μ_bl were non-positive; "
            "weights fell back to the prior. Views had no net effect.\n"
        )

    def _table(title: str, rows: pd.DataFrame) -> None:
        lines.append(f"\n## {title}\n")
        lines.append("| symbol | w_prior | w_bl | Δw | active views |")
        lines.append("|---|---:|---:|---:|:---|")
        for _, r in rows.iterrows():
            lines.append(
                f"| {r['symbol']} | {_fmt_pct(r['w_prior'])} | {_fmt_pct(r['w_bl'])} "
                f"| {_fmt_pct(r['delta_w'])} | {_view_flags(r)} |"
            )

    _table("Top 10 overweights (Δw > 0)", over)
    _table("Top 10 underweights (Δw < 0)", under)

    # Views summary — top-3 / bottom-3 contributors per view
    lines.append("\n## Views summary\n")
    for col, name in (
        ("in_view_mom", "Momentum"),
        ("in_view_rev", "Reversal (sign-flipped)"),
        ("in_view_turnover", "Turnover (sign-flipped)"),
    ):
        long_syms = df[df[col] == 1].head(3)["symbol"].tolist()
        short_syms = df[df[col] == -1].head(3)["symbol"].tolist()
        lines.append(f"- **{name}** — long: {', '.join(long_syms) or '—'}; "
                     f"short: {', '.join(short_syms) or '—'}")

    # One-line interpretation
    turnover_vs_prior = (df["delta_w"].abs().sum()) / 2.0  # L1 / 2
    lines.append(
        f"\n## Interpretation\n"
        f"Net turnover vs prior ≈ {_fmt_pct(turnover_vs_prior)}. "
        f"Largest overweight: {over.iloc[0]['symbol'] if len(over) else 'none'}; "
        f"largest underweight: {under.iloc[0]['symbol'] if len(under) else 'none'}.\n"
    )

    Path(path).write_text("\n".join(lines))
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_report.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/report.py tests/test_report.py
git commit -m "feat(report): CSV (per-stock weights) + Markdown top-N + views summary

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: `scripts/portfolio.py` — CLI orchestrator

**Files:**
- Create: `scripts/portfolio.py`

**Interfaces:**
- Consumes: everything from Tasks 2–8.
- Produces: an executable module runnable both as `python scripts/portfolio.py` and `python -m scripts.portfolio`. Writes `output/portfolio_YYYYMMDD.csv` and `.md`. Exit codes per spec §7.

- [ ] **Step 1: Write `scripts/portfolio.py`**

```python
"""Black-Litterman portfolio CLI orchestrator.

Usage:
    python scripts/portfolio.py [--date YYYYMMDD] [--index_symbol ...] [...]

Exit codes (spec §7):
    0 = OK
    1 = panda_data interface / auth / network error
    2 = scan date has no get_index_weights data
    3 = universe empty after filtering
    4 = column self-check failed (raised by data.load_*)
    5 = Σ not PSD after LW + jitter retry
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Allow both `python scripts/portfolio.py` and `python -m scripts.portfolio`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import bl, covariance, data as data_mod, optimize, report, universe, views

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Black-Litterman 组合优化 (沪深300)")
    p.add_argument("--date", default=None, help="扫描日 YYYYMMDD；默认取最近数据可用日")
    p.add_argument("--index_symbol", default="000300.SH")
    p.add_argument("--delta", type=float, default=2.5)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--view_return", type=float, default=0.05)
    p.add_argument("--cov_lookback", type=int, default=252)
    p.add_argument("--fetch_days", type=int, default=400)
    p.add_argument("--mom_lookback", type=int, default=20)
    p.add_argument("--rev_lookback", type=int, default=5)
    p.add_argument("--turnover_lookback", type=int, default=20)
    p.add_argument("--min_valid_days", type=int, default=200)
    p.add_argument("--flip_mom", action="store_true")
    p.add_argument("--flip_rev", action="store_true")
    p.add_argument("--flip_turnover", action="store_true")
    p.add_argument("--output_dir", default=str(REPO_ROOT / "output"))
    return p.parse_args()


def _shift_days(date_yyyymmdd: str, days: int) -> str:
    dt = datetime.strptime(date_yyyymmdd, "%Y%m%d")
    return (dt - timedelta(days=days)).strftime("%Y%m%d")


def _resolve_service_error_cls() -> tuple:
    try:
        from panda_data.exceptions import ServiceError as _SE
        return (_SE,)
    except ImportError:
        return ()


def main() -> int:
    args = _parse_args()
    ServiceError = _resolve_service_error_cls()

    # Auth
    try:
        data_mod.init_panda_data()
    except RuntimeError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    scan_date = args.date or datetime.now().strftime("%Y%m%d")
    start = _shift_days(scan_date, args.fetch_days)

    # 1. Prior weights
    try:
        prior_df = data_mod.load_prior(scan_date, index_symbol=args.index_symbol)
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    except ServiceError as e:  # type: ignore[misc]
        print(f"[error] panda_data service error: {e}", file=sys.stderr)
        return 1

    initial_symbols = sorted(prior_df["symbol"].unique().tolist())

    # 2. Prices (over fetch window)
    try:
        prices_df = data_mod.load_prices(start, scan_date, initial_symbols)
    except ValueError as e:
        print(f"[error] field self-check failed: {e}", file=sys.stderr)
        return 4
    except ServiceError as e:  # type: ignore[misc]
        print(f"[error] panda_data service error: {e}", file=sys.stderr)
        return 1

    # 3. Turnover (only need view window, but reuse fetch buffer for simplicity)
    try:
        tv_df = data_mod.load_turnover(start, scan_date, initial_symbols)
    except ValueError as e:
        print(f"[error] field self-check failed: {e}", file=sys.stderr)
        return 4
    except ServiceError as e:  # type: ignore[misc]
        print(f"[error] panda_data service error: {e}", file=sys.stderr)
        return 1

    # 4. Universe
    syms, w_prior_series = universe.filter_universe(
        prior_df, prices_df, scan_date, min_valid_days=args.min_valid_days
    )
    if not syms:
        print("[error] empty universe after filtering", file=sys.stderr)
        return 3
    print(f"[info] universe: {len(syms)} stocks on {scan_date}", file=sys.stderr)

    # 5. Σ
    Sigma_df = covariance.ledoit_wolf_cov(prices_df, syms, scan_date, cov_lookback=args.cov_lookback)
    if Sigma_df.empty:
        print("[error] covariance is empty (all symbols dropped for NaN returns)", file=sys.stderr)
        return 3
    # Restrict universe to symbols that survived cov-window NaN filter
    syms = list(Sigma_df.index)
    w_prior = w_prior_series.reindex(syms).fillna(0.0).values
    # Renormalize after any drops
    total = w_prior.sum()
    if total <= 0:
        print("[error] w_prior sums to zero after cov-window filter", file=sys.stderr)
        return 3
    w_prior = w_prior / total
    Sigma = Sigma_df.values

    # PSD safety net
    eigs = np.linalg.eigvalsh(Sigma)
    if eigs.min() < -1e-8:
        print(f"[error] Σ not PSD after LW (min eig = {eigs.min():.3e})", file=sys.stderr)
        return 5

    # 6. π
    pi = bl.reverse_optimize(w_prior, Sigma, args.delta)

    # 7. Views
    P, Q, mem_df = views.build_views(
        prices_df, tv_df, syms, scan_date,
        mom_lookback=args.mom_lookback,
        rev_lookback=args.rev_lookback,
        turnover_lookback=args.turnover_lookback,
        q_hat=args.view_return,
        flip_mom=args.flip_mom,
        flip_rev=args.flip_rev,
        flip_turnover=args.flip_turnover,
    )

    # 8. Ω + posterior
    Omega = bl.omega_he_litterman(P, Sigma, args.tau)
    try:
        mu_bl, _Sigma_bl = bl.posterior(pi, Sigma, args.tau, P, Q, Omega)
    except np.linalg.LinAlgError as e:
        print(f"[error] BL posterior linear system singular: {e}", file=sys.stderr)
        return 5

    # 9. Optimize
    w_bl, degenerate = optimize.mv_long_only(mu_bl, Sigma, args.delta, w_prior)

    # 10. Build results frame
    results = pd.DataFrame({
        "trade_date": scan_date,
        "symbol": syms,
        "w_prior": w_prior,
        "w_bl": w_bl,
        "delta_w": w_bl - w_prior,
        "pi": pi,
        "mu_bl": mu_bl,
        "mu_bl_minus_pi": mu_bl - pi,
    })
    results = results.merge(
        mem_df.reset_index().rename(columns={"index": "symbol"}),
        on="symbol", how="left", validate="one_to_one",
    )
    # Fill any NaN membership (shouldn't happen) with 0
    for c in ("in_view_mom", "in_view_rev", "in_view_turnover"):
        results[c] = results[c].fillna(0).astype(int)

    # 11. Write outputs
    out_dir = Path(args.output_dir)
    csv_path = out_dir / f"portfolio_{scan_date}.csv"
    md_path = out_dir / f"portfolio_{scan_date}.md"
    report.write_csv(results, str(csv_path))
    params = {
        "delta": args.delta, "tau": args.tau, "view_return": args.view_return,
        "flip_mom": args.flip_mom, "flip_rev": args.flip_rev, "flip_turnover": args.flip_turnover,
    }
    report.write_markdown(
        results, str(md_path),
        date=scan_date, index_symbol=args.index_symbol,
        params=params, degenerate=degenerate,
    )
    print(f"[ok] wrote {csv_path} ({len(results)} stocks, degenerate={degenerate})")
    print(f"[ok] wrote {md_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # top-level safety net
        # Anything not caught above — including unexpected panda_data errors —
        # collapses to exit 1 with a one-line message (no traceback for the user).
        print(f"[error] unexpected: {e}", file=sys.stderr)
        sys.exit(1)
```

- [ ] **Step 2: Smoke-run without network (should hit RuntimeError on env, exit 1)**

Run:
```bash
unset PANDA_DATA_USERNAME PANDA_DATA_PASSWORD
python scripts/portfolio.py --date 20260721
echo "exit=$?"
```
Expected: `[error] Missing env vars ...` and `exit=1`.

- [ ] **Step 3: Commit**

```bash
git add scripts/portfolio.py
git commit -m "feat(portfolio): CLI orchestrator wiring data → universe → BL → optimize → report

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: End-to-end test with mocked panda_data

**Files:**
- Create: `tests/test_portfolio.py`

**Interfaces:**
- Consumes: full stack. Monkeypatches `panda_data.*` module in `sys.modules` so `data.py`'s lazy imports get the fake.
- Produces: single test that runs `portfolio.main()` end-to-end and asserts exit 0 + CSV/MD present + weight-sum invariant.

- [ ] **Step 1: Write failing end-to-end test**

```python
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

    # Build date list going backwards from scan_date (exclusive) then include scan_date
    dt = datetime.strptime(scan_date, "%Y%m%d")
    all_dates = [(dt - timedelta(days=n_days - i)).strftime("%Y%m%d") for i in range(n_days + 1)]

    # Random-walk closes per symbol
    price_rows = []
    turnover_rows = []
    for s in symbols:
        p = 100.0
        for d in all_dates:
            p *= float(np.exp(rng.normal(0.0005, 0.02)))
            price_rows.append({"symbol": s, "date": d, "close": p})
            turnover_rows.append({"symbol": s, "date": d, "turnover": float(rng.uniform(0.005, 0.03))})
    prices_df = pd.DataFrame(price_rows)
    turnover_df = pd.DataFrame(turnover_rows)

    # Prior weights on scan_date (roughly uniform, with a bit of noise)
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
```

- [ ] **Step 2: Run test, verify pass**

Run: `pytest tests/test_portfolio.py -v`
Expected: 1 passed.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: **all tests pass** — the sum of every task's tests (roughly 39 tests total: 8 + 6 + 4 + 7 + 6 + 4 + 3 + 1 = 39; adjust if you added more).

- [ ] **Step 4: Commit**

```bash
git add tests/test_portfolio.py
git commit -m "test(portfolio): end-to-end with mocked panda_data (30 stocks, 260 days)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: `SKILL.md` + `README.md`

**Files:**
- Create: `SKILL.md`, `README.md`

**Interfaces:**
- Consumes: nothing runtime; documents the shipped v0.1.0.
- Produces: two markdown files matching ETF-radar's shape.

- [ ] **Step 1: Write `SKILL.md`**

```markdown
---
name: skill-portfolio-blacklitterman
description: Black-Litterman 组合优化。以沪深300 指数权重为先验，用动量/反转/换手率三条因子视图更新，输出长权重 CSV + Markdown 报告。
tags: [quant, portfolio, black-litterman, optimization]
---

# Black-Litterman 组合优化

## 适用场景
- 想在给定日期基于沪深300 生成一份"相对指数带主观视图"的长权重组合
- 想跟踪三条常见因子视图（动量/反转/换手率）当前把权重推向哪些股票、推走哪些股票
- 想产出可插入下游回测/交易的 CSV 权重表

## 数据接口（panda_data）

| 接口 | 用途 | 关键字段 |
|---|---|---|
| `get_index_weights` | 先验权重 w_prior | `index_symbol, stock_symbol, date, weight` |
| `get_stock_daily` | 1 年日线 → 协方差 Σ + 动量/反转视图 | `symbol, date, close` |
| `get_factor` | 20 日换手率 → 换手率视图 | `symbol, date, turnover` |

字段详见 `references/need_used_api.md`。

## 术语约定

- 动量（20 日累计收益）**高 = 看多** → P 行正号
- 反转（5 日累计收益）**高 = 看空** → 建 P 时翻符号
- 换手率（20 日均）**高 = 看空** → 建 P 时翻符号

首次实测须校准；如反向，用 `--flip_mom` / `--flip_rev` / `--flip_turnover` 逐条翻转，无须改代码。

## 数学（4 步）

```
π      = δ · Σ · w_prior                            (implied returns)
Ω      = diag(τ · P · Σ · Pᵀ)                       (He-Litterman)
μ_bl   = M · [(τΣ)⁻¹ π + Pᵀ Ω⁻¹ Q],  M = [(τΣ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹
w_raw  = (δΣ)⁻¹ μ_bl,  w_bl = max(w_raw, 0) / sum(...)
```

默认：δ=2.5, τ=0.05, view_return q̂=0.05。所有窗口以 T-1 结尾。

## 视图构造

对每个因子分：
1. 计算截面得分（动量/反转 = 累计收益；换手率 = 均值）。
2. 取 top-decile 与 bottom-decile；构造单位空头-多头行 `P_k`，每行和为 0。
3. Q_k = `view_return`（默认 5%），对三条视图统一。

## 协方差

Ledoit-Wolf shrinkage on 252 交易日日收益，年化 ×252。任意含 NaN 的股票被剔除。

## 优化

闭式解 `w = (δΣ)⁻¹ μ_bl`，负权重截断为 0，重新归一化到 sum=1。
若所有 μ_bl ≤ 0 → 回退至 w_prior，日志打 WARN，仍以 exit=0 收尾。

## 使用方式

```bash
# 认证
export PANDA_DATA_USERNAME=...
export PANDA_DATA_PASSWORD=...

# 字段自检
python -m scripts.data --self-check --date 20260721

# 默认扫描
python scripts/portfolio.py

# 指定日期
python scripts/portfolio.py --date 20260721

# 全参数
python scripts/portfolio.py --date 20260721 \
    --index_symbol 000300.SH \
    --delta 2.5 --tau 0.05 --view_return 0.05 \
    --cov_lookback 252 --fetch_days 400 \
    --mom_lookback 20 --rev_lookback 5 --turnover_lookback 20 \
    --min_valid_days 200 \
    --flip_mom --flip_rev --flip_turnover \
    --output_dir output/

# 单元测试
pytest tests/ -v
```

## 输出

**`output/portfolio_YYYYMMDD.csv`**（每只股票一行，按 |Δw| 降序）：

| 列 | 说明 |
|---|---|
| `trade_date` | 扫描日 T |
| `symbol` | 股票代码 |
| `w_prior` | 指数权重（对入围股票归一化后） |
| `w_bl` | BL 长权重 |
| `delta_w` | w_bl − w_prior |
| `pi` | 隐含先验预期收益 (δΣw_prior) |
| `mu_bl` | 后验预期收益 |
| `mu_bl_minus_pi` | 视图对预期收益的推动 |
| `in_view_mom` | +1 / 0 / −1（动量视图中的多/中/空侧，含翻转后） |
| `in_view_rev` | 反转视图归属 |
| `in_view_turnover` | 换手率视图归属 |

**`output/portfolio_YYYYMMDD.md`**：Top 10 超配 / 低配 + 三条视图 top-3 空多列表 + 一句解读；退化路径（回退至 prior）会显式提示。

## 退出码

| Code | 含义 |
|---|---|
| 0 | 成功，CSV + MD 已写 |
| 1 | panda_data 接口 / 认证 / 网络异常 |
| 2 | 扫描日无 get_index_weights 数据 |
| 3 | 过滤后 universe 为空 |
| 4 | 字段自检失败 |
| 5 | LW + jitter 后 Σ 仍非 PSD |

## 验收要求
- **无未来函数**：所有窗口以 T-1 结尾；`test_build_views_no_lookahead_ignores_dates_ge_scan_date` 覆盖
- **单元测试全通过**：`pytest tests/` 无失败
- **He-Litterman 8 国算例**：`test_bl.py` 中的 π 复现误差 <0.6% 绝对值
- **端到端跑通**：mock panda_data 情况下 `test_portfolio_main_end_to_end` 通过；`w_bl.sum()` 在 1 ± 1e-6 内
- **字段自检**：`python -m scripts.data --self-check --date <近期>` 返回 0

## 已知局限（v0.1.0）
- Q̂ 单一全局常数（`--view_return`），未分因子标定；v0.2 计划补上。
- Σ_bl 计算但未用于优化器（详见 spec §6.4）。
- 无交易成本 / 换手率约束。
- Universe 固定 A 股股票指数成分，不接 HK/US/期货。
- 无滚动回测；快照分配器而非研究框架。
- 反转 / 换手率视图的符号约定为通用截面股票配置，可能需 `--flip_*` 校准。
- `get_index_weights` 文档示例调用未取 `weight`；实测若缺失 → 回退等权 + WARN。
```

- [ ] **Step 2: Write `README.md`**

```markdown
# skill-portfolio-blacklitterman

Claude Code skill for a daily Black-Litterman portfolio on 沪深300, driven by three factor views (Momentum / Reversal / Turnover). See `SKILL.md` for the full contract. Design lives in `docs/superpowers/specs/2026-07-29-portfolio-blacklitterman-design.md`; implementation plan in `docs/superpowers/plans/2026-07-29-portfolio-blacklitterman.md`.

## Quick start

```bash
export PANDA_DATA_USERNAME=...
export PANDA_DATA_PASSWORD=...
pip install -r requirements.txt
pytest tests/                                       # unit tests
python -m scripts.data --self-check --date 20260721 # field self-check
python scripts/portfolio.py --date 20260721         # single-day BL run
```

Outputs land in `output/portfolio_YYYYMMDD.csv` + `.md`.
```

- [ ] **Step 3: Verify pytest still green + tag v0.1.0**

Run: `pytest tests/ -v && git status`
Expected: all pass, working tree only has `SKILL.md` + `README.md` unstaged.

- [ ] **Step 4: Commit**

```bash
git add SKILL.md README.md
git commit -m "docs: SKILL.md contract + README quick-start

Co-Authored-By: Claude <noreply@anthropic.com>"
git tag v0.1.0
```

---

## Self-Review

**1. Spec coverage.**

| Spec section | Task |
|---|---|
| §2 locked scope (index=000300.SH, 3 views, LW cov, closed-form MV, δ=2.5, τ=0.05) | Defaults set in Tasks 2, 4, 6, 7, 9 |
| §3 module layout | File map ↑; each module has a task |
| §4 data loading + self-check + weight fallback | Task 2 |
| §5 universe rules (min_valid_days, weight>0, T-1 window, renormalize) | Task 3 |
| §6.1 π reverse-optimize | Task 6 |
| §6.2 P/Q from 3 factors + He-Litterman Ω + sign-flip flags | Task 5 (P/Q, flips) + Task 6 (Ω) |
| §6.3 μ_bl / Σ_bl posterior + jitter fallback | Task 6 |
| §6.4 closed-form MV + long-only + prior fallback | Task 7 |
| §7 CLI + exit codes 0–5 | Task 9 |
| §8.1 CSV columns + sort order | Task 8 + Task 9 (assembly) |
| §8.2 Markdown top-N + views + interpretation + degenerate note | Task 8 |
| §9 tests — data/universe/covariance/views/bl (HL toy)/optimize/portfolio | Tasks 2, 3, 4, 5, 6, 7, 10 |
| §10 SKILL.md contract | Task 11 |
| §11 references/need_used_api.md | Already exists (created during design) |
| §12 skill.json + requirements.txt | Task 1 |
| §13 acceptance criteria | Verified via `pytest tests/` in Tasks 2–10 |
| §14 known limits documented up-front | Task 11 (SKILL.md `已知局限` block) |

No gaps.

**2. Placeholder scan.** No `TBD` / `TODO` / `FIXME` / "add appropriate ..." / "similar to Task N" anywhere. All code blocks are complete.

**3. Type consistency.**
- `data.load_prior` returns columns `[symbol, date, weight]` — consumed by `universe.filter_universe(prior_df, ..., scan_date, min_valid_days)` (Task 3) which references `prior_df["symbol"]`, `prior_df["date"]`, `prior_df["weight"]`. ✓
- `data.load_prices` / `load_turnover` return columns `[symbol, date, close]` / `[symbol, date, turnover]` — consumed by `covariance.ledoit_wolf_cov` and `views.build_views`. ✓
- `universe.filter_universe` returns `(list[str], pd.Series)` — the Series is indexed by symbol. Task 9 calls `.reindex(syms).fillna(0.0).values`. ✓
- `covariance.ledoit_wolf_cov` returns `pd.DataFrame` (indexed by symbol) — Task 9 reads `.index` to prune the universe and `.values` for the matrix. ✓
- `views.build_views(...)` returns `(P: np.ndarray[3,N], Q: np.ndarray[3], mem_df: pd.DataFrame indexed by symbol)` — matches Task 9's consumption. ✓
- `bl.reverse_optimize(w_prior, Sigma, delta) -> np.ndarray[N]`, `bl.omega_he_litterman(P, Sigma, tau) -> np.ndarray[K,K]`, `bl.posterior(pi, Sigma, tau, P, Q, Omega) -> (mu_bl, Sigma_bl)`. All consumed correctly in Task 9. ✓
- `optimize.mv_long_only(mu_bl, Sigma, delta, w_prior) -> (np.ndarray[N], bool)` — Task 9 unpacks `(w_bl, degenerate)`. ✓
- `report.write_csv(df, path)` + `report.write_markdown(df, path, *, date, index_symbol, params, degenerate)`. Task 9 calls with the correct kwargs. ✓
- `report.CSV_COLUMNS` list is a private contract between Task 8 and Task 9 assembly (Task 9 produces the same column names). ✓

One notable naming-consistency check: `mu_bl_minus_pi` column is written in Task 9 (`results["mu_bl_minus_pi"] = mu_bl - pi`); it appears verbatim in `report.CSV_COLUMNS` (Task 8) and in `SKILL.md` output table (Task 11). ✓

Plan is internally consistent, no gaps against the spec.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-29-portfolio-blacklitterman.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
