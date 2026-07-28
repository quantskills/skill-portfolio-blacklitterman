# skill-portfolio-blacklitterman — Design

**Date:** 2026-07-29
**Author:** Claude Code (with 刘相君)
**Status:** Approved, ready for planning

## 1. Purpose

A Claude Code skill that produces a **Black-Litterman long-only portfolio** for the 沪深300 universe on a given scan date. The skill takes the index as its prior (market equilibrium), applies three factor-based views (Momentum + Reversal + Turnover), and outputs per-stock BL weights plus a Markdown report.

The skill exercises exactly three `panda_data` interfaces:

| Interface | Role |
|---|---|
| `get_index_weights` | Prior weights `w_prior` (equilibrium anchor) |
| `get_stock_daily` | 1-year daily closes → covariance Σ and Momentum/Reversal views |
| `get_factor` | 20-day turnover → Turnover view |

Layout, CLI conventions, and error-code style mirror the sibling `skill-etf-flow-radar` for consistency.

## 2. Scope (locked during brainstorming)

| Choice | Value |
|---|---|
| Benchmark | 沪深300 (`000300.SH`), CLI-overridable via `--index_symbol` |
| Universe | 沪深300 constituents on scan-date T, filtered for data availability |
| Views | 3 relative decile-spread views — Momentum(20d), Reversal(5d), Turnover(20d) |
| Covariance | Ledoit-Wolf shrinkage on 252 trading days of daily returns, annualized |
| Optimizer | Closed-form MV `w = (δΣ)⁻¹μ_bl` + long-only clip + renormalize |
| Output | `output/portfolio_YYYYMMDD.csv` + `output/portfolio_YYYYMMDD.md` |
| Defaults | δ = 2.5, τ = 0.05, Ω = He-Litterman `diag(τ · PΣPᵀ)`, view_return q̂ = 5% |

## 3. Architecture

```
skill-portfolio-blacklitterman/
├── SKILL.md                          # Contract (top-level)
├── README.md                         # Quick start
├── skill.json                        # Metadata
├── requirements.txt                  # panda_data, pandas, numpy, scikit-learn
├── references/
│   └── need_used_api.md              # Extracted docs for the 3 APIs
├── scripts/
│   ├── __init__.py
│   ├── data.py                       # panda_data init, loaders, --self-check
│   ├── universe.py                   # 沪深300 constituents ∩ data-available
│   ├── views.py                      # (P, Q) from Mom + Rev + Turnover
│   ├── covariance.py                 # Ledoit-Wolf Σ, annualized
│   ├── bl.py                         # Reverse-optimize π, BL posterior (μ_bl, Σ_bl)
│   ├── optimize.py                   # Closed-form MV + long-only clip
│   ├── report.py                     # CSV + Markdown
│   └── portfolio.py                  # CLI orchestrator
└── tests/
    ├── conftest.py                   # Monkey-patch panda_data with canned frames
    ├── test_data.py
    ├── test_universe.py
    ├── test_covariance.py
    ├── test_views.py
    ├── test_bl.py                    # Reproduce He-Litterman 8-country toy
    ├── test_optimize.py
    └── test_portfolio.py             # End-to-end with mocked data
```

### Data flow

```
scan_date T
    ↓
data.load_prior(T)           → w_prior (N × 1)             ← get_index_weights
data.load_prices(T-400d..T)  → close matrix (D × N)        ← get_stock_daily
data.load_turnover(T-400d..T)→ turnover matrix (D × N)     ← get_factor
    ↓
universe.filter()            → keep symbols with weight on T AND ≥200 valid closes
    ↓
covariance.ledoit_wolf()     → Σ (N × N, PSD, annualized)
    ↓
bl.reverse_optimize(w_prior, Σ, δ) → π
views.build(prices, turnover)      → P (3 × N), Q (3 × 1)
Ω = diag(τ · P · Σ · Pᵀ)
bl.posterior(π, Σ, τ, P, Q, Ω)     → μ_bl, Σ_bl
    ↓
optimize.mv_long_only(μ_bl, Σ, δ)  → w_bl (long-only, sums to 1)
    ↓
report.write_csv + report.write_markdown
```

## 4. Data loading (`scripts/data.py`)

### 4.1 `init_panda_data()`

Reads `PANDA_DATA_USERNAME` / `PANDA_DATA_PASSWORD`, calls `panda_data.init(...)`. Raises `RuntimeError` if either env var is missing. Idempotent via module-level flag.

### 4.2 Loaders

- **`load_prior(date, index_symbol='000300.SH')`** — `get_index_weights(index_symbol=..., start_date=date, end_date=date, fields=None)`. Returns `pd.DataFrame[symbol, weight]`, where `symbol = stock_symbol` from the response. Raises `ValueError` on empty result (weekend/holiday/pre-launch date).

  **Caveat surfaced by API doc:** the example call passes `fields=["index_symbol", "stock_symbol", "date"]` (no `weight`), but the response schema *does* list `weight` as a float. Skill passes `fields=None` to request all columns and asserts `weight` is present. If `weight` is still missing on real data, fall back to **equal weights** across returned constituents and emit `[warn] weight column missing — falling back to equal weights`. This is a first-run calibration item.

- **`load_prices(start, end, symbols)`** — `get_stock_daily(symbol=symbols, start_date=start, end_date=end, fields=['symbol','date','close'])`. Returns long-form `pd.DataFrame[symbol, date, close]`.

- **`load_turnover(start, end, symbols)`** — `get_factor(symbol=symbols, start_date=start, end_date=end, factors=['turnover'], type='stock')`. Returns `pd.DataFrame[symbol, date, turnover]`.

### 4.3 Column self-check

`_assert_columns(df, expected, name)` raises `ValueError` with a diff when a returned dataframe is missing an expected column. Called after every loader.

`python -m scripts.data --self-check --date YYYYMMDD` — runs a 1-day / few-symbol query against each loader and prints the column set. Exit 0 on all-pass, 4 on any mismatch.

### 4.4 Fetch window

- **Cov window:** default 252 trading days ending T-1. Fetch buffer 400 natural days (mirrors ETF-radar's 40-day-for-20-trading-day pattern, scaled to 252 trading days).
- **View windows:** all end T-1 (strictly historical, no lookahead).

## 5. Universe (`scripts/universe.py`)

`filter_universe(prior_df, prices_df, scan_date, min_valid_days=200) -> (list[str], pd.Series)`:

1. Start from `prior_df[prior_df.date == scan_date].symbol` (~300 names).
2. Drop stocks with fewer than `min_valid_days` non-null `close` observations in the 1-year window ending T-1 (excludes recent IPOs, extended suspensions). Window ends T-1 to stay consistent with view/cov windows.
3. Drop stocks with `weight == 0` or NaN.
4. Return the surviving symbol list and the **re-normalized** weight series (sums to 1 over the surviving set).

Empty return → CLI exits with code 3.

## 6. Math

### 6.1 Reverse-optimize the prior (`bl.reverse_optimize`)

```
π = δ · Σ · w_prior
```

`π ∈ ℝᴺ` — the market-implied equilibrium excess returns.

### 6.2 Views (`views.build`)

For each factor score vector `s` (length N), factor scores computed with all windows ending **T-1** (no T-day data):

| Factor | Formula | Sign convention |
|---|---|---|
| Momentum | 20-day cumulative return | High = bullish (P row +) |
| Reversal | 5-day cumulative return | High = **bearish** (sign-flipped) |
| Turnover | 20-day mean turnover | High = **bearish** (sign-flipped, low-turnover premium) |

Build view row k:
1. Rank stocks by score; take top decile `L` and bottom decile `S`. |L| ≈ |S| ≈ N/10 ≈ 30.
2. `P_k[i] = +1/|L|` if i ∈ L, `−1/|S|` if i ∈ S, else 0. Each row sums to zero (long-short unit-notional).
3. `Q_k = q̂` where q̂ = `--view_return` (default 0.05 annualized). Applied to all three views. Rationale: BL output ordering is dominated by view direction and Ω/τ; the absolute magnitude of Q rescales `μ_bl` roughly linearly and is a single global knob.
4. For Reversal/Turnover, the sign flip happens **at P-construction time** (top decile of the raw factor becomes the *short* side). This keeps every Q_k positive and CSV-readable.

**View uncertainty (K × K diagonal):**

```
Ω = diag(τ · P · Σ · Pᵀ)      (He-Litterman 2002)
```

Each view's variance is proportional to its portfolio variance under Σ — noisier views auto-shrink. Zero manual tuning.

**Sign-convention escape hatches:** `--flip_mom`, `--flip_rev`, `--flip_turnover` invert individual P rows without code edits (mirrors ETF-radar's `SIGN_FLIP` caveat).

### 6.3 BL posterior (`bl.posterior`)

```
M    = [(τΣ)⁻¹ + Pᵀ · Ω⁻¹ · P]⁻¹
μ_bl = M · [(τΣ)⁻¹ · π + Pᵀ · Ω⁻¹ · Q]
Σ_bl = Σ + M
```

Implemented via `numpy.linalg.solve` (never `inv`) for numerical stability. If `τΣ` or `Ω` is singular, add diagonal jitter `1e-8 · trace/N` and retry once; if still singular, exit with code 5.

### 6.4 Optimize (`optimize.mv_long_only`)

```
w_raw  = (δΣ)⁻¹ · μ_bl        # unconstrained MV using PRIOR Σ (not Σ_bl)
w_clip = max(w_raw, 0)         # long-only projection
w_bl   = w_clip / sum(w_clip)  # renormalize to sum = 1
```

**Why Σ, not Σ_bl?** With τ = 0.05, `M = Σ_bl − Σ` is small; keeping Σ makes the optimizer's magnitude invariant to view strength — turnover-vs-prior comes purely from `μ_bl − π`, which is exactly the desired active-bet interpretation.

**Degenerate case:** if `sum(w_clip) == 0` (all `μ_bl` negative — pathological), fall back to `w_prior` with `[warn] all μ_bl negative — falling back to prior`. Exit 0.

## 7. CLI (`scripts/portfolio.py`)

```bash
# Auth
export PANDA_DATA_USERNAME=...
export PANDA_DATA_PASSWORD=...

# Field self-check
python -m scripts.data --self-check --date 20260721

# Default run (latest available prior date)
python scripts/portfolio.py

# Explicit date
python scripts/portfolio.py --date 20260721

# Full knob surface
python scripts/portfolio.py --date 20260721 \
    --index_symbol 000300.SH \
    --delta 2.5 --tau 0.05 --view_return 0.05 \
    --cov_lookback 252 --fetch_days 400 \
    --mom_lookback 20 --rev_lookback 5 --turnover_lookback 20 \
    --min_valid_days 200 \
    --flip_mom --flip_rev --flip_turnover \
    --output_dir output/

# Tests
pytest tests/ -v
```

### Exit codes

| Code | Meaning |
|---|---|
| 0 | OK — CSV + MD written |
| 1 | `panda_data` interface/network/auth failure |
| 2 | Scan date has no `get_index_weights` data (holiday / bad date) |
| 3 | Universe empty after filtering |
| 4 | Column self-check failure |
| 5 | Σ not PSD after LW + jitter retry (numerical safety net) |

## 8. Output

### 8.1 `output/portfolio_YYYYMMDD.csv`

One row per universe stock, sorted by `|delta_w|` desc:

| Column | Meaning |
|---|---|
| `trade_date` | Scan date T |
| `symbol` | Stock code (e.g. `600519.SH`) |
| `w_prior` | Index weight on T (renormalized over surviving universe) |
| `w_bl` | Posterior long-only weight |
| `delta_w` | `w_bl − w_prior` (active bet) |
| `pi` | Implied prior return (δΣw_prior) |
| `mu_bl` | Posterior expected return |
| `mu_bl_minus_pi` | View-driven return update |
| `in_view_mom` | +1 / 0 / −1 (top / neutral / bottom decile of Momentum) |
| `in_view_rev` | Same for Reversal (post-sign-flip) |
| `in_view_turnover` | Same for Turnover (post-sign-flip) |

### 8.2 `output/portfolio_YYYYMMDD.md`

- **Header** — date, index, N, δ, τ, view_return, sign-flip flags in effect
- **Top 10 overweights** table (`symbol, w_prior, w_bl, delta_w, active views`)
- **Top 10 underweights** (most-negative `delta_w`)
- **Views summary** — for each of the 3 views, top-3 and bottom-3 contributors
- **One-line interpretation** — e.g. "Momentum view concentrates weight in large-cap tech; reversal trims recent winners; net turnover-vs-prior ≈ X%."

## 9. Tests

Reuse ETF-radar's monkey-patch style — `tests/conftest.py` patches `panda_data.*` functions to return canned pandas frames. No network in test suite.

| Test file | Cases |
|---|---|
| `test_data.py` | Loaders return expected columns; `_assert_columns` raises on mismatch; missing-`weight` fallback emits WARN; missing env var raises `RuntimeError` |
| `test_universe.py` | Filters stocks with too few valid closes; re-normalizes to sum=1; empty universe returns `([], empty_series)` |
| `test_covariance.py` | LW output symmetric PSD, shape N×N, annualized (×252), stocks with any NaN in cov window are dropped |
| `test_views.py` | P shape (K×N); each row sums to zero; sign flips invert rows; Q vector length K; top/bottom deciles disjoint |
| `test_bl.py` | **Reproduce He-Litterman 1999 8-country toy** — 8 assets, 2 views, known π and μ_bl to 4 decimals |
| `test_optimize.py` | Long-only clip; sum-to-one; all-negative-μ fallback with WARN; monotonicity (raising μ_bl[i] raises w_bl[i]) |
| `test_portfolio.py` | End-to-end with mocked panda_data: exit 0, files exist, CSV has ~N rows, `w_bl.sum() == 1.0 ± 1e-9` |

`test_bl.py`'s reproduction of a canonical published BL example is the single most valuable test — it catches every math regression.

## 10. SKILL.md contract

Mirrors ETF-radar's SKILL.md structure:

1. **适用场景** — 3 bullets (BL relative-to-index bets, factor-view tracking, CSV+MD downstream feed)
2. **数据接口 (panda_data)** — table of the 3 APIs with used fields
3. **术语约定** — sign conventions for Rev/Turnover views, `--flip_*` escape hatch
4. **数学** — compact 4-step formula block from Section 6
5. **输入 → 输出 → CLI → 验收要求 → 已知局限**

## 11. `references/need_used_api.md`

Verbatim extract of the three API sections from `../panda_data_api_doc.md`:

- `get_index_weights` (source lines 2214–2296)
- `get_stock_daily` (source lines 436–508)
- `get_factor` (source lines 11387–11467)

Kept as one file so field references live in one place (matches ETF-radar convention).

## 12. Metadata

- **`skill.json`** — name `skill-portfolio-blacklitterman`, version `0.1.0`, tags `[quant, portfolio, black-litterman, optimization]`, entry `scripts/portfolio.py`
- **`requirements.txt`** — `panda-data`, `pandas>=2.0`, `numpy>=1.24`, `scikit-learn>=1.3`, `pytest` (dev)

## 13. Acceptance criteria

- **No lookahead** — all view/cov windows end T-1; unit test `test_views_no_lookahead` verifies view scores on synthetic frames.
- **All unit tests green** — `pytest tests/` with zero failures.
- **Column self-check passes** — `python -m scripts.data --self-check --date <recent trading day>` returns 0.
- **End-to-end runs** — at least one real date produces CSV + MD; `w_bl.sum() ∈ [1 − 1e-9, 1 + 1e-9]`.
- **He-Litterman reproduction** — `test_bl.py` matches published μ_bl to 4 decimals.
- **Doc-code consistency** — SKILL.md formulas match `scripts/bl.py` / `scripts/optimize.py`.

## 14. Known limits (v0.1.0)

- View return magnitude Q is a single global constant (`--view_return`); realistic per-factor calibration deferred to v0.2.
- `Σ_bl` is computed but not used in the optimizer (see §6.4 rationale).
- No transaction-cost / turnover penalty in the optimizer.
- Universe fixed to A-share equity index constituents; no HK/US/futures.
- No rolling backtest — snapshot allocator, not a research framework.
- Cov-window stocks may include index-exit / re-entry across the year; MVP uses whatever close data is present.
- Sign conventions for Reversal / Turnover views are the standard cross-sectional-equity convention but may need `--flip_*` on this specific market/period; first-run calibration item.
- `get_index_weights` response schema mismatch (example call omits `weight`, response table includes it); skill validates and falls back to equal weights with WARN if `weight` is absent.
