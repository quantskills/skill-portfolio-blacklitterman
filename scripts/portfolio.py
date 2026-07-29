"""Black-Litterman portfolio CLI orchestrator.

Usage:
    python scripts/portfolio.py [--date YYYYMMDD] [--index_symbol ...] [...]

Exit codes (design §7):
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Black-Litterman 组合优化 (沪深300)")
    p.add_argument("--date", default=None, help="扫描日 YYYYMMDD；默认取今天")
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
    return p.parse_args(argv)


def _shift_days(date_yyyymmdd: str, days: int) -> str:
    dt = datetime.strptime(date_yyyymmdd, "%Y%m%d")
    return (dt - timedelta(days=days)).strftime("%Y%m%d")


def _resolve_service_error_cls() -> tuple:
    try:
        from panda_data.exceptions import ServiceError as _SE
        return (_SE,)
    except ImportError:
        return ()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
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

    # 4. Universe filter
    syms, w_prior_series = universe.filter_universe(
        prior_df, prices_df, scan_date, min_valid_days=args.min_valid_days
    )
    if not syms:
        print("[error] empty universe after filtering", file=sys.stderr)
        return 3
    print(f"[info] universe: {len(syms)} stocks on {scan_date}", file=sys.stderr)

    # 5. Σ (Ledoit-Wolf, annualized)
    Sigma_df = covariance.ledoit_wolf_cov(
        prices_df, syms, scan_date, cov_lookback=args.cov_lookback
    )
    if Sigma_df.empty:
        print("[error] covariance is empty (all symbols dropped for NaN returns)", file=sys.stderr)
        return 3

    # Restrict universe to symbols that survived cov-window NaN filter
    syms = list(Sigma_df.index)
    w_prior = w_prior_series.reindex(syms).fillna(0.0).values
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

    # 6. Reverse-optimize prior → π
    pi = bl.reverse_optimize(w_prior, Sigma, args.delta)

    # 7. Views (P, Q, membership)
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

    # 9. MV + long-only
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
        print(f"[error] unexpected: {e}", file=sys.stderr)
        sys.exit(1)
