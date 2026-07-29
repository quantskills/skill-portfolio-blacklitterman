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


def _sort_by_abs_delta_desc(df: pd.DataFrame) -> pd.DataFrame:
    order = df["delta_w"].abs().to_numpy().argsort()[::-1]
    return df.iloc[order].reset_index(drop=True)


def write_csv(results: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = results.copy().reindex(columns=CSV_COLUMNS)
    df = _sort_by_abs_delta_desc(df)
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
    df = _sort_by_abs_delta_desc(results.copy().reindex(columns=CSV_COLUMNS))
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
        lines.append(
            f"- **{name}** — long: {', '.join(long_syms) or '—'}; "
            f"short: {', '.join(short_syms) or '—'}"
        )

    # One-line interpretation
    turnover_vs_prior = (df["delta_w"].abs().sum()) / 2.0  # L1 / 2
    lines.append(
        f"\n## Interpretation\n"
        f"Net turnover vs prior ≈ {_fmt_pct(turnover_vs_prior)}. "
        f"Largest overweight: {over.iloc[0]['symbol'] if len(over) else 'none'}; "
        f"largest underweight: {under.iloc[0]['symbol'] if len(under) else 'none'}.\n"
    )

    Path(path).write_text("\n".join(lines))
