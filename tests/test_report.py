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
