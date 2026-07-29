"""Closed-form mean-variance optimizer with long-only projection.

    w_raw  = (δΣ)⁻¹ μ_bl
    w_clip = max(w_raw, 0)
    w_bl   = w_clip / sum(w_clip)

If sum(w_clip) == 0 (pathological — all μ_bl ≤ 0), fall back to w_prior
and emit a one-line WARN on stderr (design spec §6.4).
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
    """Closed-form MV weights with long-only clip.

    Returns ``(w_bl, degenerate_flag)``. ``degenerate_flag`` is True iff
    the long-only projection wiped every weight and we fell back to
    ``w_prior``.
    """
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
