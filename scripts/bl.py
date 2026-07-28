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
