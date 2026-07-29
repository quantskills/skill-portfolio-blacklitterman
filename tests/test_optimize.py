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
