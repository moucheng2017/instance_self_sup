"""Tests for tools/vmf_math.py (vMF normalizers, Banerjee kappa MLE, KL, tables).

Validation strategy (scipy-free; scipy cross-checks run when available):
  - d=3 closed forms: C_3(kappa) = kappa / (4*pi*sinh(kappa)),
    A_3(kappa) = coth(kappa) - 1/kappa. Exact for ALL kappa.
  - kappa -> 0 limits: C_d(0) = uniform density on S^{d-1}, A_d(0) = 0.
  - Bessel recurrence: I_{v-1} - I_{v+1} = (2v/kappa) I_v.
  - Large-kappa asymptotic for I_0.
  - Identity d/dkappa log C_d = -A_d (finite differences).
  - Banerjee MLE approximately inverts A_d.
  - KL: zero at identity, nonnegative, monotone in angle, matches Monte Carlo
    estimate at d=3 (exact inverse-CDF sampler).
  - Expected-likelihood kernel matches Monte Carlo at d=3.
  - Table interpolation matches exact evaluation.

Run locally:  PYTHONPATH="$PWD" conda run -n ssl_local pytest -q tests/test_vmf_math.py
"""

import math

import numpy as np

try:  # optional: only used to mark skips nicely when running under pytest
    import pytest
except ImportError:  # pragma: no cover - plain-runner fallback
    pytest = None

try:  # optional cross-check dependency
    from scipy import special as _sps
except ImportError:  # pragma: no cover
    _sps = None

from tools.vmf_math import (
    VmfTable,
    a_d,
    banerjee_kappa,
    log_c_d,
    log_expected_likelihood,
    log_iv,
    vmf_kl,
)


# ---------------------------------------------------------------------------
# exact references
# ---------------------------------------------------------------------------

def _log_sinh(kappa):
    """Stable log(sinh(kappa)) for kappa > 0."""
    kappa = np.asarray(kappa, dtype=np.float64)
    return kappa + np.log1p(-np.exp(-2.0 * kappa)) - math.log(2.0)


def _log_c3_exact(kappa):
    """log C_3(kappa) = log kappa - log(4 pi sinh kappa)."""
    kappa = np.asarray(kappa, dtype=np.float64)
    return np.log(kappa) - math.log(4.0 * math.pi) - _log_sinh(kappa)


def _a3_exact(kappa):
    """A_3(kappa) = coth(kappa) - 1/kappa."""
    kappa = np.asarray(kappa, dtype=np.float64)
    return 1.0 / np.tanh(kappa) - 1.0 / kappa


def _log_c_d_at_zero(d):
    """Uniform density on S^{d-1}: C_d(0) = Gamma(d/2) / (2 pi^{d/2})."""
    return math.lgamma(d / 2.0) - math.log(2.0) - (d / 2.0) * math.log(math.pi)


def _sample_vmf3(mu, kappa, n, rng):
    """Exact vMF sampler on S^2 via the closed-form inverse CDF of t = cos(angle)."""
    mu = np.asarray(mu, dtype=np.float64)
    u = rng.uniform(size=n)
    t = 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * kappa)) / kappa
    # random tangent directions orthogonal to mu
    raw = rng.normal(size=(n, 3))
    raw -= np.outer(raw @ mu, mu)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    z = t[:, None] * mu[None, :] + np.sqrt(np.clip(1.0 - t**2, 0.0, None))[:, None] * raw
    return z / np.linalg.norm(z, axis=1, keepdims=True)


KAPPAS = np.array([1e-3, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5])


# ---------------------------------------------------------------------------
# log_iv / log_c_d / a_d
# ---------------------------------------------------------------------------

def test_log_c3_matches_closed_form_over_huge_kappa_range():
    np.testing.assert_allclose(log_c_d(KAPPAS, 3), _log_c3_exact(KAPPAS), rtol=1e-10)


def test_a3_matches_closed_form_over_huge_kappa_range():
    np.testing.assert_allclose(a_d(KAPPAS, 3), _a3_exact(KAPPAS), rtol=1e-9, atol=1e-12)


def test_log_c_d_kappa_zero_is_uniform_density():
    for d in (2, 3, 8, 64, 256):
        assert math.isclose(float(log_c_d(0.0, d)), _log_c_d_at_zero(d), rel_tol=1e-12)
        # continuity at 0
        assert math.isclose(float(log_c_d(1e-8, d)), _log_c_d_at_zero(d), rel_tol=1e-9)


def test_a_d_kappa_zero_and_range_and_monotone():
    grid = np.linspace(0.0, 5000.0, 200)
    for d in (3, 64, 256):
        vals = a_d(grid, d)
        assert vals[0] == 0.0
        assert np.all(vals >= 0.0) and np.all(vals < 1.0)
        assert np.all(np.diff(vals) > 0.0)


def test_log_iv_bessel_recurrence():
    # I_{v-1}(k) - I_{v+1}(k) = (2 v / k) I_v(k)
    for v in (0.5, 1.0, 63.5, 127.0):
        for kappa in (1.0, 10.0, 100.0, 1e3, 1e4):
            base = float(log_iv(v, kappa))
            lo = math.exp(float(log_iv(v - 1.0, kappa)) - base)
            hi = math.exp(float(log_iv(v + 1.0, kappa)) - base)
            assert math.isclose(lo - hi, 2.0 * v / kappa, rel_tol=1e-8)


def test_log_iv_large_kappa_asymptotic_i0():
    # I_0(k) ~ e^k / sqrt(2 pi k) * (1 + 1/(8k))
    kappa = 1e4
    expected = kappa - 0.5 * math.log(2.0 * math.pi * kappa) + math.log1p(1.0 / (8.0 * kappa))
    assert math.isclose(float(log_iv(0.0, kappa)), expected, rel_tol=1e-7)


def test_dlogc_dkappa_equals_minus_a_d():
    # The identity used by the KL formula and all gradients.
    h = 1e-3
    for d in (3, 64, 256):
        for kappa in (0.5, 5.0, 37.3, 412.0, 2.5e3):
            fd = (float(log_c_d(kappa + h, d)) - float(log_c_d(kappa - h, d))) / (2.0 * h)
            assert math.isclose(fd, -float(a_d(kappa, d)), rel_tol=1e-5, abs_tol=1e-9)


def test_log_iv_vectorized_with_zero_entries():
    kappa = np.array([0.0, 1.0, 10.0])
    out0 = log_iv(0.0, kappa)
    assert out0.shape == (3,)
    assert out0[0] == 0.0  # I_0(0) = 1
    out1 = log_iv(2.0, kappa)
    assert np.isneginf(out1[0])  # I_v(0) = 0 for v > 0


def test_scipy_cross_check_log_iv_and_a_d():
    if _sps is None:
        if pytest is not None:
            pytest.skip("scipy not available")
        return
    rng = np.random.default_rng(0)
    kappas = 10.0 ** rng.uniform(-3, 4.5, size=64)
    for d in (2, 3, 64, 256):
        v = d / 2.0 - 1.0
        ref = np.log(_sps.ive(v, kappas)) + kappas
        np.testing.assert_allclose(log_iv(v, kappas), ref, rtol=1e-9)
        ref_a = _sps.ive(v + 1.0, kappas) / _sps.ive(v, kappas)
        np.testing.assert_allclose(a_d(kappas, d), ref_a, rtol=1e-8)


# ---------------------------------------------------------------------------
# banerjee_kappa
# ---------------------------------------------------------------------------

def test_banerjee_kappa_approximately_inverts_a_d():
    rbars = np.linspace(0.05, 0.95, 19)
    for d in (8, 64, 256):
        kappas = banerjee_kappa(rbars, d)
        recovered = a_d(kappas, d)
        np.testing.assert_allclose(recovered, rbars, rtol=0.02, atol=0.01)


def test_banerjee_kappa_edges_and_clamping():
    d = 256
    assert float(banerjee_kappa(0.0, d)) == 0.0
    assert float(banerjee_kappa(-0.5, d)) == 0.0  # clamped from below
    hi = float(banerjee_kappa(1.0, d))  # clamped below 1: finite
    assert np.isfinite(hi) and hi > 1e4
    assert float(banerjee_kappa(0.999999999, d)) == hi  # same clamp point


def test_banerjee_kappa_vectorized_monotone():
    d = 64
    rbars = np.linspace(0.0, 0.99, 100)
    kappas = banerjee_kappa(rbars, d)
    assert kappas.shape == (100,)
    assert np.all(np.diff(kappas) > 0.0)


# ---------------------------------------------------------------------------
# vmf_kl
# ---------------------------------------------------------------------------

def _random_unit(rng, n, d):
    z = rng.normal(size=(n, d))
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def test_vmf_kl_zero_on_identical_distributions():
    rng = np.random.default_rng(1)
    mu = _random_unit(rng, 8, 256)
    kappa = 10.0 ** rng.uniform(0, 3, size=8)
    np.testing.assert_allclose(vmf_kl(mu, kappa, mu, kappa), 0.0, atol=1e-9)


def test_vmf_kl_nonnegative_on_random_pairs():
    rng = np.random.default_rng(2)
    n = 256
    mu1, mu2 = _random_unit(rng, n, 256), _random_unit(rng, n, 256)
    k1 = 10.0 ** rng.uniform(-1, 3.3, size=n)
    k2 = 10.0 ** rng.uniform(-1, 3.3, size=n)
    kl = vmf_kl(mu1, k1, mu2, k2)
    assert kl.shape == (n,)
    assert np.all(kl >= -1e-8)


def test_vmf_kl_monotone_in_angle():
    d = 64
    mu1 = np.zeros(d)
    mu1[0] = 1.0
    angles = np.linspace(0.0, math.pi, 32)
    mu2 = np.zeros((32, d))
    mu2[:, 0] = np.cos(angles)
    mu2[:, 1] = np.sin(angles)
    kl = vmf_kl(np.broadcast_to(mu1, (32, d)), np.full(32, 50.0), mu2, np.full(32, 20.0))
    assert np.all(np.diff(kl) > 0.0)


def test_vmf_kl_matches_monte_carlo_d3():
    rng = np.random.default_rng(3)
    mu1 = np.array([0.0, 0.0, 1.0])
    mu2 = np.array([math.sin(0.7), 0.0, math.cos(0.7)])
    k1, k2 = 18.0, 5.0
    z = _sample_vmf3(mu1, k1, 400_000, rng)
    log_f1 = _log_c3_exact(k1) + k1 * (z @ mu1)
    log_f2 = _log_c3_exact(k2) + k2 * (z @ mu2)
    mc = float(np.mean(log_f1 - log_f2))
    se = float(np.std(log_f1 - log_f2) / math.sqrt(len(z)))
    analytic = float(vmf_kl(mu1, k1, mu2, k2))
    assert abs(analytic - mc) < 5.0 * se + 1e-4


# ---------------------------------------------------------------------------
# expected-likelihood kernel
# ---------------------------------------------------------------------------

def test_log_expected_likelihood_symmetric():
    rng = np.random.default_rng(4)
    mu1, mu2 = _random_unit(rng, 16, 256), _random_unit(rng, 16, 256)
    k1 = 10.0 ** rng.uniform(0, 3, size=16)
    k2 = 10.0 ** rng.uniform(0, 3, size=16)
    np.testing.assert_allclose(
        log_expected_likelihood(mu1, k1, mu2, k2),
        log_expected_likelihood(mu2, k2, mu1, k1),
        rtol=1e-10,
    )


def test_log_expected_likelihood_matches_monte_carlo_d3():
    rng = np.random.default_rng(5)
    mu1 = np.array([0.0, 0.0, 1.0])
    mu2 = np.array([math.sin(1.1), 0.0, math.cos(1.1)])
    k1, k2 = 12.0, 7.0
    z = _sample_vmf3(mu1, k1, 400_000, rng)
    f2 = np.exp(_log_c3_exact(k2) + k2 * (z @ mu2))
    mc = float(np.mean(f2))
    se = float(np.std(f2) / math.sqrt(len(z)))
    analytic = math.exp(float(log_expected_likelihood(mu1, k1, mu2, k2)))
    assert abs(analytic - mc) < 5.0 * se + 1e-7


# ---------------------------------------------------------------------------
# VmfTable
# ---------------------------------------------------------------------------

def test_vmf_table_matches_exact_on_random_kappas():
    d = 256
    table = VmfTable(d, kappa_max=2.0e4)
    rng = np.random.default_rng(6)
    kappas = rng.uniform(0.0, 2.0e4, size=512)
    np.testing.assert_allclose(table.log_c(kappas), log_c_d(kappas, d), rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(table.a(kappas), a_d(kappas, d), rtol=1e-5, atol=1e-6)


def test_vmf_table_dense_near_zero():
    d = 64
    table = VmfTable(d, kappa_max=2.0e4)
    kappas = np.linspace(0.0, 5.0, 64)
    np.testing.assert_allclose(table.a(kappas), a_d(kappas, d), atol=1e-6)


def test_vmf_table_clamps_out_of_range():
    d = 64
    table = VmfTable(d, kappa_max=1.0e3)
    assert float(table.log_c(2.0e3)) == float(table.log_c(1.0e3))
    assert float(table.a(-1.0)) == float(table.a(0.0))


def test_vmf_table_endpoints_exact():
    d = 256
    table = VmfTable(d, kappa_max=2.0e4)
    assert math.isclose(float(table.log_c(0.0)), _log_c_d_at_zero(d), rel_tol=1e-12)
    assert float(table.a(0.0)) == 0.0
