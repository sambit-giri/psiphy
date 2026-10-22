from math import factorial

import numpy as np
import pytest

from psiphy.forecasting import FisherMatrix, FisherMatrixDALI, fisher_blocks
from psiphy.forecasting.dali import _fd_weights


T = np.linspace(0, 1, 15)
SIGMA = 0.1


def cubic_sim(theta, seed, noise_seed):
    """mean = a T + b^2 T^2 + a b^2 T^3 plus additive noise; derivatives known exactly."""
    a, b = theta
    noise = 0.0
    if noise_seed:
        noise = np.random.default_rng(1000 * seed + noise_seed).normal(0, SIGMA, T.size)
    return a * T + b**2 * T**2 + a * b**2 * T**3 + noise


def analytic_d2(a, b):
    d2 = np.zeros((2, 2, T.size))
    d2[0, 1] = d2[1, 0] = 2 * b * T**3
    d2[1, 1] = 2 * T**2 + 2 * a * T**3
    return d2


def _dali(sim=cubic_sim, **kw):
    fm = FisherMatrixDALI(sim, theta_fid=[1.0, 2.0], param_names=["a", "b"], verbose=False, **kw)
    fm.estimate_covariance(seeds=range(1, 21), noise_seeds=range(1, 11))
    fm.estimate_derivatives(delta_frac=0.05, seeds=range(1, 4), noise_seed=0)
    return fm


# ── finite-difference weights ──────────────────────────────────────────────────

@pytest.mark.parametrize("offsets, k, expected", [
    ([-1, 1],           1, [-1/2, 1/2]),
    ([-2, -1, 1, 2],    1, [1/12, -2/3, 2/3, -1/12]),
    ([-1, 0, 1],        2, [1, -2, 1]),
    ([-2, -1, 0, 1, 2], 2, [-1/12, 4/3, -5/2, 4/3, -1/12]),
    ([-2, -1, 1, 2],    3, [-1/2, 1, -1, 1/2]),
])
def test_fd_weights_textbook(offsets, k, expected):
    np.testing.assert_allclose(_fd_weights(offsets, k), expected, atol=1e-12)


def test_fd_weights_rejects_bad_stencils():
    with pytest.raises(ValueError, match="at least 3 stencil points"):
        _fd_weights([-1, 1], 2)
    with pytest.raises(ValueError, match="O\\(h\\^2\\)"):
        _fd_weights([-1, 0, 1], 2, accuracy=4)
    with pytest.raises(ValueError, match="distinct"):
        _fd_weights([-1, 1, 1], 1)


@pytest.mark.parametrize("kw, match", [
    (dict(order=1), "order"),
    (dict(order=4), "order"),
    (dict(accuracy=3), "accuracy"),
    (dict(accuracy=0), "accuracy"),
])
def test_invalid_arguments_raise_before_simulating(kw, match):
    fm = _dali()
    calls = []
    fm.simulator = lambda *a, **k: calls.append(a)
    with pytest.raises(ValueError, match=match):
        fm.estimate_dali_derivatives(**kw)
    assert not calls


def test_requires_covariance_and_first_derivatives():
    fm = FisherMatrixDALI(cubic_sim, theta_fid=[1.0, 2.0], param_names=["a", "b"], verbose=False)
    with pytest.raises(AssertionError, match="estimate_covariance"):
        fm.estimate_dali_derivatives()


# ── derivatives ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("accuracy", [2, 4])
def test_second_derivatives_match_analytic(accuracy):
    # noise is additive and identical at every stencil point, so it cancels
    fm = _dali().estimate_dali_derivatives(order=2, delta_frac=0.05, seeds=range(1, 4),
                                           noise_seeds=[1, 2], accuracy=accuracy)
    np.testing.assert_allclose(fm.d2mu_dtheta, analytic_d2(1.0, 2.0), atol=1e-7)
    assert fm.d2mu_samples_ic.shape == (3, 2, 2, T.size)
    assert fm.d3mu_dtheta is None


def test_third_derivatives_match_analytic():
    fm = _dali().estimate_dali_derivatives(order=3, delta_frac=0.05, seeds=range(1, 3))
    d3 = np.zeros((2, 2, 2, T.size))
    for idx in [(0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        d3[idx] = 2 * T**3
    np.testing.assert_allclose(fm.d2mu_dtheta, analytic_d2(1.0, 2.0), atol=1e-7)
    np.testing.assert_allclose(fm.d3mu_dtheta, d3, atol=1e-5)


def test_reuses_first_derivative_and_covariance_points():
    fm = _dali()
    n0 = len(fm._cache)
    fm.estimate_dali_derivatives(order=2, delta_frac=0.05, seeds=range(1, 4), noise_seed=0)
    # the +-h points are cached by estimate_derivatives; new are the 4 corners of mu_,ab and
    # the noiseless fiducial point (the covariance runs used noise seeds 1..10)
    assert len(fm._cache) - n0 == (4 + 1) * 3
    n1 = len(fm._cache)
    fm.estimate_dali_derivatives(order=2, delta_frac=0.05, seeds=range(1, 4), noise_seed=0)
    assert len(fm._cache) == n1


def test_parallel_matches_serial():
    pytest.importorskip("joblib")
    out = {n_jobs: _dali().estimate_dali_derivatives(order=2, seeds=range(1, 4), noise_seeds=[1, 2],
                                                      n_jobs=n_jobs)
           for n_jobs in (1, 2)}
    np.testing.assert_allclose(out[2].d2mu_dtheta, out[1].d2mu_dtheta)


# ── noise diagnostics ──────────────────────────────────────────────────────────

def _random_curvature_sim(theta, seed, noise_seed):
    """Linear mean, plus a curvature in b whose sign and size change from seed to seed."""
    a, b = theta
    c = np.random.default_rng(seed).normal(0, 50.0)
    noise = np.random.default_rng(1000 * seed + noise_seed).normal(0, SIGMA, T.size) if noise_seed else 0.0
    return a + b * T + c * (b - 2.0)**2 * T**2 + noise


def test_warns_for_noise_dominated_derivatives(capsys):
    fm = _dali(_random_curvature_sim)
    fm.estimate_dali_derivatives(order=2, seeds=range(1, 6))
    out = capsys.readouterr().out
    assert "WARNING" in out and "d(b)(b)" in out
    # mu_,aa and mu_,ab vanish up to roundoff, which is far below the 1-sigma noise tolerance
    assert "d(a)(a)" not in out and "d(a)(b)" not in out
    assert fm.dali_snr[2][(1, 1)] < 3


def test_no_warning_for_well_measured_derivatives(capsys):
    _dali().estimate_dali_derivatives(order=2, seeds=range(1, 4), noise_seeds=[1, 2])
    assert "WARNING" not in capsys.readouterr().out


def test_warns_for_single_seed(capsys):
    fm = _dali().estimate_dali_derivatives(order=2, seeds=[1])
    assert "at least 2 seeds" in capsys.readouterr().out
    assert fm.dali_snr is None


# ── base-class integration ─────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    fm = _dali().estimate_dali_derivatives(order=3, seeds=range(1, 3))
    fm.compute()
    fm.save(tmp_path / "dali.npz")
    fl = FisherMatrixDALI.load(tmp_path / "dali.npz")
    assert isinstance(fl, FisherMatrixDALI)
    np.testing.assert_allclose(fl.d2mu_dtheta, fm.d2mu_dtheta)
    np.testing.assert_allclose(fl.d3mu_samples_ic, fm.d3mu_samples_ic)
    np.testing.assert_allclose(fl.F, fm.F)


def test_fisher_blocks_with_subclass():
    fms = fisher_blocks(cubic_sim, [1.0, 2.0], ["a", "b"], {"all": {}},
                        covariance_kw=dict(seeds=range(1, 5), noise_seeds=range(1, 5)),
                        derivative_kw=dict(seeds=range(1, 3)),
                        fisher_class=FisherMatrixDALI)
    assert isinstance(fms["all"], FisherMatrixDALI)
    assert type(FisherMatrix.joint(list(fms.values()))) is FisherMatrix


# ── DALI likelihood ────────────────────────────────────────────────────────────

def _quadratic_sim(theta, seed, noise_seed):
    """Mean quadratic in theta, so the doublet DALI likelihood is exact."""
    a, b = theta
    noise = np.random.default_rng(1000 * seed + noise_seed).normal(0, SIGMA, T.size) if noise_seed else 0.0
    return a * T + b**2 * T**2 + a * b + noise


def _quadratic_fm(order=2):
    fm = FisherMatrixDALI(_quadratic_sim, theta_fid=[1.0, 2.0], param_names=["a", "b"], verbose=False)
    fm.estimate_covariance(seeds=range(1, 21), noise_seeds=range(1, 11))
    fm.estimate_derivatives(delta_frac=0.05, seeds=range(1, 3), noise_seed=0)
    fm.estimate_dali_derivatives(order=order, delta_frac=0.05, seeds=range(1, 3), noise_seed=0)
    return fm.compute()


def test_order_1_is_the_fisher_gaussian():
    fm = _quadratic_fm()
    D = np.random.default_rng(0).normal(0, 0.1, (50, 2))
    np.testing.assert_allclose(fm.dali_logpdf(fm.theta_fid + D, order=1),
                               -0.5 * np.einsum('na,ab,nb->n', D, fm.F, D), rtol=1e-10)


@pytest.mark.parametrize("order", [2, 3])
def test_doublet_is_exact_for_quadratic_mean(order):
    fm = _quadratic_fm(order)
    theta = fm.theta_fid + np.random.default_rng(1).normal(0, 0.3, (50, 2))
    r = np.array([_quadratic_sim(t, 1, 0) for t in theta]) - _quadratic_sim(fm.theta_fid, 1, 0)
    np.testing.assert_allclose(fm.dali_logpdf(theta, order=order),
                               -0.5 * np.einsum('na,ab,nb->n', r, fm.C_inv, r), rtol=1e-6)


def test_logpdf_matches_tensor_expansion():
    fm = _dali().estimate_dali_derivatives(order=3, seeds=range(1, 3))
    T_ = fm.dali_tensors()
    np.testing.assert_allclose(T_[(1, 1)], fm.dmu_dtheta @ fm.C_inv @ fm.dmu_dtheta.T)
    D = np.array([0.03, -0.05])
    expected = 0.0
    for (i, j), t in T_.items():
        term = t
        for _ in range(i + j):
            term = term @ D
        expected -= (1 if i == j else 2) * term / (2 * factorial(i) * factorial(j))
    np.testing.assert_allclose(fm.dali_logpdf(fm.theta_fid + D), expected, rtol=1e-8)


def test_logpdf_prior_and_order_checks():
    fm = _quadratic_fm()
    assert fm.dali_logpdf([5.0, 2.0], prior_bounds=[(0, 2), (0, 3)]) == -np.inf
    assert np.isfinite(fm.dali_logpdf([1.0, 2.0], prior_bounds=[(0, 2), (0, 3)]))
    with pytest.raises(ValueError, match="order=3"):
        fm.dali_logpdf([1.0, 2.0], order=3)


@pytest.mark.parametrize("method", ["importance", "emcee"])
def test_sampling_recovers_exact_posterior_moments(method):
    np.random.seed(0)
    fm = _quadratic_fm()
    s = fm.sample(20_000, method=method)
    # reference: importance sampling of the exact likelihood with a broad proposal
    rng = np.random.default_rng(2)
    X = fm.theta_fid + rng.normal(0, 3, (200_000, 2)) * fm.sigma_1d()
    r = np.array([_quadratic_sim(t, 1, 0) for t in X]) - _quadratic_sim(fm.theta_fid, 1, 0)
    logw = -0.5 * np.einsum('na,ab,nb->n', r, fm.C_inv, r) + 0.5 * np.sum(((X - fm.theta_fid) / (3 * fm.sigma_1d()))**2, axis=1)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    mean = w @ X
    np.testing.assert_allclose(s.mean(axis=0), mean, atol=0.1 * fm.sigma_1d().max())  # MC error of correlated chains
    np.testing.assert_allclose(s.std(axis=0), np.sqrt(w @ (X - mean)**2), rtol=0.05)


def test_sample_fisher_and_getdist():
    fm = _quadratic_fm()
    assert fm.sample(10, approx='fisher').shape == (10, 2)
    with pytest.raises(ValueError, match="approx"):
        fm.sample(10, approx='gauss')
    g = fm.to_getdist(5000, label='DALI', prior_bounds=[(0, 2), (0, 4)])
    assert g.getParamNames().list() == ["a", "b"]


def test_getdist_fisher_is_the_gaussian():
    '''approx='fisher' must not pick up the DALI default of the overridden sample().'''
    np.random.seed(0)
    fm = _quadratic_fm()
    g = fm.to_getdist(50_000, approx='fisher')
    assert g.weights is None or np.allclose(g.weights, g.weights[0])
    np.testing.assert_allclose(g.getMeans(), fm.theta_fid, atol=0.02 * fm.sigma_1d().max())
    np.testing.assert_allclose(np.cov(g.samples.T), fm.F_inv, rtol=0.05)


def test_linear_model_dali_equals_fisher():
    def linear(theta, seed, noise_seed):
        a, b = theta
        noise = np.random.default_rng(1000 * seed + noise_seed).normal(0, SIGMA, T.size) if noise_seed else 0.0
        return a + b * T + noise
    fm = FisherMatrixDALI(linear, theta_fid=[1.0, 2.0], param_names=["a", "b"], verbose=False)
    fm.estimate_covariance(seeds=range(1, 21), noise_seeds=range(1, 11))
    fm.estimate_derivatives(delta_frac=0.05, seeds=range(1, 3), noise_seed=0)
    fm.estimate_dali_derivatives(order=3, delta_frac=0.05, seeds=range(1, 3), noise_seed=0)
    theta = fm.theta_fid + np.random.default_rng(0).normal(0, 0.5, (50, 2))
    np.testing.assert_allclose(fm.dali_logpdf(theta, order=3), fm.dali_logpdf(theta, order=1),
                               rtol=1e-8, atol=1e-8)
