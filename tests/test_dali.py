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


# ── combining blocks ───────────────────────────────────────────────────────────

def _cubic_block(theta, seed, noise_seed, sl):
    return cubic_sim(theta, seed, noise_seed)[sl]


_SPLIT = {'lo': dict(sl=slice(0, 6)), 'hi': dict(sl=slice(6, None))}
_BLOCK_KW = dict(covariance_kw=dict(seeds=range(1, 21), noise_seeds=range(1, 11)),
                 derivative_kw=dict(delta_frac=0.05, seeds=range(1, 4), noise_seed=0),
                 dali_kw=dict(order=3, delta_frac=0.05, seeds=range(1, 4), noise_seeds=[1, 2]))


def _full_dali():
    fm = FisherMatrixDALI(cubic_sim, theta_fid=[1.0, 2.0], param_names=["a", "b"], verbose=False)
    fm.estimate_covariance(**_BLOCK_KW['covariance_kw'])
    fm.estimate_derivatives(**_BLOCK_KW['derivative_kw'])
    return fm.estimate_dali_derivatives(**_BLOCK_KW['dali_kw']).compute()


def test_joint_of_split_vector_matches_full():
    full = _full_dali()
    fms = fisher_blocks(_cubic_block, [1.0, 2.0], ["a", "b"], _SPLIT,
                        fisher_class=FisherMatrixDALI, **_BLOCK_KW)
    joint = FisherMatrixDALI.joint(list(fms.values()), verbose=False).compute()
    assert isinstance(joint, FisherMatrixDALI)
    np.testing.assert_allclose(joint.d2mu_dtheta, full.d2mu_dtheta)
    np.testing.assert_allclose(joint.d3mu_samples_ic, full.d3mu_samples_ic)
    theta = full.theta_fid + np.random.default_rng(0).normal(0, 0.2, (20, 2))
    np.testing.assert_allclose(joint.dali_logpdf(theta), full.dali_logpdf(theta), rtol=1e-8)


def test_joint_checks_blocks():
    fms = fisher_blocks(_cubic_block, [1.0, 2.0], ["a", "b"], _SPLIT,
                        fisher_class=FisherMatrixDALI, **{**_BLOCK_KW, 'dali_kw': None})
    with pytest.raises(AssertionError, match="estimate_dali_derivatives"):
        FisherMatrixDALI.joint(list(fms.values()))
    lo, hi = fms.values()
    lo.estimate_dali_derivatives(order=3, seeds=range(1, 4), noise_seed=0)
    hi.estimate_dali_derivatives(order=2, seeds=range(1, 3), noise_seed=0)
    with pytest.raises(ValueError, match="different seeds"):
        FisherMatrixDALI.joint([lo, hi])
    hi.estimate_dali_derivatives(order=2, seeds=range(1, 4), noise_seed=0)
    joint = FisherMatrixDALI.joint([lo, hi], verbose=False)
    assert joint.d2mu_dtheta.shape == (2, 2, T.size) and joint.d3mu_dtheta is None


def test_fisher_blocks_dali_kw_needs_subclass():
    with pytest.raises(ValueError, match="FisherMatrixDALI"):
        fisher_blocks(_cubic_block, [1.0, 2.0], ["a", "b"], _SPLIT, **_BLOCK_KW)


def test_add_independent_data_sums_log_likelihoods():
    a, b = _full_dali(), _quadratic_fm(order=2)
    combined = a + b
    assert isinstance(combined, FisherMatrixDALI)
    np.testing.assert_allclose(combined.F, a.F + b.F)
    theta = a.theta_fid + np.random.default_rng(0).normal(0, 0.2, (20, 2))
    # b stops at order 2, so the combination does too
    np.testing.assert_allclose(combined.dali_logpdf(theta),
                               a.dali_logpdf(theta, order=2) + b.dali_logpdf(theta), rtol=1e-8)
    assert combined.d3mu_dtheta is None
    fisher_only = a + FisherMatrix.joint([_full_dali()], verbose=False).compute()
    np.testing.assert_allclose(fisher_only.dali_logpdf(theta), 2 * a.dali_logpdf(theta, order=1), rtol=1e-8)


# ── step-size convergence and plotting ─────────────────────────────────────────

def test_convergence_test_higher_orders_restores():
    fm = _dali().estimate_dali_derivatives(order=3, delta_frac=0.05, seeds=range(1, 3))
    d2, d3 = fm.d2mu_dtheta.copy(), fm.d3mu_dtheta.copy()
    res = fm.convergence_test_delta_frac(order=3, delta_fracs=[0.01, 0.05, 0.2], seeds=range(1, 3), plot=True)
    assert set(res) == {'d(a)(a)', 'd(a)(b)', 'd(b)(b)', 'd(a)(a)(a)', 'd(a)(a)(b)', 'd(a)(b)(b)', 'd(b)(b)(b)'}
    # cubic mean: the stencils are exact at every step
    np.testing.assert_allclose(res['d(b)(b)']['norms'], np.linalg.norm(analytic_d2(1.0, 2.0)[1, 1]), rtol=1e-6)
    assert res['d(a)(b)(b)']['snr'].shape == (3,)
    np.testing.assert_array_equal(fm.d2mu_dtheta, d2)
    np.testing.assert_array_equal(fm.d3mu_dtheta, d3)


def test_convergence_test_order_1_is_the_base_test():
    fm = _dali()
    res = fm.convergence_test_delta_frac(order=1, delta_fracs=[0.01, 0.1], seeds=range(1, 3), plot=False)
    assert set(res) == {'a', 'b'} and len(res['a']['norms']) == 2
    with pytest.raises(ValueError, match="order"):
        fm.convergence_test_delta_frac(order=4)


def test_plot_dali():
    from getdist import MCSamples
    fm = _quadratic_fm(order=3)
    exact = MCSamples(samples=fm.sample(2000), names=["a", "b"])
    g = fm.plot_dali(others={'Exact': exact}, n_samples=5000)
    labels = [t.get_text() for t in g.legend.get_texts()]
    assert labels == ['Fisher', 'DALI-2', 'DALI-3', 'Exact']


# ── non-linear toy model ───────────────────────────────────────────────────────

def test_power_law_spectrum_dali_beats_fisher():
    '''PowerLawSpectrum: analytic derivatives are recovered, and DALI-3 moments match the
    exact posterior where the Fisher ellipse does not.'''
    from psiphy.toy_models import PowerLawSpectrum
    from psiphy.mcmc import importance_sampling
    from scipy.stats import multivariate_normal

    model = PowerLawSpectrum(ells=np.geomspace(10, 2000, 10), noise_sigma=0.5)

    def sim(theta, seed, noise_seed):
        x = model._spectrum(theta)
        return x + np.random.default_rng(1000 * seed + noise_seed).normal(0, 0.5, x.size) if noise_seed else x

    fm = FisherMatrixDALI(sim, theta_fid=[1.0, 1.0], param_names=model.param_names, verbose=False)
    fm.estimate_covariance(seeds=range(1, 51), noise_seeds=range(1, 11))
    fm.estimate_derivatives(delta_frac=0.01, seeds=[1], noise_seed=0)
    fm.estimate_dali_derivatives(order=3, delta_frac=0.02, seeds=[1, 2], noise_seed=0)
    fm.compute()

    P, L = (model.ells / model.ell_pivot) ** 0, np.log(model.ells / model.ell_pivot)
    np.testing.assert_allclose(fm.d2mu_dtheta[0, 1], P * L, atol=1e-2)
    np.testing.assert_allclose(fm.d2mu_dtheta[1, 1], P * L**2, atol=1e-2)
    np.testing.assert_allclose(fm.d3mu_dtheta[1, 1, 1], P * L**3, atol=5e-2)

    proposal = multivariate_normal(fm.theta_fid, 9 * fm.F_inv)
    X = proposal.rvs(100_000, random_state=0)

    def moments(logp):
        w = importance_sampling(logp, X, log_proposal_fn=proposal.logpdf, vectorized=True)['weights']
        mean = w @ X
        return mean, np.sqrt(w @ (X - mean)**2)

    def exact(theta):
        r = np.array([model._spectrum(t) for t in theta]) - model._spectrum(fm.theta_fid)
        return -0.5 * np.einsum('na,ab,nb->n', r, fm.C_inv, r)

    m_ex, s_ex = moments(exact)
    m_f, s_f = moments(lambda t: fm.dali_logpdf(t, order=1))
    m_d, s_d = moments(lambda t: fm.dali_logpdf(t, order=3))
    sigma = fm.sigma_1d()
    assert abs(m_f[0] - m_ex[0]) > 0.15 * sigma[0]          # Fisher misses the shift in A_s
    np.testing.assert_allclose(m_d, m_ex, atol=0.03 * sigma.min())
    np.testing.assert_allclose(s_d, s_ex, rtol=0.05)
