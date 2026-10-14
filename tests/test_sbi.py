import numpy as np

from psiphy.sbi.rejection_abc import rABC


def test_rejection_abc_recovers_mean(rng, gaussian_model, gaussian_obs):
    theta_true, x_obs = gaussian_obs
    sigma = theta_true[1]
    sim = lambda mu: gaussian_model.simulate([float(mu), sigma])
    dist = lambda a, b: abs(np.mean(a) - np.mean(b))

    abc = rABC(sim, dist, x_obs, prior={"mean": None}, bounds={"mean": (0.0, 3.0)},
               N=2000, eps=0.2)
    abc.run()
    accepted = np.asarray(abc.accepted_param)
    assert accepted.size > 0
    assert abs(np.mean(accepted) - np.mean(x_obs)) < 0.2


# ── BOLFI ─────────────────────────────────────────────────────────────────────

import pytest

from psiphy.sbi.bolfi import BOLFI

MU = np.array([0.5, -1.0])
COV = np.array([[0.2, 0.05], [0.05, 0.1]])
PREC = np.linalg.inv(COV)
BOUNDS = {"a": (-2.0, 3.0), "b": (-3.0, 1.0)}


def quad_distance(theta):
    """Noiseless chi^2 discrepancy: with the 'exp' kernel the posterior is N(MU, COV)."""
    d = np.asarray(theta) - MU
    return float(d @ PREC @ d)


@pytest.fixture(scope="module")
def bolfi():
    np.random.seed(0)
    return BOLFI(quad_distance, BOUNDS, verbose=False).learn_likelihood(n_calls=40, random_state=0)


def test_bolfi_surrogate(bolfi):
    assert bolfi.X.shape == (40, 2) and bolfi.y.shape == (40,)
    assert bolfi.param_names == ["a", "b"]
    np.testing.assert_allclose(bolfi.X[np.argmin(bolfi.y)], MU, atol=0.2)
    pts = np.array([MU, MU + [0.3, 0.2], MU - [0.2, 0.3]])
    np.testing.assert_allclose(bolfi.predict_distance(pts), [quad_distance(p) for p in pts], atol=0.3)
    assert isinstance(bolfi.log_likelihood(MU), float)


def test_bolfi_log_posterior_outside_prior(bolfi):
    assert bolfi.log_posterior([10.0, 0.0]) == -np.inf
    lp = bolfi.log_posterior(np.array([MU, [10.0, 0.0]]))
    assert np.isfinite(lp[0]) and lp[1] == -np.inf


@pytest.mark.parametrize("method, kwargs", [
    ("MCMC", {"n_walkers": 16, "n_burn": 300}),
    ("IS", {}),
    ("SMC", {"n_steps": 5}),
    ("NestedSampling", {"n_live": 200}),
])
def test_bolfi_sample_posterior(bolfi, method, kwargs):
    if method == "MCMC":
        pytest.importorskip("emcee")
    if method == "NestedSampling":
        pytest.importorskip("dynesty")
    np.random.seed(1)
    s = bolfi.sample_posterior(3000, method=method, **kwargs)
    assert s.shape == (3000, 2)
    np.testing.assert_allclose(s.mean(0), MU, atol=0.12)
    np.testing.assert_allclose(np.cov(s.T), COV, atol=0.07)


def test_bolfi_save_load_resume(bolfi, tmp_path):
    fn = str(tmp_path / "bolfi.pkl")
    bolfi.save_likelihood_model(fn)
    fresh = BOLFI(quad_distance, BOUNDS, verbose=False).load_likelihood_model(fn)
    np.testing.assert_allclose(fresh.predict_distance(MU), bolfi.predict_distance(MU), atol=1e-6)

    fresh.learn_likelihood(n_calls=3, filename=fn, random_state=1)
    assert len(fresh.y) == 43
    with pytest.raises(ValueError):
        BOLFI(quad_distance, {"a": (0, 1), "b": (0, 1)}, verbose=False).load_likelihood_model(fn)


def test_bolfi_validation():
    with pytest.raises(ValueError):
        BOLFI(quad_distance, BOUNDS, acquisition="nope")
    with pytest.raises(ValueError):
        BOLFI(quad_distance, BOUNDS, distance_kernel="nope")
    with pytest.raises(RuntimeError):
        BOLFI(quad_distance, BOUNDS).log_likelihood(MU)
    with pytest.raises(ValueError):
        BOLFI(quad_distance, BOUNDS, verbose=False).sample_posterior(10, method="nope")


@pytest.mark.parametrize("acquisition", ["EI", "PI"])
def test_bolfi_other_acquisitions(acquisition):
    b = BOLFI(quad_distance, BOUNDS, acquisition=acquisition, verbose=False)
    b.learn_likelihood(n_calls=25, random_state=0)
    np.testing.assert_allclose(b.X[np.argmin(b.y)], MU, atol=0.3)


def test_bolfi_batch():
    b = BOLFI(quad_distance, BOUNDS, verbose=False)
    b.learn_likelihood(n_calls=30, batch_size=4, random_state=0)
    assert len(b.y) == 30
    assert len(np.unique(np.round(b.X, 6), axis=0)) == 30


# ── LFIRE ─────────────────────────────────────────────────────────────────────

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from psiphy.sbi.lfire import LFIRE
from psiphy.toy_models import GaussianSignal

SIGMA_KNOWN = 0.8


@pytest.fixture(scope="module")
def mean_only():
    """1-D problem: unknown mean, known sigma; posterior N(xbar, sigma^2/N) under a flat prior."""
    np.random.seed(11)
    gs = GaussianSignal(N=40)
    x_obs = gs.simulate([1.5, SIGMA_KNOWN])
    sim = lambda t: gs.simulate([t[0], SIGMA_KNOWN])
    return sim, x_obs


def test_lfire_log_ratio(mean_only):
    sim, x_obs = mean_only
    lf = LFIRE(sim, x_obs, {"mu": (0, 3)}, n_m=200, n_theta=200, summary=lambda x: [x.mean()],
               verbose=False, random_state=0)
    near, score_near = lf.log_ratio([x_obs.mean()], return_score=True)
    far, score_far = lf.log_ratio([0.2], return_score=True)
    assert near > far
    assert 0.5 <= score_near < score_far <= 1.0  # far data are easier to tell apart from the marginal
    assert lf.X_marginal.shape == (200, 1)


def test_lfire_grid_matches_analytic(mean_only):
    sim, x_obs = mean_only
    np.random.seed(12)
    # the log ratio is quadratic in the sample mean, so the features need the square as well
    lf = LFIRE(sim, x_obs, {"mu": (0, 3)}, n_m=400, n_theta=400,
               summary=lambda x: [x.mean(), x.mean() ** 2], penalty="l2", verbose=False, random_state=0)
    th, p = lf.run_grid(n_grid=31)
    assert th.shape == (31,) and np.isclose(p.sum(), 1.0)
    mean = np.sum(th * p)
    std = np.sqrt(np.sum((th - mean) ** 2 * p))
    exact_std = SIGMA_KNOWN / np.sqrt(len(x_obs))
    assert abs(mean - x_obs.mean()) < 0.5 * exact_std
    assert 0.6 * exact_std < std < 1.6 * exact_std


@pytest.fixture(scope="module")
def lfire_2d():
    np.random.seed(3)
    gs = GaussianSignal(N=50)
    x_obs = gs.simulate([1.5, 0.8])
    poly = PolynomialFeatures(2, include_bias=False)
    summary = lambda x: poly.fit_transform([[x.mean(), np.log(x.std())]])[0]
    lf = LFIRE(gs.simulate, x_obs, {"mu": (0, 3), "sigma": (0.1, 2.5)}, n_m=300, n_theta=300,
               summary=summary, penalty="l2", verbose=False, random_state=0)
    lf.learn_posterior(n_calls=40, random_state=0)
    # exact posterior moments from the analytic likelihood on a fine grid
    mu, sg = np.meshgrid(np.linspace(0.6, 2.0, 141), np.linspace(0.45, 1.3, 86))
    logp = np.array([gs.log_prob(t, x_obs) for t in np.c_[mu.ravel(), sg.ravel()]])
    w = np.exp(logp - logp.max()); w /= w.sum()
    pts = np.c_[mu.ravel(), sg.ravel()]
    m = w @ pts
    return lf, m, np.sqrt(w @ (pts - m) ** 2)


def test_lfire_surrogate_posterior(lfire_2d):
    lf, exact_mean, exact_std = lfire_2d
    np.random.seed(1)
    s = lf.sample_posterior(3000, method="MCMC", n_walkers=16, n_burn=300)
    assert np.all(np.abs(s.mean(0) - exact_mean) < exact_std)
    np.testing.assert_allclose(s.std(0), exact_std, rtol=0.5)
    th, p = lf.posterior_grid(n_grid=20)
    assert th.shape == (400, 2) and np.isclose(p.sum(), 1.0)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_lfire_plot(lfire_2d):
    from psiphy.plotting.corner import plot_lfire
    import matplotlib.pyplot as plt
    lf, _, _ = lfire_2d
    lf.posterior_grid(n_grid=20)
    plot_lfire(lf, smooth=1)
    plt.close("all")


def test_lfire_save_resume(mean_only, tmp_path):
    sim, x_obs = mean_only
    fn = str(tmp_path / "lfire.pkl")
    kw = dict(n_m=100, n_theta=100, summary=lambda x: [x.mean()], penalty="l2", verbose=False)
    a = LFIRE(sim, x_obs, {"mu": (0, 3)}, **kw).learn_posterior(n_calls=6, filename=fn, random_state=0)
    b = LFIRE(sim, x_obs, {"mu": (0, 3)}, **kw).learn_posterior(n_calls=2, filename=fn, random_state=1)
    assert len(b.y) == 8
    np.testing.assert_array_equal(b.X_marginal, a.X_marginal)  # marginal set restored, not re-simulated
    with pytest.raises(ValueError):
        LFIRE(sim, x_obs, {"mu": (0, 3)}, **{**kw, "n_theta": 50}).load_model(fn)


def test_lfire_custom_classifier_and_validation(mean_only):
    sim, x_obs = mean_only
    lf = LFIRE(sim, x_obs, {"mu": (0, 3)}, n_m=100, n_theta=100, summary=lambda x: [x.mean()],
               classifier=LogisticRegression(C=10.0), verbose=False)
    assert np.isfinite(lf.log_ratio([1.5]))
    with pytest.raises(ValueError):
        LFIRE(sim, x_obs, {"mu": (0, 3)}, penalty="l3")
    with pytest.raises(RuntimeError):
        lf.log_likelihood([1.5])


def test_bolfi_n_repeats_averages(tmp_path):
    calls = []

    def noisy(theta):
        calls.append(1)
        return quad_distance(theta) + np.random.normal(0, 1.0)

    np.random.seed(0)
    b = BOLFI(noisy, BOUNDS, n_repeats=5, verbose=False).learn_likelihood(n_calls=12, random_state=0)
    assert len(calls) == 60 and b.n_simulations == 60 and len(b.y) == 12
    # the stored value at a point is the mean of n_repeats noisy evaluations
    np.random.seed(1)
    expected = np.mean([quad_distance(MU) + np.random.normal(0, 1.0) for _ in range(5)])
    np.random.seed(1)
    assert np.isclose(b._objective(MU), expected)

    fn = str(tmp_path / "b.pkl")
    b.save_likelihood_model(fn)
    with pytest.raises(ValueError):  # data averaged differently: refuse to mix
        BOLFI(noisy, BOUNDS, n_repeats=1, verbose=False).load_likelihood_model(fn)
    with pytest.raises(ValueError):
        BOLFI(noisy, BOUNDS, n_repeats=0)
