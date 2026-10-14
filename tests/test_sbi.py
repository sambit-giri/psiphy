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
