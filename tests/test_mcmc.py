from types import SimpleNamespace

import numpy as np
import pytest
from psiphy.mcmc import (
    UniformPrior,
    eff_sample_size,
    importance_sampling,
    sequential_importance_sampling,
    SMC,
)
from psiphy.toy_models import GaussianSignal


@pytest.fixture
def gaussian_setup():
    np.random.seed(0)
    gs = GaussianSignal(N=30)
    theta_true = np.array([1.5, 0.8])
    x_obs = gs.simulate(theta_true)
    prior = UniformPrior([(0.0, 3.0), (0.1, 2.5)])
    log_post = lambda t: gs.log_prob(t, x_obs) + prior.log_prob(t)
    return gs, theta_true, x_obs, prior, log_post


class TestUniformPrior:
    def test_sample_shape(self):
        p = UniformPrior([(-1, 1), (0, 2)])
        s = p.sample(50)
        assert s.shape == (50, 2)

    def test_in_support(self):
        p = UniformPrior([(0, 1), (0, 1)])
        assert np.isfinite(p.log_prob([0.5, 0.5]))

    def test_out_of_support(self):
        p = UniformPrior([(0, 1), (0, 1)])
        assert p.log_prob([2.0, 0.5]) == -np.inf

    def test_dict_bounds(self):
        p = UniformPrior({'a': (-1, 1), 'b': (0, 5)})
        assert p.ndim == 2

    def test_sample_within_bounds(self):
        p = UniformPrior([(0, 1), (2, 4)])
        s = p.sample(200)
        assert np.all(s[:, 0] >= 0) and np.all(s[:, 0] <= 1)
        assert np.all(s[:, 1] >= 2) and np.all(s[:, 1] <= 4)


class TestEffSampleSize:
    def test_uniform_weights(self):
        w = np.ones(100)
        assert abs(eff_sample_size(w) - 100) < 1e-6

    def test_degenerate_weights(self):
        w = np.zeros(100)
        w[0] = 1.0
        assert abs(eff_sample_size(w) - 1.0) < 1e-6

    def test_between_bounds(self):
        w = np.random.dirichlet(np.ones(50))
        ess = eff_sample_size(w)
        assert 1 <= ess <= 50


RESULT_KEYS = {"samples", "weights", "equal_weight_samples", "ess", "log_evidence"}


class TestImportanceSampling:
    def test_result(self, gaussian_setup):
        gs, theta_true, x_obs, prior, log_post = gaussian_setup
        prop = prior.sample(500)
        res = importance_sampling(log_post, prop)
        assert set(res) == RESULT_KEYS
        assert res["samples"] is prop  # weights refer to the proposal samples themselves
        assert res["weights"].shape == (500,)
        assert res["equal_weight_samples"].shape == (500, 2)
        assert abs(res["weights"].sum() - 1.0) < 1e-9
        assert 1 <= res["ess"] <= 500
        assert res["log_evidence"] is None

    def test_mean_estimate(self, gaussian_setup):
        gs, theta_true, x_obs, prior, log_post = gaussian_setup
        np.random.seed(1)
        res = importance_sampling(log_post, prior.sample(5000))
        post_mean, post_sigma = gs.analytic_posterior(x_obs)
        est = np.average(res["samples"][:, 0], weights=res["weights"])
        assert abs(est - post_mean) < 2 * post_sigma
        assert abs(res["equal_weight_samples"][:, 0].mean() - post_mean) < 3 * post_sigma

    def test_log_evidence(self):
        # N(0,1) likelihood x Uniform(-10,10) prior: Z = 1/20
        np.random.seed(5)
        prior = UniformPrior([(-10, 10)])
        loglike = lambda t: -0.5 * t[0] ** 2 - 0.5 * np.log(2 * np.pi)
        res = importance_sampling(lambda t: loglike(t) + prior.log_prob(t), prior.sample(20000),
                                  log_proposal_fn=prior.log_prob)
        assert abs(res["log_evidence"] - np.log(1 / 20)) < 0.05


class TestSMC:
    def test_result(self, gaussian_setup):
        gs, theta_true, x_obs, prior, log_post = gaussian_setup
        np.random.seed(2)
        res = SMC(lambda t: gs.log_prob(t, x_obs), prior, n_particles=200, n_steps=3, verbose=False)
        assert set(res) == RESULT_KEYS | {"log_likelihood"}
        assert res["samples"].shape == res["equal_weight_samples"].shape == (200, 2)
        assert res["weights"].shape == res["log_likelihood"].shape == (200,)
        assert abs(res["weights"].sum() - 1.0) < 1e-9

    def test_mean_estimate(self, gaussian_setup):
        gs, theta_true, x_obs, prior, log_post = gaussian_setup
        np.random.seed(4)
        res = SMC(lambda t: gs.log_prob(t, x_obs), prior, n_particles=1000, n_steps=8, verbose=False)
        post_mean, post_sigma = gs.analytic_posterior(x_obs)
        assert abs(np.average(res["samples"][:, 0], weights=res["weights"]) - post_mean) < 2 * post_sigma

    def test_log_evidence(self):
        # N(mu, Sigma) normalised likelihood inside a big box prior: Z = 1 / box volume
        np.random.seed(6)
        prior = UniformPrior([(-10, 10), (-10, 10)])
        cov = np.array([[1.0, 0.3], [0.3, 0.5]])
        icov, norm = np.linalg.inv(cov), -0.5 * np.log(np.linalg.det(2 * np.pi * cov))
        loglike = lambda t: norm - 0.5 * (t - 1.0) @ icov @ (t - 1.0)
        res = SMC(loglike, prior, n_particles=3000, n_steps=20, verbose=False)
        assert abs(res["log_evidence"] - np.log(1 / 400)) < 0.25  # seed-to-seed scatter ~0.07


class TestSIS:
    def test_result(self, gaussian_setup):
        gs, theta_true, x_obs, prior, log_post = gaussian_setup
        np.random.seed(7)
        res = sequential_importance_sampling(log_post, prior, n_particles=300, n_steps=3, verbose=False)
        assert set(res) == RESULT_KEYS
        assert res["samples"].shape == (300, 2)
        assert res["log_evidence"] is None


# ── MHSampler ─────────────────────────────────────────────────────────────────

class TestMHSampler:
    @pytest.fixture(autouse=True)
    def _need_emcee(self):
        pytest.importorskip("emcee")

    def _run(self, log_prob, n_steps=3000, **kw):
        from psiphy.mcmc import MHSampler
        np.random.seed(1)
        return MHSampler(log_prob, n_walkers=16, **kw).run(log_prob.mean, n_steps=n_steps)

    def test_recovers_gaussian(self, gaussian_log_prob):
        s = self._run(gaussian_log_prob)
        chain = s.get_chain(flat=True, discard=500, thin=5)
        np.testing.assert_allclose(chain.mean(0), gaussian_log_prob.mean, atol=0.1)
        np.testing.assert_allclose(np.cov(chain.T), gaussian_log_prob.cov, atol=0.1)

    def test_chain_shapes(self, gaussian_log_prob):
        s = self._run(gaussian_log_prob, n_steps=200)
        assert s.get_chain().shape == (200, 16, 2)
        assert s.get_chain(flat=True, discard=100, thin=2).shape == (50 * 16, 2)
        assert s.get_log_prob(flat=True).shape == (200 * 16,)
        assert s.n_steps == 200

    def test_continue_run(self, gaussian_log_prob):
        s = self._run(gaussian_log_prob, n_steps=100)
        s.run(n_steps=50)
        assert s.n_steps == 150

    def test_summary(self, gaussian_log_prob):
        s = self._run(gaussian_log_prob, param_names=["a", "b"])
        out = s.summary(print_out=False)
        assert set(out) >= {"a", "b", "acceptance_fraction", "n_samples"}
        assert abs(out["a"]["mean"] - 0.5) < 0.1
        assert 0 < out["acceptance_fraction"] < 1

    def test_save_load_and_resume(self, gaussian_log_prob, tmp_path):
        pytest.importorskip("h5py")
        from psiphy.mcmc import MHSampler
        fn = str(tmp_path / "chain.h5")
        s = self._run(gaussian_log_prob, n_steps=100, filename=fn)
        loaded = MHSampler.load(fn, log_prob_fn=gaussian_log_prob)
        np.testing.assert_array_equal(loaded.get_chain(), s.get_chain())

        resumed = MHSampler(gaussian_log_prob, n_walkers=16, filename=fn).run(n_steps=40)
        assert resumed.n_steps == 140
        with pytest.raises(ValueError):
            MHSampler(gaussian_log_prob, n_walkers=16, filename=fn).run(gaussian_log_prob.mean, 10)

    def test_bad_backend(self, gaussian_log_prob):
        from psiphy.mcmc import MHSampler
        with pytest.raises(ValueError):
            MHSampler(gaussian_log_prob, backend="nope")

    def test_requires_init(self, gaussian_log_prob):
        from psiphy.mcmc import MHSampler
        with pytest.raises(ValueError):
            MHSampler(gaussian_log_prob).run(n_steps=10)


# ── NestedSampler ─────────────────────────────────────────────────────────────

def _normalised_gaussian_loglike(mean, cov):
    icov = np.linalg.inv(cov)
    norm = -0.5 * np.log(np.linalg.det(2 * np.pi * cov))
    return lambda t: norm - 0.5 * (np.asarray(t) - mean) @ icov @ (np.asarray(t) - mean)


class TestNestedSampler:
    BOX = [(-10.0, 10.0), (-10.0, 10.0)]

    @pytest.fixture(scope="class", params=["dynesty", "nautilus"])
    def result(self, request):
        backend = request.param
        pytest.importorskip({"dynesty": "dynesty", "nautilus": "nautilus"}[backend])
        from psiphy.mcmc import NestedSampler
        from psiphy.utils.prior import UniformPrior
        target = SimpleNamespace(mean=np.array([0.5, -1.0]),
                                 cov=np.array([[1.0, 0.3], [0.3, 0.5]]))
        loglike = _normalised_gaussian_loglike(target.mean, target.cov)
        ns = NestedSampler(loglike, UniformPrior({"a": self.BOX[0], "b": self.BOX[1]}),
                           backend=backend, n_live=300, seed=3)
        return ns.run(dlogz=0.1), target, backend

    def test_evidence(self, result):
        ns, _, _ = result
        # Likelihood is a normalised Gaussian well inside the box: Z = 1 / box volume
        assert abs(ns.get_result()["log_evidence"] - (-np.log(400.0))) < 0.3

    def test_posterior_moments(self, result):
        ns, target, _ = result
        res = ns.get_result()
        mean = np.average(res["samples"], weights=res["weights"], axis=0)
        np.testing.assert_allclose(mean, target.mean, atol=0.15)
        np.testing.assert_allclose(res["equal_weight_samples"].mean(0), target.mean, atol=0.15)
        np.testing.assert_allclose(np.cov(res["equal_weight_samples"].T), target.cov, atol=0.2)

    def test_result_keys(self, result):
        ns, _, backend = result
        res = ns.get_result()
        assert set(res) == {"log_evidence", "log_evidence_err", "samples", "weights",
                            "log_likelihood", "equal_weight_samples", "ess", "backend"}
        assert np.isclose(res["weights"].sum(), 1.0)
        assert res["samples"].shape[1] == 2
        assert res["log_likelihood"].shape == res["weights"].shape
        assert (res["log_evidence_err"] is None) == (backend == "nautilus")

    def test_to_getdist(self, result):
        ns, target, _ = result
        mc = ns.to_getdist()
        assert mc.getParamNames().list() == ["a", "b"]
        np.testing.assert_allclose(mc.getMeans(), target.mean, atol=0.15)


def test_nested_sampler_validation():
    from psiphy.mcmc import NestedSampler
    with pytest.raises(ValueError):
        NestedSampler(lambda t: 0.0, lambda u: u)  # n_dim missing
    with pytest.raises(ValueError):
        NestedSampler(lambda t: 0.0, lambda u: u, n_dim=2, backend="multinest")
    with pytest.raises(RuntimeError):
        NestedSampler(lambda t: 0.0, lambda u: u, n_dim=2).get_result()


# ── vectorised evaluation ─────────────────────────────────────────────────────

class TestVectorized:
    """Batched (vectorized=True) and per-particle paths must agree."""

    MEAN = np.array([1.0, 0.5])

    @staticmethod
    def _loglike_batch(theta):
        theta = np.atleast_2d(theta)
        return -0.5 * np.sum(((theta - TestVectorized.MEAN) / 0.2) ** 2, axis=1)

    def _loglike_single(self, theta):
        return float(self._loglike_batch(theta)[0])

    def test_importance_sampling_same_weights(self):
        prop = UniformPrior([(0, 2), (0, 1)]).sample(500)
        w_loop = importance_sampling(self._loglike_single, prop)["weights"]
        w_vec = importance_sampling(self._loglike_batch, prop, vectorized=True)["weights"]
        np.testing.assert_allclose(w_vec, w_loop)

    @pytest.mark.parametrize("fn", [SMC, sequential_importance_sampling])
    def test_recovers_mean(self, fn):
        np.random.seed(2)
        prior = UniformPrior([(0, 2), (0, 1)])
        res = fn(self._loglike_batch, prior, n_particles=1000, verbose=False, vectorized=True)
        np.testing.assert_allclose(np.average(res["samples"], weights=res["weights"], axis=0),
                                   self.MEAN, atol=0.05)

    def test_smc_single_batched_call_per_step(self):
        calls = []

        def counting(theta):
            calls.append(len(theta))
            return self._loglike_batch(theta)

        np.random.seed(3)
        SMC(counting, UniformPrior([(0, 2), (0, 1)]), n_particles=300, n_steps=4,
            verbose=False, vectorized=True)
        assert calls[0] == 300            # initial batch
        assert len(calls) <= 1 + 4        # at most one batched call per move step
