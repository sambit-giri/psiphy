import numpy as np
import pytest
from psiphy.utils import sampling_space


def test_lh_sampling_shape():
    lhd = sampling_space.LH_sampling(n_params=3, samples=5, mins=0, maxs=1)
    assert lhd.shape == (5, 3)


def test_lh_sampling_bounds():
    lhd = sampling_space.LH_sampling(n_params=2, samples=10, mins=[-1, 0], maxs=[1, 5])
    assert np.all(lhd[:, 0] >= -1) and np.all(lhd[:, 0] <= 1)
    assert np.all(lhd[:, 1] >= 0) and np.all(lhd[:, 1] <= 5)


def test_mc_sampling_shape():
    mcd = sampling_space.MC_sampling(n_params=2, samples=8, mins=0, maxs=1)
    assert mcd.shape == (8, 2)


def test_mc_sampling_bounds():
    mcd = sampling_space.MC_sampling(n_params=2, samples=20, mins=[0, -2], maxs=[1, 2])
    assert np.all(mcd[:, 0] >= 0) and np.all(mcd[:, 0] <= 1)
    assert np.all(mcd[:, 1] >= -2) and np.all(mcd[:, 1] <= 2)


# ── priors ────────────────────────────────────────────────────────────────────

from scipy import integrate, stats

from psiphy.utils.prior import (
    UniformPrior,
    LogUniformPrior,
    GaussianPrior,
    MixturePrior,
)


def _priors():
    return {
        "uniform": UniformPrior({"a": (-1, 1), "b": (0, 5)}),
        "loguniform": LogUniformPrior([(0.1, 10), (1e-3, 1)]),
        "gaussian": GaussianPrior([0.5, -1.0], [[1.0, 0.3], [0.3, 0.5]]),
        "mixture": MixturePrior([GaussianPrior([-2, 0], 0.25), GaussianPrior([2, 0], 0.25)],
                                weights=[1, 3]),
    }


@pytest.mark.parametrize("name", list(_priors()))
def test_prior_sample_and_log_prob_shapes(rng, name):
    p = _priors()[name]
    s = p.sample(200)
    assert s.shape == (200, 2)
    lp = p.log_prob(s)
    assert lp.shape == (200,)
    assert np.all(np.isfinite(lp))
    assert isinstance(p.log_prob(s[0]), float)
    np.testing.assert_allclose(p.log_prob(s[:5]), [p.log_prob(t) for t in s[:5]])


@pytest.mark.parametrize("name", ["uniform", "loguniform", "gaussian"])
def test_prior_transform_matches_sampling(rng, name):
    p = _priors()[name]
    u = rng.uniform(size=(4000, 2))
    t = p.prior_transform(u)
    s = p.sample(4000)
    assert np.all(np.isfinite(p.log_prob(t)))
    for d in range(2):
        assert stats.ks_2samp(t[:, d], s[:, d]).pvalue > 1e-3


def test_uniform_prior_support_and_normalisation():
    p = UniformPrior([(0, 2), (0, 5)])
    assert p.log_prob([3.0, 1.0]) == -np.inf
    assert not p.in_support([3.0, 1.0])
    assert np.isclose(np.exp(p.log_prob([1.0, 1.0])), 1 / 10)
    assert UniformPrior({"x": (0, 1)}).names == ["x"]


def test_loguniform_normalised():
    p = LogUniformPrior([(0.1, 10)])
    val, _ = integrate.quad(lambda x: np.exp(p.log_prob(x)), 0.1, 10)
    assert np.isclose(val, 1.0)
    assert p.log_prob(20.0) == -np.inf


def test_loguniform_rejects_nonpositive():
    with pytest.raises(ValueError):
        LogUniformPrior([(0, 1)])


def test_gaussian_moments(rng):
    p = _priors()["gaussian"]
    s = p.sample(20000)
    np.testing.assert_allclose(s.mean(0), p.mean, atol=0.05)
    np.testing.assert_allclose(np.cov(s.T), p.cov, atol=0.05)
    assert np.isclose(p.log_prob(p.mean), stats.multivariate_normal(p.mean, p.cov).logpdf(p.mean))


def test_mixture_weights(rng):
    p = _priors()["mixture"]
    s = p.sample(10000)
    assert np.isclose(np.mean(s[:, 0] > 0), 0.75, atol=0.03)


def test_mixture_dim_mismatch():
    with pytest.raises(ValueError):
        MixturePrior([UniformPrior([(0, 1)]), UniformPrior([(0, 1), (0, 1)])])


def test_scipy_conversion():
    p = UniformPrior([(0, 2), (1, 3)])
    sp = p.to_scipy()
    assert np.isclose(sum(d.logpdf(x) for d, x in zip(sp, [1.0, 2.0])), p.log_prob([1.0, 2.0]))


@pytest.mark.parametrize("name", ["uniform", "loguniform", "gaussian"])
def test_torch_conversion(name):
    torch = pytest.importorskip("torch")
    p = _priors()[name]
    td = p.to_torch()
    x = p.sample(5)
    np.testing.assert_allclose(td.log_prob(torch.as_tensor(x, dtype=torch.float32)).numpy(),
                               p.log_prob(x), rtol=1e-4)


def test_mcmc_reexports_uniform_prior():
    from psiphy.mcmc import UniformPrior as McmcUniformPrior
    assert McmcUniformPrior is UniformPrior


# ── Bayesian optimisation ─────────────────────────────────────────────────────

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from psiphy.utils import bayesian_opt as bopt


@pytest.fixture
def gp_1d():
    X = np.array([[0.1], [0.4], [0.6], [0.9]])
    y = (X[:, 0] - 0.3) ** 2
    return GaussianProcessRegressor(Matern(0.3, nu=2.5), normalize_y=True).fit(X, y), X, y


@pytest.mark.parametrize("name", ["EI", "PI", "LCB", "UCB"])
def test_acquisition_shapes(gp_1d, name):
    gpr, X, y = gp_1d
    v = bopt.get_acquisition(name)(np.linspace(0, 1, 7)[:, None], gpr, y.min())
    assert v.shape == (7,) and np.all(np.isfinite(v))


def test_ei_zero_at_far_worse_point(gp_1d):
    gpr, X, y = gp_1d
    ei = bopt.expected_improvement(X, gpr, y_best=y.min())
    assert ei[np.argmax(y)] < ei[np.argmin(y)] + 1e-3


@pytest.mark.parametrize("name", ["EI", "LCB"])
def test_propose_near_minimum(gp_1d, name):
    gpr, X, y = gp_1d
    x = bopt.propose_location(gpr, [[0, 1]], acquisition=name, X_sample=X, y_sample=y,
                              random_state=0, **({"kappa": 0.1} if name == "LCB" else {}))
    assert x.shape == (1, 1)
    assert abs(x[0, 0] - 0.3) < 0.15


def test_propose_maximise(gp_1d):
    gpr, X, y = gp_1d
    x = bopt.propose_location(gpr, [[0, 1]], acquisition="UCB", minimize=False, kappa=0.0, random_state=0)
    assert x[0, 0] > 0.85  # GP mean is largest at the right edge


def test_propose_batch_distinct(gp_1d):
    gpr, X, y = gp_1d
    xs = bopt.propose_location(gpr, [[0, 1]], acquisition="LCB", X_sample=X, y_sample=y,
                               batch_size=3, random_state=0)
    assert xs.shape == (3, 1)
    assert np.min(np.diff(np.sort(xs[:, 0]))) > 1e-3


def test_propose_validation(gp_1d):
    gpr, X, y = gp_1d
    with pytest.raises(ValueError):
        bopt.propose_location(gpr, [[0, 1]], acquisition="nope")
    with pytest.raises(ValueError):
        bopt.propose_location(gpr, [[0, 1]], acquisition="EI")
    with pytest.raises(ValueError):
        bopt.propose_location(gpr, [[0, 1]], batch_size=2)
