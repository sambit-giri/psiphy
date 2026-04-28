import numpy as np
import pytest
from psiphy.toy_models import NoisyLine, GaussianSignal, MA2Model, PowerLawSpectrum, LotkaVolterra


class TestNoisyLine:
    def setup_method(self):
        self.model = NoisyLine(Nx=20)

    def test_simulate_shape(self):
        theta = np.array([-0.9, 4.0])
        sim = self.model.simulate(theta)
        assert sim.shape == (20,)

    def test_sample_prior_shape(self):
        theta = self.model.sample_prior()
        assert theta.shape == (2,)

    def test_sample_prior_in_bounds(self):
        for _ in range(20):
            slope, intercept = self.model.sample_prior()
            assert self.model.slope_range[0] <= slope <= self.model.slope_range[1]
            assert self.model.intercept_range[0] <= intercept <= self.model.intercept_range[1]

    def test_log_prob_finite(self):
        theta = np.array([-0.9, 4.0])
        x = self.model.simulate(theta)
        lp = self.model.log_prob(theta, x)
        assert np.isfinite(lp)

    def test_x_grid_length(self):
        assert len(self.model.x) == 20


class TestGaussianSignal:
    def setup_method(self):
        self.model = GaussianSignal(N=30)

    def test_simulate_shape(self):
        theta = np.array([0.5, 1.5])
        sim = self.model.simulate(theta)
        assert sim.shape == (30,)

    def test_sample_prior_shape(self):
        theta = self.model.sample_prior()
        assert theta.shape == (2,)

    def test_sample_prior_in_bounds(self):
        for _ in range(20):
            mean, sigma = self.model.sample_prior()
            assert self.model.mean_range[0] <= mean <= self.model.mean_range[1]
            assert self.model.sigma_range[0] <= sigma <= self.model.sigma_range[1]

    def test_log_prob_finite(self):
        theta = np.array([0.0, 1.0])
        x = self.model.simulate(theta)
        lp = self.model.log_prob(theta, x)
        assert np.isfinite(lp)

    def test_log_prob_negative_sigma(self):
        theta = np.array([0.0, -1.0])
        x = self.model.simulate(np.array([0.0, 1.0]))
        assert self.model.log_prob(theta, x) == -np.inf


class TestMA2Model:
    def setup_method(self):
        self.model = MA2Model(n_obs=50)

    def test_simulate_shape(self):
        theta = np.array([0.6, 0.2])
        summ = self.model.simulate(theta)
        assert summ.shape == (4,)

    def test_simulate_raw_shape(self):
        theta = np.array([0.6, 0.2])
        x = self.model.simulate_raw(theta)
        assert x.shape == (50,)

    def test_sample_prior_in_region(self):
        for _ in range(50):
            t1, t2 = self.model.sample_prior()
            assert abs(t2) < 1
            assert t2 + t1 > -1
            assert t2 - t1 > -1

    def test_log_prob_is_none(self):
        theta = self.model.sample_prior()
        x = self.model.simulate(theta)
        assert self.model.log_prob(theta, x) is None


class TestPowerLawSpectrum:
    def setup_method(self):
        self.model = PowerLawSpectrum()

    def test_simulate_shape(self):
        theta = np.array([1.0, 0.965])
        x = self.model.simulate(theta)
        assert x.shape == (len(self.model.ells),)

    def test_sample_prior_in_bounds(self):
        for _ in range(20):
            A_s, n_s = self.model.sample_prior()
            assert self.model.A_s_range[0] <= A_s <= self.model.A_s_range[1]
            assert self.model.n_s_range[0] <= n_s <= self.model.n_s_range[1]

    def test_log_prob_finite(self):
        theta = np.array([1.0, 0.965])
        x = self.model.simulate(theta)
        assert np.isfinite(self.model.log_prob(theta, x))


class TestLotkaVolterra:
    def setup_method(self):
        self.model = LotkaVolterra()
        self.theta = np.array([1.0, 0.1, 0.075, 1.5])

    def test_simulate_shape(self):
        x = self.model.simulate(self.theta)
        assert x.shape == (2 * len(self.model.t_obs),)

    def test_noiseless_shape(self):
        traj = self.model.noiseless(self.theta)
        assert traj.shape == (2, len(self.model.t_obs))

    def test_log_prob_finite(self):
        x = self.model.simulate(self.theta)
        assert np.isfinite(self.model.log_prob(self.theta, x))

    def test_sample_prior_shape(self):
        theta = self.model.sample_prior()
        assert theta.shape == (4,)
