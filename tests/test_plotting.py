"""Tests for psiphy.plotting.diagnostics."""

import numpy as np
import pytest

from psiphy.plotting.diagnostics import (
    SampledDistribution,
    GriddedProbabilities,
    PosteriorComparison,
    plot_triangle,
    print_chain_stats,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)
TRUE_VALUES = [0.5, 1.0]
N = 2000


def make_samples():
    return RNG.multivariate_normal([0.5, 1.0], [[0.01, 0], [0, 0.02]], size=N)


def make_grid():
    x = np.linspace(0, 1, 40)
    y = np.linspace(0, 2, 50)
    X, Y = np.meshgrid(x, y, indexing='ij')
    grid = np.exp(-0.5 * ((X - 0.5) ** 2 / 0.01 + (Y - 1.0) ** 2 / 0.02))
    return grid, [x, y]


# ── SampledDistribution ───────────────────────────────────────────────────────

class TestSampledDistribution:
    def setup_method(self):
        samples = make_samples()
        self.dist = SampledDistribution(samples, true_values=TRUE_VALUES,
                                        labels=[r'x_1', r'x_2'])

    def test_get_samples_shape(self):
        s, w = self.dist.get_samples()
        assert s.shape == (N, 2)
        assert w is None

    def test_n_params(self):
        assert self.dist.n_params == 2

    def test_weighted_stats(self):
        means, cov, sigmas, *_ = self.dist._weighted_stats()
        assert means.shape == (2,)
        assert cov.shape == (2, 2)
        np.testing.assert_allclose(means, TRUE_VALUES, atol=0.05)

    def test_credible_intervals(self):
        cis = self.dist.credible_intervals(levels=(68, 95))
        assert len(cis) == 2
        for ci in cis:
            assert ci['lo68'] < ci['median'] < ci['hi68']
            assert ci['lo95'] <= ci['lo68']
            assert ci['hi95'] >= ci['hi68']

    def test_z_scores(self):
        z = self.dist.z_scores()
        assert z.shape == (2,)
        assert np.all(z >= 0)

    def test_bias(self):
        b = self.dist.bias()
        assert b.shape == (2,)
        np.testing.assert_allclose(b, 0, atol=0.1)

    def test_pit(self):
        p = self.dist.pit()
        assert p.shape == (2,)
        assert np.all((p >= 0) & (p <= 1))
        np.testing.assert_allclose(p, 0.5, atol=0.15)

    def test_mahalanobis(self):
        d = self.dist.mahalanobis()
        assert np.isfinite(d)
        assert d >= 0

    def test_kl_divergence(self):
        other = SampledDistribution(make_samples())
        kl = self.dist.kl_divergence(other)
        assert kl.shape == (2,)
        assert np.all(kl >= 0)

    def test_print_stats(self, capsys):
        self.dist.print_stats()
        out = capsys.readouterr().out
        assert 'mean' in out
        assert '68%' in out

    def test_weighted_samples(self):
        samples = make_samples()
        w = RNG.dirichlet(np.ones(N))
        dist = SampledDistribution(samples, weights=w, true_values=TRUE_VALUES)
        means, *_ = dist._weighted_stats()
        np.testing.assert_allclose(means, TRUE_VALUES, atol=0.1)

    def test_1d_input(self):
        x = RNG.normal(0.5, 0.1, 500)
        dist = SampledDistribution(x, true_values=[0.5])
        assert dist.n_params == 1
        cis = dist.credible_intervals()
        assert len(cis) == 1


# ── GriddedProbabilities ──────────────────────────────────────────────────────

class TestGriddedProbabilities:
    def setup_method(self):
        grid, coords = make_grid()
        self.grid = grid
        self.coords = coords
        self.dist = GriddedProbabilities(grid, coords=coords,
                                         true_values=TRUE_VALUES,
                                         labels=[r'x_1', r'x_2'],
                                         true_ranges=[[0.4, 0.6], [0.8, 1.2]])

    def test_get_samples(self):
        s, w = self.dist.get_samples()
        assert s.shape[1] == 2
        np.testing.assert_allclose(w.sum(), 1.0)

    def test_credible_intervals_from_marginals(self):
        cis = self.dist.credible_intervals(levels=(68, 95))
        assert len(cis) == 2
        np.testing.assert_allclose(cis[0]['median'], 0.5, atol=0.05)
        np.testing.assert_allclose(cis[1]['median'], 1.0, atol=0.05)

    def test_z_scores(self):
        z = self.dist.z_scores()
        assert np.all(z < 0.5)

    def test_score(self):
        s = self.dist.score()
        assert 0 <= s <= 1
        assert s > 0.5

    def test_score_no_ranges(self):
        dist = GriddedProbabilities(self.grid, coords=self.coords)
        assert np.isnan(dist.score())

    def test_default_coords(self):
        grid = np.ones((10, 10))
        dist = GriddedProbabilities(grid)
        assert dist.n_params == 2

    def test_mahalanobis(self):
        d = self.dist.mahalanobis()
        assert np.isfinite(d)

    def test_pit(self):
        p = self.dist.pit()
        np.testing.assert_allclose(p, 0.5, atol=0.1)


# ── PosteriorComparison ───────────────────────────────────────────────────────

class TestPosteriorComparison:
    def setup_method(self):
        samples_a = make_samples()
        samples_b = make_samples() + 0.1
        self.da = SampledDistribution(samples_a, true_values=TRUE_VALUES,
                                      labels=[r'x_1', r'x_2'])
        self.db = SampledDistribution(samples_b, true_values=TRUE_VALUES,
                                      labels=[r'x_1', r'x_2'])
        self.comp = PosteriorComparison()
        self.comp.add(self.da, label='A')
        self.comp.add(self.db, label='B')

    def test_plot_boxplot(self):
        import matplotlib
        matplotlib.use('Agg')
        fig = self.comp.plot_boxplot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_plot_forest(self):
        import matplotlib
        matplotlib.use('Agg')
        fig = self.comp.plot_forest()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_legend(self):
        import matplotlib
        matplotlib.use('Agg')
        fig = self.comp.legend()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_metrics_table(self):
        result = self.comp.metrics_table()
        try:
            import pandas as pd
            assert hasattr(result, 'columns')
            assert 'mahalanobis' in result.columns
            assert len(result) == 2
        except ImportError:
            assert isinstance(result, list)
            assert len(result) == 2

    def test_empty_raises(self):
        comp = PosteriorComparison()
        with pytest.raises(ValueError, match='No posteriors'):
            comp.plot_boxplot()

    def test_mixed_types(self):
        grid, coords = make_grid()
        dg = GriddedProbabilities(grid, coords=coords, true_values=TRUE_VALUES,
                                  labels=[r'x_1', r'x_2'])
        comp = PosteriorComparison()
        comp.add(self.da, label='Sampled')
        comp.add(dg, label='Grid')
        import matplotlib
        matplotlib.use('Agg')
        fig = comp.plot_boxplot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close('all')


# ── standalone functions ──────────────────────────────────────────────────────

class TestPrintChainStats:
    def test_from_samples(self, capsys):
        samples = make_samples()
        print_chain_stats(samples, labels=[r'x_1', r'x_2'])
        out = capsys.readouterr().out
        assert 'mean' in out
        assert '68%' in out
        assert '95%' in out

    def test_from_precomputed(self, capsys):
        means = np.array([0.5, 1.0])
        sigmas = np.array([0.1, 0.14])
        cis = [{'lo68': 0.4, 'hi68': 0.6, 'lo95': 0.3, 'hi95': 0.7},
               {'lo68': 0.86, 'hi68': 1.14, 'lo95': 0.72, 'hi95': 1.28}]
        print_chain_stats(means, sigmas=sigmas, cis=cis, labels=[r'x_1', r'x_2'])
        out = capsys.readouterr().out
        assert '0.5000' in out


# ── derived parameters & contour plots ───────────────────────────────────────

class TestDerivedParameters:
    def setup_method(self):
        samples = make_samples()
        self.dist = SampledDistribution(samples, true_values=TRUE_VALUES,
                                        labels=[r'x_1', r'x_2'])
        # S8-like derived: x_1 * sqrt(x_2 / 1.0)
        self.dist.add_derived(lambda s: s[:, 0] * np.sqrt(s[:, 1]),
                              label=r'S_8', true_value=0.5 * 1.0)

    def test_get_full_samples_shape(self):
        s, w = self.dist.get_full_samples()
        assert s.shape == (N, 3)  # 2 original + 1 derived
        assert w is None

    def test_full_labels(self):
        lbls = self.dist._full_labels()
        assert len(lbls) == 3
        assert r'S_8' in lbls

    def test_full_true_values(self):
        tv = self.dist._full_true_values()
        assert len(tv) == 3
        assert tv[2] == pytest.approx(0.5)

    def test_resolve_param_by_index(self):
        assert self.dist._resolve_param(0) == 0
        assert self.dist._resolve_param(2) == 2

    def test_resolve_param_by_label(self):
        assert self.dist._resolve_param(r'S_8') == 2

    def test_chaining(self):
        samples = make_samples()
        dist = SampledDistribution(samples)
        result = dist.add_derived(lambda s: s[:, 0] + s[:, 1])
        assert result is dist

    def test_derived_values_correct(self):
        s, _ = self.dist.get_full_samples()
        orig_s, _ = self.dist.get_samples()
        expected = orig_s[:, 0] * np.sqrt(orig_s[:, 1])
        np.testing.assert_allclose(s[:, 2], expected)

    def test_plot_contour_1d_original(self):
        import matplotlib
        matplotlib.use('Agg')
        fig = self.dist.plot_contour_1d(param=0)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_plot_contour_1d_derived(self):
        import matplotlib
        matplotlib.use('Agg')
        fig = self.dist.plot_contour_1d(param=r'S_8')
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_plot_contour_2d(self):
        import matplotlib
        matplotlib.use('Agg')
        fig = self.dist.plot_contour_2d(param1=0, param2=1)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_plot_contour_2d_with_derived(self):
        import matplotlib
        matplotlib.use('Agg')
        fig = self.dist.plot_contour_2d(param1=0, param2=r'S_8')
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_plot_contour_into_existing_axes(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = self.dist.plot_contour_1d(param=0, ax=ax)
        assert result is ax
        plt.close('all')

    def test_grid_derived(self):
        grid, coords = make_grid()
        dist = GriddedProbabilities(grid, coords=coords, true_values=TRUE_VALUES,
                                    labels=[r'x_1', r'x_2'])
        dist.add_derived(lambda s: s[:, 0] * np.sqrt(s[:, 1]),
                         label=r'S_8', true_value=0.5)
        s, w = dist.get_full_samples()
        assert s.shape[1] == 3


class TestPosteriorComparisonContours:
    def setup_method(self):
        import matplotlib
        matplotlib.use('Agg')
        samples_a = make_samples()
        samples_b = make_samples() + 0.1
        da = SampledDistribution(samples_a, true_values=TRUE_VALUES,
                                 labels=[r'x_1', r'x_2'])
        db = SampledDistribution(samples_b, true_values=TRUE_VALUES,
                                 labels=[r'x_1', r'x_2'])
        self.comp = PosteriorComparison()
        self.comp.add(da, label='A')
        self.comp.add(db, label='B')

    def test_contour_1d(self):
        fig = self.comp.plot_contour_1d(param=0)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_contour_2d(self):
        fig = self.comp.plot_contour_2d(param1=0, param2=1)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_contour_2d_filled(self):
        fig = self.comp.plot_contour_2d(param1=0, param2=1, filled=True)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_contour_into_axes(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = self.comp.plot_contour_1d(param=0, ax=ax)
        assert result is ax
        plt.close('all')

    def test_empty_raises(self):
        comp = PosteriorComparison()
        with pytest.raises(ValueError, match='No posteriors'):
            comp.plot_contour_1d()
        with pytest.raises(ValueError, match='No posteriors'):
            comp.plot_contour_2d()
