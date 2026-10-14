import numpy as np
import pytest

from psiphy.forecasting.fisher import FisherMatrix, fisher_blocks


T = np.linspace(0, 1, 15)
SIGMA = 0.1


def linear_sim(theta, seed, noise_seed):
    """x = a + b*t + noise; analytic Fisher F = J^T J / sigma^2."""
    a, b = theta
    noise = 0.0
    if noise_seed:
        noise = np.random.default_rng(1000 * seed + noise_seed).normal(0, SIGMA, T.size)
    return a + b * T + noise


@pytest.fixture
def fisher():
    fm = FisherMatrix(linear_sim, theta_fid=[1.0, 2.0], param_names=["a", "b"], verbose=False)
    fm.estimate_covariance(seeds=range(1, 41), noise_seeds=range(1, 26))
    fm.estimate_derivatives(delta_frac=0.01, seeds=range(1, 3), noise_seed=0)
    return fm.compute()


def test_shapes(fisher):
    assert fisher.F.shape == (2, 2)
    assert fisher.sigma_1d().shape == (2,)


def test_matches_analytic(fisher):
    J = np.stack([np.ones_like(T), T], axis=1)
    F_true = J.T @ J / SIGMA**2
    np.testing.assert_allclose(fisher.F, F_true, rtol=0.15)


def test_correlation_matrix(fisher):
    R = fisher.correlation_matrix()
    np.testing.assert_allclose(np.diag(R), 1.0)
    assert np.all(np.abs(R) <= 1.0 + 1e-12)


# ── parallel (joblib) paths must match serial ─────────────────────────────────

def _fm(**kw):
    return FisherMatrix(linear_sim, theta_fid=[1.0, 2.0], param_names=["a", "b"], verbose=False, **kw)


def test_parallel_matches_serial():
    pytest.importorskip("joblib")
    out = {}
    for n_jobs in (1, 2):
        fm = _fm()
        fm.estimate_covariance(seeds=range(1, 9), noise_seeds=range(1, 6), n_jobs=n_jobs)
        fm.estimate_derivatives(delta_frac=0.01, seeds=range(1, 3), noise_seeds=[1, 2],
                                n_stencil=4, n_jobs=n_jobs)
        out[n_jobs] = fm.compute()
    np.testing.assert_allclose(out[2].C, out[1].C)
    np.testing.assert_allclose(out[2].dmu_dtheta, out[1].dmu_dtheta)
    np.testing.assert_allclose(out[2].F, out[1].F)


def test_run_simulations_parallel(tmp_path):
    pytest.importorskip("joblib")
    fm = _fm(cache_file=str(tmp_path / "cache.pkl"))
    fm.run_simulations(seeds=range(1, 5), noise_seeds=range(3), n_jobs=2)
    assert len(fm._cache) == 12
    x = fm._cache[fm._cache_key(fm.theta_fid, 3, 2)]
    np.testing.assert_allclose(x, linear_sim(fm.theta_fid, 3, 2))
    # second call: everything cached, nothing re-run
    assert fm._prerun_parallel([fm.theta_fid], range(1, 5), range(3), 2, desc="") == 0
    assert len(_fm(cache_file=str(tmp_path / "cache.pkl"))._cache) == 12


def test_convergence_test_parallel():
    pytest.importorskip("joblib")
    res = {}
    for n_jobs in (1, 2):
        fm = _fm()
        fm.estimate_covariance(seeds=range(1, 9), noise_seeds=range(1, 6))
        res[n_jobs] = fm.convergence_test_delta_frac(delta_fracs=[0.01, 0.1], seeds=range(1, 3),
                                                     plot=False, n_jobs=n_jobs)
    for p in ("a", "b"):
        np.testing.assert_allclose(res[2][p]["norms"], res[1][p]["norms"])


# ── joint Fisher of correlated blocks ─────────────────────────────────────────

def _block_sim(sl):
    return lambda theta, seed, noise_seed: linear_sim(theta, seed, noise_seed)[sl]


def _run(sim, **deriv_kw):
    fm = FisherMatrix(sim, theta_fid=[1.0, 2.0], param_names=["a", "b"], verbose=False)
    fm.estimate_covariance(seeds=range(1, 9), noise_seeds=range(1, 6))
    fm.estimate_derivatives(delta_frac=0.01, seeds=range(1, 4), **deriv_kw)
    return fm


def test_joint_of_split_vector_matches_full():
    '''Splitting one data vector into blocks and joining them gives the full Fisher matrix.'''
    kw = dict(noise_seeds=[1, -1, 2, -2])
    full = _run(linear_sim, **kw).compute(debias_derivatives=True)
    blocks = [_run(_block_sim(slice(0, 6)), **kw), _run(_block_sim(slice(6, None)), **kw)]
    joint = FisherMatrix.joint(blocks, verbose=False).compute(debias_derivatives=True)
    np.testing.assert_allclose(joint.C, full.C)
    np.testing.assert_allclose(joint.F, full.F)


def test_joint_pairs_by_key_and_mixes_noiseless_blocks():
    '''Rows are paired by (seed, noise_seed), not by order; a noiseless block adds no noise scatter.'''
    a = _run(_block_sim(slice(0, 6)), noise_seeds=[1, -1, 2, -2])
    b = _run(_block_sim(slice(6, None)), noise_seed=0)
    order = np.random.default_rng(0).permutation(len(b.cov_keys))
    b.cov_keys = [b.cov_keys[i] for i in order]
    b.x_realisations = b.x_realisations[order]
    joint = FisherMatrix.joint([a, b], verbose=False)
    np.testing.assert_allclose(joint.x_realisations[:, :6], a.x_realisations)
    assert joint.dmu_samples_ns.shape[2] == 15
    assert np.allclose(joint.dmu_samples_ns[:, :, 6:], joint.dmu_samples_ns[0, :, 6:])
    joint.compute(debias_derivatives=True)


def _sliced_sim(theta, seed, noise_seed, sl):
    return linear_sim(theta, seed, noise_seed)[sl]


def test_fisher_blocks_with_cache_files(tmp_path):
    '''Per-block caches joined back give the full Fisher matrix; a rerun reads only the caches.'''
    kw = dict(covariance_kw=dict(seeds=range(1, 9), noise_seeds=range(1, 6)),
              derivative_kw=dict(delta_frac=0.01, seeds=range(1, 4), noise_seeds=[1, -1, 2, -2]))
    blocks = {'lo': dict(sl=slice(0, 6)), 'hi': dict(sl=slice(6, None))}
    cache = lambda name: str(tmp_path / f'{name}.pkl')
    fms = fisher_blocks(_sliced_sim, [1.0, 2.0], ['a', 'b'], blocks, cache_file=cache, **kw)
    full = _run(linear_sim, noise_seeds=[1, -1, 2, -2]).compute(debias_derivatives=True)
    joint = FisherMatrix.joint(list(fms.values()), verbose=False).compute(debias_derivatives=True)
    np.testing.assert_allclose(joint.F, full.F)
    failing = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('simulated although cached'))
    fisher_blocks(failing, [1.0, 2.0], ['a', 'b'], blocks, cache_file=cache, **kw)


def test_chunked_parallel_matches_unchunked(tmp_path):
    pytest.importorskip("joblib")
    out = {}
    for n_chunks in (1, 3):
        fm = _fm(cache_file=str(tmp_path / f"c{n_chunks}.pkl"))
        fm.estimate_covariance(seeds=range(1, 9), noise_seeds=range(1, 6), n_jobs=2, n_chunks=n_chunks)
        fm.estimate_derivatives(delta_frac=0.01, seeds=range(1, 5), noise_seeds=[1, 2], n_stencil=4,
                                n_jobs=2, n_chunks=n_chunks)
        out[n_chunks] = fm.compute()
    np.testing.assert_allclose(out[3].F, out[1].F)
    assert len(out[3]._cache) == len(out[1]._cache)
