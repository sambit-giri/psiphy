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
