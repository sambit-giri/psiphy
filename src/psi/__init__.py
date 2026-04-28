"""
PSI — Parameter and Statistical Inference

A Python package for cosmological parameter inference, providing tools for
simulation-based inference (SBI), MCMC sampling, Fisher forecasting, and
posterior diagnostics.

Submodules
----------
psi.sbi        : Likelihood-free / simulation-based inference (BOLFI, LFIRE, ABC, SRE)
psi.mcmc       : MCMC samplers (MH/emcee, nested sampling, HMC)
psi.forecasting: Fisher matrix forecasting
psi.plotting   : Posterior corner plots and diagnostics
psi.toy_models : Example simulators for testing and benchmarking
psi.utils      : Shared utilities (distances, KDE, sampling, helpers)
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("psi")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

import numpy
numpy.seterr(all='ignore')

from . import utils
from . import sbi
from . import mcmc
from . import forecasting
from . import plotting
from . import toy_models
