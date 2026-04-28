"""
psiphy — Package for Statistical Inference of Physics

A Python package for cosmological parameter inference, providing tools for
simulation-based inference (SBI), MCMC sampling, Fisher forecasting, and
posterior diagnostics.

Submodules
----------
psiphy.sbi        : Likelihood-free / simulation-based inference (BOLFI, LFIRE, ABC, SRE)
psiphy.mcmc       : MCMC samplers (MH/emcee, nested sampling, HMC)
psiphy.forecasting: Fisher matrix forecasting
psiphy.plotting   : Posterior corner plots and diagnostics
psiphy.toy_models : Example simulators for testing and benchmarking
psiphy.utils      : Shared utilities (distances, KDE, sampling, helpers)
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("psiphy")
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
