# psiphy — Package for Statistical Inference of Physics

<p align="center">
  <img src="logo.png" alt="psiphy logo" width="500"/>
</p>

[![License](https://img.shields.io/github/license/sambit-giri/PSI.svg)](https://github.com/sambit-giri/PSI/blob/master/LICENSE)
[![GitHub Repository](https://img.shields.io/github/repo-size/sambit-giri/PSI)](https://github.com/sambit-giri/PSI)
[![CI status](https://github.com/sambit-giri/PSI/actions/workflows/ci.yml/badge.svg)](https://github.com/sambit-giri/PSI/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/Documentation-here-blue)](https://sambit-giri.github.io/PSI)

A Python package for cosmological parameter inference, providing tools for
simulation-based inference (SBI), MCMC sampling, Fisher forecasting, and
posterior diagnostics.

## Installation

Install directly from GitHub:

    pip install git+https://github.com/sambit-giri/PSI.git

To include optional dependencies (e.g. MCMC samplers):

    pip install "git+https://github.com/sambit-giri/PSI.git#egg=psiphy[mcmc]"

For a local editable install (recommended for development):

    git clone https://github.com/sambit-giri/PSI.git
    cd PSI
    pip install -e ".[dev]"
