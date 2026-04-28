<img src="docs/logo_cropped.png" alt="psiphy logo" width="220" align="right"/>

# psiphy — Package for Statistical Inference of Physics

[![License](https://img.shields.io/github/license/sambit-giri/psiphy.svg)](https://github.com/sambit-giri/psiphy/blob/master/LICENSE)
[![GitHub Repository](https://img.shields.io/github/repo-size/sambit-giri/psiphy)](https://github.com/sambit-giri/psiphy)
[![CI status](https://github.com/sambit-giri/psiphy/actions/workflows/ci.yml/badge.svg)](https://github.com/sambit-giri/psiphy/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/Documentation-here-blue)](https://sambit-giri.github.io/psiphy)

A Python package for cosmological parameter inference, providing tools for
simulation-based inference (SBI), MCMC sampling, Fisher forecasting, and
posterior diagnostics.

## Installation

Install directly from GitHub:

    pip install git+https://github.com/sambit-giri/psiphy.git

To include optional dependencies (e.g. MCMC samplers):

    pip install "git+https://github.com/sambit-giri/psiphy.git#egg=psiphy[mcmc]"

For a local editable install (recommended for development):

    git clone https://github.com/sambit-giri/psiphy.git
    cd psiphy
    pip install -e ".[dev]"

## Running the tests

The test suite uses [pytest](https://pytest.org). After installing with the `dev` extras:

    pytest tests/ -v

Tests cover package imports, toy model simulators, and sampling utilities. New tests are added alongside each new module.

## Contributing and feedback

Bug reports, feature requests, and questions are welcome — please open an issue on the [GitHub issue tracker](https://github.com/sambit-giri/psiphy/issues). Pull requests are also encouraged.
