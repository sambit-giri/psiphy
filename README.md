# psiphy — Package for Statistical Inference of Physics

<img src="docs/logo_cropped.png" alt="psiphy logo" width="120" align="right"/>

[![License](https://img.shields.io/github/license/sambit-giri/psiphy.svg)](https://github.com/sambit-giri/psiphy/blob/master/LICENSE)
[![GitHub Repository](https://img.shields.io/github/repo-size/sambit-giri/psiphy)](https://github.com/sambit-giri/psiphy)
[![CI status](https://github.com/sambit-giri/psiphy/actions/workflows/ci.yml/badge.svg)](https://github.com/sambit-giri/psiphy/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/Documentation-here-blue)](https://sambit-giri.github.io/psiphy)

A Python package for cosmological parameter inference, providing tools for
simulation-based inference (SBI), MCMC sampling, Fisher forecasting, and
posterior diagnostics.

## What's inside

| Module | Contents |
|---|---|
| `psiphy.mcmc` | `MHSampler` (emcee), `NestedSampler` (nautilus or dynesty, with Bayesian evidence), importance sampling and `sequential_monte_carlo` (likelihood tempering, evidence estimate) |
| `psiphy.utils.prior` | Uniform, log-uniform, Gaussian and mixture priors with one interface for all samplers (`sample`, `log_prob`, `prior_transform`) |
| `psiphy.sbi` | Likelihood-free inference: `BOLFI` (Bayesian optimisation), `LFIRE` (ratio estimation), rejection ABC |
| `psiphy.forecasting` | `FisherMatrix`: numerical Fisher forecasts with several estimators and parallel simulations |
| `psiphy.plotting` | Posterior diagnostics: triangle plots, credible intervals, z-scores, comparisons between posteriors |
| `psiphy.toy_models` | Simulators with known answers for testing and tutorials |

Quick example:

```python
from psiphy.mcmc import MHSampler
from psiphy.utils.prior import UniformPrior
from psiphy.toy_models import NoisyLine

line = NoisyLine()
y_obs = line.simulate([-0.8, 5.0])
prior = UniformPrior({'slope': (-3, 3), 'intercept': (0, 10)})

sampler = MHSampler(lambda t: prior.log_prob(t) + line.log_prob(t, y_obs), n_walkers=32,
                    param_names=prior.names)
sampler.run(prior.sample(32), n_steps=2000)
sampler.summary()
```

## Tutorials

Notebooks in [`examples/`](examples/), also rendered in the [documentation](https://sambit-giri.github.io/psiphy/tutorials.html):

* [MCMC sampling and priors](examples/mcmc_tutorial.ipynb)
* [Nested sampling and Bayesian evidence](examples/nested_sampling_tutorial.ipynb)
* [Importance sampling and SMC](examples/importance_sampling_tutorial.ipynb) (includes a calibration check over many data sets)
* [Likelihood-free inference: rejection ABC, BOLFI, LFIRE](examples/lfi_tutorial.ipynb)
* [Fisher forecasting](examples/fisher_analysis.ipynb)
* [Plotting diagnostics](examples/plotting_diagnostics_tutorial.ipynb)
* [Toy models](examples/toy_models_overview.ipynb)

## Installation

From PyPI:

    pip install psiphy

Optional dependencies are grouped as extras: `mcmc` (emcee, dynesty, nautilus), `sbi`
(sbi, torch), `hmc` (JAX/blackjax, for the planned HMC sampler):

    pip install "psiphy[mcmc]"

The latest development version from GitHub:

    pip install git+https://github.com/sambit-giri/psiphy.git

For a local editable install (recommended for development):

    git clone https://github.com/sambit-giri/psiphy.git
    cd psiphy
    pip install -e ".[dev]"

## Running the tests

The test suite uses [pytest](https://pytest.org). After installing with the `dev` extras:

    pytest tests/ -v

Tests cover every module, checking samplers against analytic posteriors and evidences where these are known. New tests are added alongside each new module.

## Contributing and feedback

Bug reports, feature requests, and questions are welcome — please open an issue on the [GitHub issue tracker](https://github.com/sambit-giri/psiphy/issues). Pull requests are also encouraged.
