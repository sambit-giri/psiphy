Quickstart
==========

Short examples of the main tools. Each runs in a few seconds; the
:doc:`tutorials` explain the methods and their options in detail.

Define a prior
--------------

All samplers take the same prior objects. Named bounds give parameter names to
the outputs.

.. code-block:: python

   from psiphy.utils.prior import UniformPrior, GaussianPrior

   prior = UniformPrior({'slope': (-3, 3), 'intercept': (0, 10)})
   prior.sample(5)                  # (5, 2) array of draws
   prior.log_prob([0.5, 4.0])       # log density

MCMC
----

Sample a posterior with an ensemble MCMC sampler (requires the ``mcmc`` extra).

.. code-block:: python

   from psiphy.mcmc import MHSampler
   from psiphy.toy_models import NoisyLine

   line = NoisyLine()
   y_obs = line.simulate([-0.8, 5.0])

   def log_posterior(theta):
       return prior.log_prob(theta) + line.log_prob(theta, y_obs)

   sampler = MHSampler(log_posterior, n_walkers=32, param_names=prior.names)
   sampler.run(prior.sample(32), n_steps=2000)
   samples = sampler.get_chain(flat=True, discard='auto')
   sampler.summary()

Nested sampling and evidence
----------------------------

Nested sampling returns the Bayesian evidence for model comparison together
with posterior samples (requires the ``mcmc`` extra).

.. code-block:: python

   from psiphy.mcmc import NestedSampler

   ns = NestedSampler(lambda t: line.log_prob(t, y_obs), prior, backend='nautilus')
   result = ns.run().get_result()
   result['log_evidence'], result['samples'], result['weights']

Likelihood-free inference
-------------------------

When only a simulator is available, BOLFI learns the posterior from a small
number of simulations. ``distance`` simulates at ``theta`` and compares summary
statistics of the simulated and observed data — here the least-squares slope and
intercept, scaled by their expected scatter, so that ``exp(-distance / 2)``
behaves like a likelihood.

.. code-block:: python

   import numpy as np
   from psiphy.sbi.bolfi import BOLFI

   A = np.column_stack([line.x, np.ones_like(line.x)])
   fit = lambda y: np.linalg.lstsq(A, y, rcond=None)[0]       # (slope, intercept)
   inv_cov = A.T @ A / line.noise_sigma**2                      # inverse covariance of the fit
   p_obs = fit(y_obs)

   def distance(theta):
       dp = fit(line.simulate(theta)) - p_obs
       return dp @ inv_cov @ dp

   bolfi = BOLFI(distance, prior, verbose=False)
   bolfi.learn_likelihood(n_calls=60)
   bolfi_samples = bolfi.sample_posterior(2000)

Fisher forecast
---------------

Forecast parameter constraints from a simulator with signature
``simulator(theta, seed, noise_seed)``.

.. code-block:: python

   from psiphy.forecasting.fisher import FisherMatrix

   def simulator(theta, seed, noise_seed):
       rng = np.random.default_rng(1000 * seed + noise_seed)
       noise = rng.normal(0, line.noise_sigma, line.x.size) if noise_seed else 0.0
       return theta[0] * line.x + theta[1] + noise

   fm = FisherMatrix(simulator, theta_fid=[-0.8, 5.0], param_names=['slope', 'intercept'],
                     verbose=False)
   fm.estimate_covariance(seeds=range(1, 21), noise_seeds=range(1, 11))
   fm.estimate_derivatives(delta_frac=0.01, seeds=range(1, 3), noise_seed=0)
   fm.compute()
   fm.sigma_1d()                    # marginalised 1-sigma forecasts

Plot and summarise
------------------

.. code-block:: python

   from psiphy.plotting.diagnostics import plot_triangle, print_chain_stats

   plot_triangle(samples, labels=['m', 'b'], true_values=[-0.8, 5.0], engine='getdist')
   print_chain_stats(samples, labels=['m', 'b'])

``engine='corner'`` (the default) needs the ``corner`` package.
