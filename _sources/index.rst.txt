.. image:: logo.png
   :alt: psiphy logo
   :width: 500px
   :align: center

|

psiphy — Package for Statistical Inference of Physics
======================================================

A Python package for cosmological parameter inference, providing tools for
simulation-based inference (SBI), MCMC sampling, Fisher forecasting, and
posterior diagnostics.

=========================  ==============================================================
Module                     Contents
=========================  ==============================================================
``psiphy.mcmc``            ``MHSampler`` (emcee), ``NestedSampler`` (nautilus or dynesty,
                           with Bayesian evidence), importance sampling,
                           ``sequential_monte_carlo``
``psiphy.utils.prior``     Uniform, log-uniform, Gaussian and mixture priors with one
                           interface for all samplers
``psiphy.sbi``             Likelihood-free inference: ``BOLFI``, ``LFIRE``, rejection ABC
``psiphy.forecasting``     ``FisherMatrix``: numerical Fisher forecasts
``psiphy.plotting``        Triangle plots, credible intervals, posterior diagnostics
``psiphy.toy_models``      Simulators with known answers for testing
=========================  ==============================================================

.. code-block:: bash

   pip install psiphy

New here? Start with :doc:`installation` and :doc:`quickstart`, then follow the
:doc:`tutorials`.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/utils
   api/sbi
   api/mcmc
   api/forecasting
   api/plotting
   api/toy_models

.. toctree::
   :maxdepth: 1
   :caption: Project

   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
