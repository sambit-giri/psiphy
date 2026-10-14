Installation
============

psiphy requires Python 3.9 or newer.

From PyPI
---------

.. code-block:: bash

   pip install psiphy

The core install covers priors, importance sampling and sequential Monte Carlo,
likelihood-free inference (BOLFI, LFIRE, rejection ABC), Fisher forecasting and
the plotting tools.

Optional dependencies
---------------------

Some samplers rely on external packages, grouped as extras:

=========  ==========================================  =======================================
Extra      Installs                                    Needed for
=========  ==========================================  =======================================
``mcmc``   emcee, dynesty, nautilus-sampler            ``MHSampler``, ``NestedSampler``
``sbi``    sbi, torch                                  neural simulation-based inference
``hmc``    jax, jaxlib, blackjax                       Hamiltonian Monte Carlo (in development)
``docs``   sphinx and extensions                       building this documentation
``dev``    all of the above, pytest, ruff              development
=========  ==========================================  =======================================

.. code-block:: bash

   pip install "psiphy[mcmc]"          # one extra
   pip install "psiphy[mcmc,sbi]"      # several

``MHSampler`` can store chains in HDF5 files (``filename=``), which needs ``h5py``.

Development version
-------------------

The latest version from GitHub:

.. code-block:: bash

   pip install git+https://github.com/sambit-giri/psiphy.git

For development, clone the repository and install it in editable mode with all
extras:

.. code-block:: bash

   git clone https://github.com/sambit-giri/psiphy.git
   cd psiphy
   pip install -e ".[dev]"

Check the installation:

.. code-block:: python

   import psiphy
   print(psiphy.__version__)
