Contributing and support
========================

Questions, bug reports and feature requests are welcome on the
`GitHub issue tracker <https://github.com/sambit-giri/psiphy/issues>`_.
Pull requests are encouraged.

Development setup
-----------------

.. code-block:: bash

   git clone https://github.com/sambit-giri/psiphy.git
   cd psiphy
   pip install -e ".[dev]"

Running the tests
-----------------

.. code-block:: bash

   pytest

Tests check samplers against analytic posteriors and evidences where these are
known. New code should come with tests.

Building the documentation
--------------------------

The tutorials are Jupyter notebooks in ``examples/``. They are rendered as they
are saved (not executed during the build), so run a notebook before committing
changes to it. Then build the HTML pages:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

and open ``docs/_build/html/index.html``. Building requires ``pandoc``.
