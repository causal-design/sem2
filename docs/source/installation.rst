Installation
============

Use Python 3.11 or later. From a checkout of the repository, install the
package in editable mode:

.. code-block:: console

   python -m pip install -e .

Import classes and functions from their defining modules, rather than from
package-level shortcuts:

.. code-block:: python

   from sem2.model.models.base import SEM2
   from sem2.model.models.discrete import SEM2Discrete
   from sem2.model.models.linear import SEM2Linear
   from sem2.trainer.trainer import fit

Build the documentation
-----------------------

From the repository root:

.. code-block:: console

   python -m pip install -e '.[docs]'
   python -m sphinx -M html docs/source docs/_build -W --keep-going

Open ``docs/_build/html/index.html``. Documentation builds import the real
package; its runtime dependencies must be installed.
