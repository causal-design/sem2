PoliticalDemocracy
==================

Fit the bundled PoliticalDemocracy data using the analytical Gaussian
likelihood and the package's :func:`~sem2.trainer.trainer.fit` interface.
From an installed source checkout, run:

.. code-block:: console

   python -m experiments.political_democracy --epochs 2000 --seed 0

The example centers the observed indicators, fixes intercepts at zero and
performs one full-data Adam update per epoch. It reports the three structural
coefficients on lavaan's unit-first-loading latent scales, their differences
from the rounded `lavaan ML reference <https://lavaan.ugent.be/tutorial/sem.html>`_,
and final full-data likelihood, gradient and covariance diagnostics.

This is an experiment, not a pytest recovery check. Completing the update budget
does not establish convergence, and the reference estimates are not known causal
effects. Shared residual noises impose a particular covariance parameterization;
coefficient agreement alone does not establish equivalence to lavaan's model.

The :download:`worked notebook <../linear_sem.ipynb>` explains the model,
centering, latent-scale conversion and covariance comparisons. Open it in the
checkout's ``docs/`` directory with a kernel using the installed SEM2 environment.
The :download:`Python example <../../experiments/political_democracy.py>` contains
the fitting and reporting code; the notebook calls the same functions.
