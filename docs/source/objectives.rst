Objectives and composition
==========================

Energy discrepancies
--------------------

Energy losses use Euclidean distances on quantitative values or explicitly scored
categories. Supply ``VariableInfo.numeric_mapping`` for categorical variables;
integer category codes alone do not define a distance.

Set defaults in ``TrainingConfig``::

    cfg = TrainingConfig(
        sample_bs=512,
        energy_score_beta=1.0,
        energy_score_unbiased=True,
    )
    objective = EnergyLoss(cfg, model, beta=0.5)

Constructor arguments override the config. ``evaluate(..., sample_bs=..., beta=...)``
temporarily overrides the draw count and exponent without changing stored settings.
``unbiased=False`` at construction includes same-draw pairs instead of applying
the finite-sample correction. The continuous loss is an energy score; the discrete
loss also subtracts the data/data term and uses half-energy-distance scaling.

``DiscreteEnergyLoss(cfg, states)`` precomputes its distance matrix at construction
from the named state support and numerical mappings. ``EnergyLoss`` and
``DiscreteIVEnergyLoss`` construct it when the observed variables and scores are
first available, then reuse it across batches and instrument groups. Changing
the observed variables or scores rebuilds it. A temporary exponent override
reuses the underlying Euclidean distances without changing the default matrix.

For IV fitting, set ``batch_size`` at least as large as the dataset. Eligibility
is checked once before training, after incomplete rows are removed: every declared
instrument level needs at least two observations. Conditional discrepancies use
empirical group frequencies. The instrument itself does not need numerical scores
because it selects groups rather than entering their distance calculation.

Energy exponents are checked at construction and for explicit overrides, rather
than for each distance calculation. Batch-size checks protect pair denominators;
they inspect tensor shapes, not device values.

.. autoclass:: sem2.operators.divergence.EnergyLoss
   :members: evaluate

.. autoclass:: sem2.operators.discrete_energy_loss.DiscreteEnergyLoss
   :members: energy_score, energy_dist

.. autoclass:: sem2.operators.discrete_iv.DiscreteIVEnergyLoss
   :members: evaluate

Covariance discrepancy
----------------------

.. autoclass:: sem2.operators.divergence.WeightedFrobeniusDistance
   :members: evaluate

Weighted objectives
-------------------

.. autoclass:: sem2.operators.sem2_loss.SEM2Loss
   :members: evaluate
