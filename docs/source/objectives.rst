Objectives and composition
==========================

Energy discrepancies
--------------------

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
