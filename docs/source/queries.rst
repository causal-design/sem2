Moments and interventions
=========================

Population queries
------------------

``MeanPotentialOutcome`` averages a named scalar outcome under a fixed
intervention. ``ATE`` subtracts the reference-regime mean from the
alternative-regime mean. Neither operator fits a model, changes its parameters,
or requires ``TrainingConfig``. They can evaluate an already fitted model or a
known model whose parameters were supplied directly.

An intervention dictionary replaces the named structural equations with fixed
stored values. Omitted variables follow their equations, including regeneration
of downstream variables. ``{}`` denotes the natural regime; it is not a
zero-valued intervention. Each regime is interpreted independently: assignments
in the alternative do not fill omissions in the reference. Joint and nonbinary
interventions use the same API.

This executable example uses known coefficients:

.. literalinclude:: ../../experiments/causal_queries.py
   :language: python

The three values are 12, 12, and 3. Fixing ``mediator`` changes the joint regime;
leaving it unspecified regenerates it from ``exposure``.

Evaluation and scoring
----------------------

* ``sample_bs`` defaults to 512 and controls Monte Carlo draws, not data batches.
  Constructor settings can be overridden for one ``evaluate`` call without
  changing later calls.
* ``seed=None`` continues the current random stream unless the constructor has
  an explicit seed. An integer resets the stream at evaluation entry without
  saving or restoring RNG state. A contrast seeds once and evaluates reference
  then alternative using sequential draws; paired noise is not guaranteed.
* Evaluation temporarily uses ``model.eval()`` to disable dropout. This is **not**
  ``torch.inference_mode()`` or ``torch.no_grad()``: gradients remain available.
  Every module's previous train/eval flag is restored, including after errors.
  The model's cached node values and RNG state may change.
* Continuous means use target-only sampling. Discrete means weight outcome states
  by their probabilities, then average over exogenous draws. State enumeration
  does not make the remaining Monte Carlo integration exact. Exogenous outcomes
  are averaged from their draws rather than selected from the endogenous table.
* Categorical means require explicit finite scores covering every declared
  state. Without scores, construction warns and raises ``ValueError``; integer
  category codes are never silently treated as quantities. For a probability,
  supply a complete zero/one event-indicator mapping, for example
  ``VariableInfo("Y", numeric_mapping={0: 0., 1: 0., 2: 1.})`` for the probability
  of state 2 in a three-state outcome. Label-keyed mappings may also supply an
  ``encoding`` from labels to stored state IDs. Scores do not recode interventions.
* Outcomes must be scalar user-declared nodes. Fixed assignments are checked
  against declared support at construction, before discrete state coercion.
  Unknown names, internal helper nodes and fractional discrete assignments fail
  before sampling. The model schema and query specification should stay fixed
  during the query's lifetime.

Both operators return ``(ScalarResult, details)``. ``result.value`` retains its
computation graph. Details record the outcome, fixed regime(s), method
(``monte_carlo`` or ``probability_weighted``), draw count, seed and evaluation mode.
Contrasts also report detached reference and alternative means. Existing
``SEM2Loss`` composition can use the differentiable scalar directly; evaluation
alone performs no optimizer step or partial-identification search.

These are **model-implied population quantities**, not identification checks,
confidence intervals, ATT estimates or individual counterfactuals. Valid causal
interpretation requires a defensible structural model and identification
argument; fitting observational data does not supply either automatically.

Low-level draws
---------------

``model.sample(batch_size=..., do_intervention=batch)`` accepts a ``DataBatch``
of already-valid stored values, with one row per draw and matching node widths,
on the model device. It does not broadcast fixed dictionaries, encode labels,
score outcomes or validate support. The public query constructors provide the
fixed-dictionary boundary above. ``do_intervention`` replaces the old
``set_nodes`` keyword; ``ATE`` now requires explicit ``outcome``, ``alternative``
and ``reference`` rather than names from training configuration.

API
---

.. autoclass:: sem2.operators.potential_outcomes.PotentialOutcomeSample
   :members: evaluate

.. autoclass:: sem2.operators.potential_outcomes.MeanPotentialOutcome
   :members: evaluate

.. autoclass:: sem2.operators.potential_outcomes.ATE
   :members: evaluate

Covariance
----------

.. autoclass:: sem2.operators.covariance.Covariance
   :members: evaluate
