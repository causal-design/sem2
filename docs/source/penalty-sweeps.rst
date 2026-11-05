Penalty sweeps
==============

``sem2.analysis`` provides higher-level analyses using the existing models,
operators and trainer. ``penalty_sweep`` fits one fresh model per supplied weight,
minimizing ``fit_loss + weight * target``. Positive weights favor smaller target
values; negative weights favor larger ones. Zero requests an unpenalized fit.

Supply a factory returning a fresh model, its data-taking scalar fit loss, and
its model-only scalar target. Both operators must refer to that model. The fit
loss must already follow a minimization convention; negate log-likelihood.

.. code-block:: python

   import pandas as pd

   from sem2.analysis.penalty_sweep import penalty_sweep
   from sem2.default_configs import TrainingConfig
   from sem2.model.models.linear import SEM2Linear
   from sem2.operators.linear import LogLikelihood
   from sem2.operators.potential_outcomes import MeanPotentialOutcome
   from sem2.operators.sem2_loss import SEM2Loss

   def factory():
       model = SEM2Linear("u ~ Normal(0,1); y <- 1*u", device="cpu")
       loss = SEM2Loss([(LogLikelihood(model), -1.)])
       # Fixing zero-mean noise at zero makes the target exactly the intercept.
       target = MeanPotentialOutcome(model, outcome="y", intervention={"u": 0.})
       return model, loss, target

   config = TrainingConfig(device="cpu", optim="SGD", momentum=0., lr=0.5,
                           epochs=20, batch_size=3, sample_bs=1)
   results = penalty_sweep(factory, pd.DataFrame({"y": [-1., 0., 1.]}),
                           weights=[0.1, -0.1], config=config)
   for result in results:
       print(result.weight, result.fit_loss, result.target, result.max_abs_gradient)

Results and interpretation
--------------------------

Each ``SweepResult`` retains the fitted model in eval mode, the weight, final
fit loss, target, combined objective, maximum absolute parameter gradient, and
component-prefixed diagnostics. The diagnostics also include the complete-case
data preparation report. All reported values are evaluated after the last update
on the complete retained data, not taken from the trainer's last minibatch log.
The combined objective and its component values come from the same evaluation.

Final evaluation uses ``config.sample_bs``; training uses the operators' own
sampling settings. Construct those operators consistently with the configuration
when the same sample count is wanted throughout. Gradients exclude optimizer
weight decay. Small gradients do not prove global optimality, and Monte Carlo
components make diagnostics noisy.

Weights retain their input order. No warm starts, automatic weight selection,
restart selection, extrapolation, or bound certification is performed. Repeat
calls with fresh factories to assess initialization sensitivity. Factories control
initialization; ``config.rng_seed`` retains its existing training-only meaning.
Without explicit seeds, the shared random stream is not reset. Returned models
remain in memory; failures propagate rather than being silently omitted.

Binary instrumental-variable example
------------------------------------

From a source checkout, install the example dependency and run:

.. code-block:: bash

   python -m pip install -e '.[examples]'
   python -m experiments.discrete_iv_bounds

The example assumes randomized binary instrument ``Z``, binary treatment ``A``
and outcome ``Y``, and exclusion of ``Z`` from the outcome equation. It makes no
monotonicity assumption. Its latent variable enumerates all response types
``(A(0), A(1), Y(0), Y(1))``; fixed response tables implement these definitions,
and only their mixing probabilities are learned. No generating probabilities
are supplied to the fitted models. Enumeration makes the objective and target
exact for a given model, without latent Monte Carlo integration.

An independent NumPy/SciPy linear program obtains the sharp ATE identification
interval under those assumptions, subject to solver tolerance. Separate intervals
are calculated for the population table and the empirical conditional table.
An infeasible empirical table raises an error, rather than being silently
projected onto the IV model. These are not confidence intervals.

The default run uses 2,048 simulated observations, 1,500 updates per fresh model,
and weights ``[0.03, 0.003, -0.03, -0.003]``. It prints the reference intervals,
fit loss, total variation, maximum cell error, ATE, objective, final gradient and
signed difference from the corresponding empirical reference endpoint. Nonzero
penalties permit observational misfit, so estimates can lie outside the exact-fit
reference interval. This comparison checks the numerical workflow; it does not
turn arbitrary penalty-sweep extrema into certified bounds.

API
---

.. autoclass:: sem2.analysis.penalty_sweep.SweepResult
   :members:

.. autofunction:: sem2.analysis.penalty_sweep.penalty_sweep
