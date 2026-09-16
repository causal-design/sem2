# 3. Minimum fitting interface and diagnostics

**Agreed trainer/diagnostics boundary for F15–F18/F25 in the [v0.1 plan](../v0.1.md).**
This handoff records the decisions from the fitting-interface discussion, including
its [decision note on PR #50](https://github.com/causal-design/sem2/pull/50#issuecomment-5633054241).
It is a specification for review and subsequent implementation, not a claim that
the proposed result reporting already works. It does not approve every
objective-specific proposal from the earlier audit.

## Agreed v0.1 boundary

- **The package owns the loop.** Users call `fit(...)`; one canonical trainer
  handles optimization. Consolidate the package and experimental trainers rather
  than add another fitting engine. No user-written loop or public `step` API is
  required. A private update function is an implementation detail.
- **Keep the agreed data interface.** Accept a pandas DataFrame and use the shared
  preparation/name-alignment rules in the [minimal model–data handoff](minimal-model-data-interface.md).
  Users do not supply a mapping. Removing hard-coded observed-variable names or
  column positions is implementation/verification work under that agreement, not
  an open interface choice. Treatment, outcome, and instrument roles remain in
  their separate query/objective work.
- **Use a configurable, prespecified epoch budget.** Ordinary fitting trains one
  model on the supplied batches. Built-in cross-fitting, a validation-data loop,
  and convergence-based early stopping are outside v0.1. Do not silently split
  the user's data. Completing the budget is not proof of convergence.
- **Keep objective handling separate from optimization.** The trainer runs the
  selected objective; family-specific loss definitions, reference cases, and
  tolerances remain in their respective review work. This handoff does not
  introduce a new objective hierarchy or settle the remaining mathematical choices.

The ordinary user workflow is: supply an initialized model and DataFrame, select
an objective and epoch budget, call `fit` once, then optionally inspect or plot the
returned history before sampling or evaluating queries. Fitting does not require
users to build folds, write an update loop, or interpret gradient diagnostics.

## Required diagnostics, optional display

The fit result retains the fitted model and preparation report described by the
model–data handoff, with a small inspectable training record:

- **Training-loss history**, with documented normalization and iteration/epoch
  indexing. Keep history values detached from the computation graph.
- **A final loss evaluation after the last update**, distinguished from averages
  of online batch losses evaluated at changing parameter values.
- **Completed-update count and termination reason.** Report budget completion
  honestly rather than returning `converged=True` merely because the loop ended.
- **A scalar gradient-norm history for debugging.** Use the raw L2 norm after
  backward and before updating/clipping. Keep it accessible to developers and
  interested users, but out of the default user-facing plot.
- **Separate data-fit loss for query-specific optimization**, rather than showing
  only a combined objective whose decrease could conceal worsening data fit.

Invalid data and nonfinite loss/gradients must fail clearly rather than produce
an apparently successful fit. These checks are correctness guards, not an
automatic convergence or recovery framework.

Provide a simple optional training-loss plot. No mandatory plotting, dashboard,
or rich result-object hierarchy is required. Loss/gradient traces can reveal
optimization problems but do not certify generalization, identification, causal
validity, or a global optimum. Simulation-based objectives can produce noisy traces.

## Possible future extensions — unscheduled

These are candidates for later review, **not v0.1 requirements or commitments to
a particular release, deadline, or implementation**:

1. **Validation-loss monitoring:** explicitly supplied held-out data, a documented
   evaluation frequency, and comparable loss normalization/simulation settings.
   No silent train/validation split.
2. **Automatic stopping and best-iterate restoration:** select a monitoring
   criterion, tolerances/patience, and a policy for the returned parameter state.
   Those choices require future design review; none is selected here.
3. **Cross-fitting:** fold-wise fits and out-of-fold evaluation where a specific
   estimator needs them. Cross-fitting is distinct from mini-batching and from a
   single train/validation split; it is not required for every SEM fit.
4. **Public `step`/`evaluate` primitives:** reconsider only when an advanced
   custom-loop use case justifies the additional supported API.

The broader trainer framework features already deferred by the parent plan remain
deferred; this note does not change other threads' query or initialization decisions.

## Implementation acceptance

Use a few focused checks, alongside the existing model/example acceptance cases:

1. `fit` and direct trainer use share preparation, objective handling, and one
   loop; matched settings give equivalent behavior. Reordered/renamed data follow
   the existing model–data contract rather than benchmark-specific names.
2. A finite training iterable completes the configured budget and reports the
   actual update count without a convergence claim.
3. Returned loss/component/gradient histories are detached and interpretable;
   the final evaluation uses the returned fitted parameters after the last update.
4. Invalid inputs and nonfinite loss/gradients fail clearly; ordinary fitting does
   not require an ATE metric, validation data, cross-fitting, or a public `step`.
5. The optional loss plot works with the returned record, while fitting also works
   without displaying a plot. Query-specific results expose the data-fit component.

These are future implementation checks, not tests executed by this documentation PR.

## Inspected implementation starting points

The [package trainer](../../../sem2/sem2trainer.py#L42-L80) and
[experimental duplicate](../../../experiments/trainer.py#L43-L89) at PR #50's head
(`ce7d205ccf953490b9451498bce0985e166a08a8`) both iterate over `range(cfg.epochs)`
and supplied batches. They log epoch losses but do not return the training record
specified above, split folds, run a validation loop, or apply convergence-based
stopping criteria. This is source inspection, not a new training run.
