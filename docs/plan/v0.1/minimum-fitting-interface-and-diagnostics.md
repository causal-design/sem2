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

## Questions for review

The agreed requirements above remain unchanged. The first two categories concern
v0.1; only the third is deferred beyond it. Questions identify choices, not
approved defaults or requirements to make every option configurable.

### 1. Objective decisions needed for v0.1

These belong to the separate objective review, not a later release; see items
4–6 of the [parent meeting agenda](../v0.1.md#next-steps-and-review-decisions).

- **Which continuous energy estimator should we use?** Specify the distance and
  its exponent, variable scaling, and overall normalization. Decide whether to
  compare all observation–simulation pairs or paired draws, and how to handle
  within-sample self-pairs; observation and simulation counts may differ.
- **How should finite-discrete fitting compare categories?** Specify the
  discrepancy, normalization, and distance for categories without numerical
  scores, including how variable scales enter. State when relabeling should
  preserve the fit; preserving category meaning alone does not guarantee this,
  and categorical likelihood is not a new prerequisite.
- **How should IV fitting weight instrument levels and form valid batches?**
  Decide whether conditional discrepancies use observed-frequency weights and
  specify their normalization relative to the signed estimand penalty. Full-data
  eligibility remains at least two retained observations per declared instrument level;
  choose full-data evaluation or batches that also make the estimator valid.

**Recommendation for the meeting:** settle these mathematical choices and the IV
batching rule before implementation, while assigning routine trainer details below
to implementation review.

### 2. Trainer details to settle during implementation review

These details are unspecified, not explicitly postponed beyond v0.1. A small
proposal can resolve them; bring back choices that change statistical meaning or
agreed user-visible behavior rather than treating every detail as a meeting blocker.

- **Should Gaussian fitting minimize negative mean log-likelihood?** Confirm the
  sign and averaging over retained observations for the observed-data likelihood.
  The agreed intercept and observed-variable rules remain unchanged.
- **How does the caller select the objective?** Choose a name or supplied object
  and document the accepted form. `fit` and direct trainer use must share it;
  this does not require a new objective hierarchy.
- **What state changes during and between fits?** State whether fitting mutates
  the supplied model and whether another call resets or retains optimizer state.
  Preserve the sweep's agreed fresh-start default and explicit warm-start option.
- **How are randomness and sample counts controlled?** Document fitting/simulation
  seeds, observation batch size, and simulation count. The caller supplies an
  initialized model; distinguish its initialization from randomness during fitting,
  and apply the objective-specific IV batching rule above.
- **How often are histories recorded?** Choose per-update, per-epoch, or configurable
  recording, with clear indexing and aggregation. Recording loss and raw L2
  gradient-norm histories is already required; their inclusion is not open.
- **What data and draws feed the final evaluation?** Specify the retained
  observations and simulation draws used with the final fitted parameters.
  State when the required IV counts, conditional-table total variation, and
  largest cell-probability error are computed, separately for lower/upper searches.
- **How does a failed fit reach the caller?** Choose an exception or explicitly
  failed result, and specify how settings and available partial history are retained.
  A sweep must retain failure information without treating the run as successful;
  preserve the existing [sweep records and endpoint policy](identification-and-partial-identification.md#fitting-and-extrapolation-agreed-workflow).

### 3. Questions deferred beyond v0.1

These concern the [unscheduled extensions above](#possible-future-extensions--unscheduled),
not missing v0.1 decisions. No release or implementation is assigned to them.

- **How would held-out validation be monitored?** Later review would choose the
  supplied validation data, evaluation frequency, and comparable loss settings.
- **When would training stop automatically, and which iterate would be returned?**
  Monitoring criteria, patience/tolerances, and best-iterate restoration are deferred.
- **Which estimators need cross-fitting?** Fold construction and out-of-fold
  evaluation await a specific estimator; ordinary minibatching is not cross-fitting.
- **Is a public custom-loop API needed?** Reconsider trainer `step`/`evaluate`
  primitives only for a justified advanced use case; required final loss evaluation
  and the agreed query/operator `.evaluate()` interface are not deferred.
