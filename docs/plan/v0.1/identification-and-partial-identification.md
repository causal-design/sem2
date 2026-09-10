# 5. Identification checks and partial-identification results

Agreed F33–F35 specification for the [v0.1 plan](../v0.1.md), from this topic's
Slack discussion. This is a plan for implementation, not a claim that the package
already meets these requirements. It refines the earlier meeting decisions;
the historical records remain unchanged.

**Review and merge hold:** obtain approval from Slack reviewer `U044USKH8HH`
before merging this plan. This is a Slack ID, not a verified GitHub username.
Implementation and default-branch integration remain separate human decisions.

**Only unresolved topic: the detailed rules for automatic regression-window
selection.** Approach 2 (exhaustive contiguous-window selection) is selected for
v0.1. Its scoring, tolerance, ranking, and edge-case defaults await team discussion;
the [selector review](automatic-regression-selection.md) records the exact current
code behavior and a concrete recommended specification for that discussion.

The extrapolation code is now available and has been inspected. Integration,
fresh-start default with an explicit warm-start option, user-supplied penalties,
and the general warning/unavailable policy are settled. This update replaces the
earlier handoff's missing-code blocker; it does not reopen those decisions.[3]
Implementation is paused pending a fresh request after team discussion.

## Graphical identification and numerical searches

- Project the supplied DAG onto the observed variables as an ADMG and apply ID
  for one observational margin. Data fusion, latent-variable targets, and
  identification from additional parametric or proxy assumptions are outside
  this check's v0.1 scope.
- Return a Boolean for a supported identification problem. `True` means the
  checked target is identified; `False` means it is not identified within that
  graphical problem. Unsupported inputs must raise an explicit error, not return
  `False`. An identification formula is not required.
- Make clear which interventional distribution the check concerns. Identification
  of that distribution supports its means and contrasts; failure to identify a
  whole distribution is not automatically a proof that every scalar functional
  is nonidentified. Do not overstate the Boolean's scope.
- If ID succeeds, do not run a lower/upper search by default. Estimating the
  identified target can still require fitting. If ID fails, warn that the
  checked target is not point identified, may be partially identified, and that
  the user can explicitly request a query-specific numerical search.
- Keep ordinary query evaluation separate from training. A numerical search adds
  the chosen estimand, with the appropriate sign, to the fitting objective;
  changing an ordinary evaluation query must not secretly start a bound search.
  Nearly coincident numerical endpoints do not establish identification.

## Discrete IV assumptions, supports, and estimands

Use the standard discrete-IV model: an exogenous/randomized instrument,
exclusion of a direct instrument effect on the outcome, consistency, and
possible unobserved treatment–outcome confounding. Do not silently add
monotonicity/no-defiers, linearity, or restrictions on that confounding. The ATE
is the population contrast for the selected treatment values, not a complier
average effect. Interpretation is conditional on the supplied graph and these
assumptions; observational fit does not validate them.

Generalize the binary/example-specific restrictions throughout the IV workflow:
use named instrument, treatment, and outcome roles; declared finite supports;
and correctly aligned observed margins, marginalizing unobserved variables.
Do not assume names `Z`, `X`, `Y`, fixed column positions, or instrument levels
`0` and `1`. Support user-selected treatment contrasts such as level `2` versus
level `0`, with the requested direction. Preserve the distinction between an
IV-specific fitting objective and a general finite-state sampler.

Reject invalid states and intervention values before lossy integer conversion.
Match category identities through their encoding map rather than treating a
failed match as the first state.

### Categorical outcomes

Retain each variable's original categorical status, category labels, and encoding
map through internal numerical preparation. Any variable may later be selected
as the outcome. An automatic internal code does not supply meaningful numerical
outcome values.

For a numerical-value-dependent estimand such as ATE on an originally categorical
outcome without a supplied numerical interpretation, warn:

> ATE is not well-defined for a categorical outcome without a numerical
> interpretation. Choose a distributional estimand, such as a contrast in
> category probabilities, or explicitly convert the outcome to numerical values.

Do not silently compute an ATE from category codes. Distinguish automatic
encoding from a user's deliberate numerical conversion, and use meaningful
outcome values when computing means. A genuinely numerical outcome can have
finite support without triggering this warning. Encoding categorical treatments,
instruments, or covariates preserves fixed-level query meaning when category
identities are retained; this does not promise encoding-invariant numerical fits.

Coordinate this provenance requirement with the
[model–data handoff](minimal-model-data-interface.md) and F12/F28 conversion work.
This plan does not choose the conversion helper's exact API or mapping rules.

### Strict instrument-level eligibility

Require **at least two retained observations per declared instrument level**,
after complete-case filtering across the selected observed variables. If a level
is absent or has only one observation, stop before fitting and warn with the
affected levels, their counts, and the required minimum.

Do not silently drop levels, merge categories, or add pseudo-counts. Check the
retained dataset, not an individual minibatch; batching must separately avoid
invalid loss evaluations. This minimum makes the chosen loss usable, not the
bounds statistically reliable. It is an IV fitting rule, not an ID verdict.

## Reference checks and diagnostic reporting

Use independent finite-IV response-function linear-programming bounds in tests,
with Balke–Pearl bounds as a binary-IV reference. Match the numerical search and
reference to the same observed probability table, assumptions, treatment
contrast, and numerical outcome values. A population reference and the bounds
for a sampled empirical table are different comparisons.

**LP/reference solvers are test-only, not a runtime package bounds backend.**
Use a small collection of reference cases, including binary IV and a
ternary-treatment case with a selected contrast such as `2` versus `0`. Keep logical
correctness checks pass/fail, but do not prescribe a numerical bound-recovery
accuracy threshold in v0.1. Report endpoint errors, range width, any missed part
of the reference interval, and available simulation truth for manual judgment.
These are development diagnostics, not confidence-interval coverage guarantees.

Keep both agreed fit summaries, separately for the lower and upper search fits:

- Training diagnostics retain the observational discrepancy without the signed
  estimand term, alongside the overall objective and estimand history. Its scale
  depends on the encoding/distance; a negative unbiased discrepancy estimate
  does not mean better-than-perfect fit.
- The user-facing discrete summary reports each instrument level's retained
  count, total-variation distance between model and data conditional probability
  tables, and largest absolute cell-probability error. Total variation is half
  the sum of absolute cell differences. Compare the same observed margin and
  category mapping on both sides.

Use these summaries in synthetic tests too. Simulation truth and independently
known bounds are additional test information, unavailable with ordinary real
data. Neither a small fit discrepancy nor a narrow searched range certifies
sharpness, global optimality, an enclosing identified set, or sampling coverage.
Report lower and upper values as numerical search estimates under the stated
assumptions and settings, not automatically as certified causal bounds or
confidence intervals. Do not attach a fictitious fitted distribution to an
extrapolated endpoint. No automatic good-fit threshold is selected here.

## Settled extrapolation integration

The supplied `estimate_ate_bound` branch contains
`experiments/est_ate_bound.py`. Its current regression selector is an experimental
building block, not the native v0.1 interface or the selected replacement
algorithm.[1] Preserve the relevant computation while integrating it into the
package architecture; do not copy the CSV-driven demo into the public workflow.

### Two stages and native results

1. Run the explicitly requested query-specific training sweep through the
   canonical fitting machinery. For every supplied penalty, retain the signed
   penalty, ATE estimate, observational fitting discrepancy, total objective,
   initialization information, and available training/failure diagnostics.
2. Pass the resulting native, in-memory records to a separate postprocessor.
   It selects a regression window on each sign side and estimates the zero
   intercept. Postprocessing must not train models or generate additional points.

Use package-wide model/data/result conventions. CSV is an independent
serialization option, not the exchange format between these stages. There is no
requirement to support legacy result CSVs, fixed column layouts, or compatibility
adapters. Public symbol names and field spellings are ordinary implementation
choices within this contract, not additional unsettled product decisions.

### Penalties and initialization

- The user supplies all penalty points. Do not generate a grid, extend a range,
  run an adaptive sweep, or introduce an active point-guidance interface.
- Fit ATE against the **signed penalty**, not log penalty, and extrapolate to
  penalty zero. Positive penalties favor smaller ATEs and supply the lower-end
  search; negative penalties favor larger ATEs and supply the upper-end search.
  Keep the two regressions separate.
- Document the actual penalty points/ranges used by worked examples. They are
  setup-specific examples, not a universal range recommendation or a default
  grid.
- **Fresh start is the default. Warm start remains an explicit option in v0.1.**
  Both modes must be available. Fresh starts reset the trainable state for each
  point rather than carrying over the preceding fit; warm starts reuse a preceding
  fit within a sweep. Record the actual run order and initialization behavior so
  results are interpretable. Do not silently drop a mode or add an extensive
  multistart requirement.
- No resumability, resume API, stop/continue prompt, or fit-only-new-points
  workflow is required. This does not prevent calling a pure postprocessor on
  an existing in-memory result; it must not imply resuming training.

### Selection family and result interpretation

Use **approach 2: exhaustive contiguous-window selection** for v0.1. It searches
possible approximately linear regions rather than assuming that an extreme-end
core is reliable. Among adequately supported, comparably good candidates, prefer
proximity to zero. Do not revive inward growth from an arbitrary trusted core
(approach 1), or require robust/consensus fitting (approach 3) as another v0.1
implementation or comparison.

A computable but unreliable endpoint is returned with a warning. An endpoint
that cannot be computed is unavailable; do not invent a number. The detailed
fallback ranking, selector metadata, and warning criteria belong to the one
remaining selector discussion, not a reopening of this failure policy.

An apparently linear window and a stable intercept are diagnostics, not a
certificate of causal validity, sharpness, or global optimization. Retain the
observational fit diagnostics for the underlying runs; the extrapolated endpoint
does not itself correspond to a newly fitted distribution.

## Implementation and review checklist

After team discussion and a fresh implementation request:

1. Add the single-margin projection/ID check and a few identifiable,
   nonidentified, and unsupported-input cases.
2. Generalize IV names/supports and contrasts; preserve categorical provenance;
   validate states and strict instrument counts. Reuse shared data preparation
   and the canonical trainer.
3. Add the small binary/non-binary reference collection and both fit summaries.
   Include a ternary treatment contrast, categorical-outcome warning, and
   absent/singleton instrument-level checks. Numerical bound-recovery quality
   remains a manual review item.
4. Connect the native sweep and postprocessor with both initialization modes,
   fresh start by default, and only user-supplied penalties.
5. Implement the selector only after resolving the recommendations in the
   [automatic regression-selection review](automatic-regression-selection.md).
   Verify selected points as well as intercepts using a small set of clean,
   bent-end, flat, and weak-input cases.

This PR records the plan and source audit only. The unchanged experimental
postprocessor was exercised on synthetic numeric inputs for the companion
review. No new selector, SEM2 training, gradient check, package-wide test run,
merge, or deployment is claimed.

## Sources

[1] [Inspected estimate_ate_bound branch source: experiments/est_ate_bound.py](https://github.com/causal-design/sem2/blob/6143748181bcf6abbaf8d2cfe88c3ac323d3758a/experiments/est_ate_bound.py)
[3] [Earlier identification handoff (PR #53)](https://github.com/causal-design/sem2/pull/53)
