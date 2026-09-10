# 5. Identification checks and partial-identification results

Agreed F33–F35 specification for the [v0.1 plan](../v0.1.md), from this topic's
Slack discussion. This is a plan for implementation, not a claim that the package
already meets these requirements. It refines the earlier meeting decisions;
the historical records remain unchanged.

**Review and merge hold:** obtain approval from Slack reviewer `U044USKH8HH`
before merging this plan. This is a Slack ID, not a verified GitHub username.
Implementation and default-branch integration remain separate human decisions.

**Only unresolved decision in this topic: IV extrapolation is undecided and
blocked because its current implementation is absent from the inspected
repository state.** Review Max's implementation before choosing the integration,
extrapolation procedure, or initialization/warm-start policy. Do not assume the
code must be written from scratch or silently choose a minimum number of starts.

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

## Blocked IV extrapolation and implementation handoff

The repository recheck found a fixed-weight IV training example, not an
extrapolation runner.[1] The observational-plus-estimand loss composition is
already present as a building block.[2] The recheck covered remote `main` at
`75d370c`, the node-support checkout at `d9897a0`, notebook source cells,
experiments, tests, and the two then-open PRs (#50 and #52). No penalty-setting
sweep with endpoint extrapolation was located. This observation does not rule
out an implementation in Max's local work, a fork, or another unreviewed branch.

The absence of the current implementation blocks review of **IV extrapolation**
and its **initialization/warm-start policy**. Obtain the exact branch, file, or PR
and inspect it before settling those choices. Do not invent a penalty schedule,
extrapolation model, or minimum restart count. Extensive multistart/continuation
machinery remains outside v0.1's agreed scope.

Bounded implementation work can proceed on the settled parts after review:

1. Add the single-margin projection/ID check, its small result contract, and a
   few reviewed identifiable, nonidentified, and unsupported-input cases.
2. Correct IV-specific names/supports, selectable contrasts, categorical
   provenance, state validation, and strict count checks. Reuse shared data
   preparation and the canonical trainer rather than adding another fit loop.
3. Add the small binary/non-binary reference collection and both fit summaries.
   Include essential regressions for a ternary treatment contrast, a category
   encoding that must not bypass the outcome warning, and absent/singleton
   instrument levels. Numerical quality stays a manual review item.
4. Keep extrapolation marked blocked/undecided until its code can be reviewed.
   Implementation PRs must report real test output. This documentation PR does
   not establish new fitting, gradient, numerical-bound, or runtime ID behavior.

## Sources

[1] [IV training example at the inspected revision](https://github.com/causal-design/sem2/blob/d9897a0d334a647a62c90685220470db13ca5330/experiments/trainer.py)
[2] [SEM2Loss composition at the inspected revision](https://github.com/causal-design/sem2/blob/d9897a0d334a647a62c90685220470db13ca5330/sem2/operators/scalar_operators.py)
