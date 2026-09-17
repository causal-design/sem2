# 5. Identification checks and partial-identification results

Design plan for F33–F35 of the [v0.1 plan](README.md), including the general
variable and estimand requirements discussed alongside them. **Within the F33–F35
method discussion, the remaining open topic is the detailed rule for automatic regression-window selection.** Its broad
approach is agreed; recommendations below remain unapproved and unvalidated.
Implementation is paused pending team discussion and a fresh request.

Obtain approval from Slack reviewer `U044USKH8HH` before merging this plan.
This Slack ID is not a verified GitHub username. Plan approval, implementation,
and integration into the default branch are separate decisions.

## General design requirements for variables and estimands

These apply throughout the package, not just to the IV example.

The September 17 [implementation plan](../v0.1.md) makes steps 1–5 preparatory,
with new classes/methods justified against simpler reuse, and step 6 a separate
trainer technical review. v0.1 variables are scalar. Deliberate category-to-numeric
scoring is separate preprocessing before training, not automatic internal encoding.
The Kang–Schafer example targets ATT in the factual-treated population; its bounded
adaptation/averaging review is in the [query handoff](causal-query-interface-and-two-stage-example.md#kangschafer-att-example-additional-bounded-target).
The population ATE IV searches below and the single-margin ID contract are unchanged;
neither automatically supplies an ATT estimator or verdict under additional assumptions.

- **Preserve what variables mean after numerical conversion.** Retain each
  variable's original categorical status, labels, and mapping after both automatic
  encoding and deliberate conversion. Record any numerical interpretation supplied
  by the user. Any variable may later become an outcome; internal category codes
  alone do not make its mean or ATE meaningful.

  For an originally categorical outcome without a numerical interpretation, warn
  that ATE is not well-defined and suggest a contrast in category probabilities
  or deliberate conversion to meaningful numerical values. A genuinely numerical
  outcome with finite support does not trigger this warning. Preserve category
  identities for treatments and covariates too; this preserves query meaning,
  not necessarily numerical fits under different encodings. Coordinate with the
  [model–data plan](minimal-model-data-interface.md) and F12/F28 conversion work;
  exact conversion rules and naming are outside this discussion.

- **Let users choose treatment levels and contrast direction.** For ATE, `2`
  versus `0` means the population mean outcome under treatment level `2` minus
  that under level `0`, not the difference between observed groups. Use that
  contrast consistently in ordinary queries, numerical searches, and reporting.
  Name treatment and outcome roles explicitly rather than fixing column positions
  or example names. Validate intervention values before numerical coercion and
  match categories through their mapping; never silently substitute another state.

## Graphical identification and numerical searches

- **Return a clear identification verdict.** Project the supplied DAG onto the
  observed variables as an acyclic directed mixed graph and apply ID for one
  observational margin. Return `True` or `False` for supported problems and an
  explicit error for unsupported inputs; an identification formula is not required.
  Data fusion, latent-variable targets, and identification from extra parametric
  or proxy assumptions are outside this check's scope.
- **State what was checked.** Identification of an interventional distribution
  supports its means and contrasts. Failure to identify the whole distribution
  does not automatically establish nonidentification of every scalar contrast.
- **Keep bound searches explicit.** If ID succeeds, do not search for lower and
  upper values by default, although estimation may still require fitting. If it
  fails, warn and offer an explicitly requested, estimand-specific numerical
  search. Such a search adds the signed estimand to the fitting objective;
  changing an ordinary evaluation query must not start training. Nearly equal
  numerical endpoints do not prove identification.

## Discrete IV example

Use an exogenous/randomized instrument, exclusion of a direct instrument effect
on the outcome, consistency, and possible unobserved treatment–outcome
confounding. Do not add monotonicity/no-defiers or linearity. ATE remains the
population contrast for the selected treatment levels, not a complier effect.
Observational fit does not validate these assumptions.

Support named instrument, treatment, and outcome roles and declared finite
supports, including nonbinary treatments and instruments. Compare the same
observed margins, marginalizing unobserved variables. Reuse the general variable
and query rules above; keep IV-specific fitting separate from general sampling.

Require **at least two retained observations per declared instrument level**
after complete-case filtering across the selected observed variables. Otherwise,
stop before fitting and report affected levels, counts, and the minimum. Do not
drop levels, merge categories, or add pseudo-counts. Check the retained dataset;
batching must separately avoid invalid loss calculations. This is a fitting
requirement, not an identification or reliability guarantee.

## Numerical results and checks

Keep both fit summaries, separately for lower and upper searches:

- The observational fitting discrepancy, separate from the signed estimand term,
  alongside total objective and estimand history. Its scale depends on encoding
  and distance; a negative unbiased discrepancy estimate is not better-than-perfect fit.
- For discrete IV, each instrument level's retained count, total-variation distance
  between model and data conditional probability tables, and largest absolute
  cell-probability error. Compare identical observed margins and category mappings.

Use independent finite-IV linear-programming bounds **in tests only**, with
Balke–Pearl bounds as the binary reference. Include a small binary/nonbinary
collection, with a treatment contrast such as `2` versus `0`. Match the observed
probability table, assumptions, contrast, and numerical outcome values; population
bounds and bounds for a sampled table are different references. Report endpoint
errors, range width, any missed reference interval, and available simulation truth
for manual review. Logical checks are pass/fail; no numerical bound-recovery
accuracy threshold or automatic good-fit threshold is agreed for v0.1.

Report numerical search estimates, not automatically sharp bounds or confidence
intervals. Good fit, a narrow range, or a straight regression does not certify
causal validity, global optimality, an enclosing identified set, or sampling
coverage. An extrapolated endpoint has no newly fitted distribution attached to it.

## Fitting and extrapolation: agreed workflow

1. **Fit at the user's supplied penalties.** Use the package's common fitting
   machinery. Retain each signed penalty, ATE, observational discrepancy, total
   objective, initialization/run order, and available training or failure information.
2. **Select a region and extrapolate using those results.** This separate
   calculation uses in-memory results, without training again or adding penalties.
   Positive penalties supply the lower estimate; negative penalties supply the
   upper estimate. Regress on signed penalty, not log penalty, and predict at zero.

**Fresh start is the default; warm start remains an explicit option.** Fresh
starts reset the trainable state for each penalty; warm starts reuse a preceding
fit. No extensive multistart requirement is added. Do not generate or extend grids,
recommend a universal penalty range, or require resumable training. Worked examples
should state their actual penalties. File export is separate; no legacy CSV
compatibility is required.

Return a computable but unreliable endpoint with a warning. Mark an uncomputable
endpoint unavailable; one available endpoint does not make a complete interval.

## Automatic regression selection

- **Decision:** Which fitted penalty points should determine the line extrapolated
  to zero?
- **Chosen approach 2:** Search all eligible consecutive stretches of points
  (*windows*) for an approximately linear region. Among adequately supported,
  comparably good fits, prefer proximity to zero. The usable region can be in
  the middle: fits very close to zero may be poor, while larger penalties may
  lie beyond the approximately linear region.

The [existing experimental method](https://github.com/causal-design/sem2/blob/6143748181bcf6abbaf8d2cfe88c3ac323d3758a/experiments/est_ate_bound.py)
grows from an unchecked starting group and assesses each added point using a line
that already includes it. Earlier synthetic checks showed this could accept a
large bend. That motivates replacing the selection rule, not merely adjusting its
threshold; it does not validate the replacement. Robust/consensus fitting was
also considered but is not selected for v0.1.

### High-level algorithm

The search family is agreed; the following scoring and ranking details are proposed.

1. Separate positive and negative penalties and order each side by distance from zero.
2. Examine every eligible window, allowing either end of the curve to be excluded
   but no skipped usable interior point.
3. Fit an ordinary unweighted straight line. Assess prediction error by leaving
   out each point in turn and predicting it from the remaining points.
4. Compare errors against a common scatter estimate for that side to identify
   acceptable and similarly good fits.
5. Among comparable fits, prefer a window nearer zero, then one with more points.
6. Extrapolate the chosen line to zero and report the selected points and warnings,
   following the agreed computable/unavailable policy above.

## Checks before implementation is accepted

After agreement and a fresh implementation request, check ID results and unsupported
inputs; general categorical meaning and chosen contrasts in a non-IV case; absent
or singleton instrument levels; the IV reference cases and both fit summaries; and
both initialization modes. For selection, check chosen points and intercepts on a
few straight, flat, curved-end, sparse/failed-fit, and rescaled examples, including
narrow distant windows and crossed endpoints. Earlier executions concerned only
the old selector, not the proposed replacement or fitted SEM2 trajectories.

## Open questions: selector details

These recommendations are **not approved defaults**. Exact formulas, cutoffs,
and tie-breakers still need agreement; they are not statistical guarantees.

- **Which points are eligible?** Recommend at least three distinct penalties,
  warning when only three support the selected line; two points are a warned
  fallback and cannot assess linearity. Keep zero for diagnostics, reject duplicate
  specifications, and omit failed/non-finite results while retaining their reasons.
  A window may span an omitted run with a warning, but must include every usable
  interior point. Finite results with training warnings remain usable and flagged.
- **What counts as sufficiently linear?** Recommend leave-one-out prediction error
  and one shared scatter estimate per side, estimated after removing local linear
  trends from adjacent triples. With few points, noise and curvature are hard to
  distinguish; warm starts can also create dependence. The scale estimate,
  numerical safeguards, and acceptable/comparable-error thresholds remain open,
  especially for nearly noiseless data. These are not significance tests or
  tolerances users should have to tune in ordinary use.
- **How should fit quality, proximity, and support compete?** Recommend quality
  first, then the nearest edge to zero, then more points among comparably good
  windows. The comparison band and final tie-breakers remain open.
- **When should we warn, and which unreliable estimate should we return?** Recommend
  using the change in intercept after removing either boundary point as a warning,
  rather than another selection score. Warn about a narrow window far from zero,
  unavailable sensitivity calculations, and crossed endpoints; do not silently
  swap or repair endpoints. If no window passes, prefer the best prediction score;
  if none can be scored, prefer a computable window with the most points, at least
  two distinct penalties. Warning thresholds and fallback ordering remain open.
- **What should accompany the estimate?** Recommend the selected points, fitted
  line, fit quality, sensitivity, warnings, and selection or unavailable reason.
  The precise contents remain to be agreed.
