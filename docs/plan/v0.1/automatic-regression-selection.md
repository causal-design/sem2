# Automatic regression selection: current code and v0.1 proposal

Companion to [Identification checks and partial-identification results](identification-and-partial-identification.md), F33–F35 of the [v0.1 plan](../v0.1.md).

**Status:** the team selected approach 2, exhaustive contiguous-window selection.
Its detailed rules are the only remaining unsettled topic in this handoff.
The recommendations below are for team discussion, not approved defaults or an
implemented/validated selector. In particular, the explicit scale formula and
numerical cutoffs below are new draft choices that make the recommendation
concrete; they were not previously agreed in Slack. Implementation is paused.

The other decisions remain settled: native in-memory sweep/postprocessing stages,
user-supplied penalties only, signed-penalty extrapolation to zero, fresh start by
default with warm start available, no resumability, and estimates with warnings
when unreliable but computable. The main handoff records the full contract.

Each decision starts with a question, a short action summary, and any alternatives
considered, followed by the full explanation. **Agreed** and **proposed** actions
remain distinct; this reorganization does not approve any pending selector rule.

## 1. What the current code does

- **Which selection approach should replace the current code?**

  **Action (agreed):** Use exhaustive contiguous-window search (approach 2),
  preferring proximity to zero among comparably good windows.

  **Alternatives considered:** Inward growth from a trusted core (approach 1,
  rejected); robust/consensus fitting (approach 3, not selected for v0.1).

**Detailed explanation and current-code evidence**

Inspected source: `experiments/est_ate_bound.py` on `estimate_ate_bound`, commit
`6143748181bcf6abbaf8d2cfe88c3ac323d3758a`. The remote branch tip was rechecked for
this handoff. The unchanged local source used for the probes has Git blob
`9dd863afdb808d3610cd0d7f4de657230f38bfe4`.[1]

### Exact selection sequence

The function is
`estimate_ate_residual_comp(high, x_all, y_all, min_points=3, studentized_threshold=1.0)`.
It performs the following steps.[1]

1. Reshape one-dimensional `x_all` into a single-column array. With fewer than
   two rows, return an **unfitted** `LinearRegression`, the supplied arrays, and
   a `NaN` intercept (lines 23–30).
2. Sort by the raw, signed first x column in ascending order and reorder y to
   match (lines 32–35). There is no internal positive/negative split.
3. Set the initial size to `min(max(2, min_points), n)`. With `high=True`, select
   the largest x values and consider remaining points in descending x order.
   With `high=False`, select the smallest x values and consider the rest in
   ascending order (lines 37–44). The initial core is accepted without a
   linearity or intercept-stability check.
4. Tentatively insert the next candidate into that core. Fit a statsmodels OLS
   regression **including the candidate**, with an added constant. Obtain
   `resid_studentized_internal` from the fitted model's influence diagnostics
   (lines 46–60).
5. Check only the new candidate's absolute internally studentized residual:
   position zero for `high=True`, position minus one otherwise. Accept it only
   when the value is **strictly less than** `studentized_threshold`. On the first
   failure, stop; do not consider later points or another core (lines 62–68).
6. Refit `sklearn.linear_model.LinearRegression` on the retained points and return
   `(final_model, x_selected, y_selected, final_model.intercept_)` (lines 70–78).
   The intercept is the prediction at signed penalty zero.

This is inward growth from a privileged extreme-end core, not an exhaustive
window search. Its trial statistic uses the candidate in the fitted line and
residual scale used to assess that same candidate. It is not a prediction error
from a line fitted without the candidate. It neither checks energy distance nor
uses it as a regression weight or point-selection criterion.[1]

The function does not explicitly validate distinct penalties, finite values,
rank, side membership, or zero penalties. It does not produce a structured
reliability warning. Duplicate x values are not equivalent to distinct support
for extrapolation even when sklearn returns an intercept.[1]

### Demo behavior is not the future package interface

`main()` reads at most eight rows from `results_mediated_10.csv`, requesting four
columns, but passes only the first (x) and second (y) to the selector. It calls
both directions on the **same full array**, with threshold **1.5**, whereas the
function default is **1.0**. The `high=True` call is described/plotted as lower
bound but stored in variables named `upper`; the final printed upper/lower
labels reverse the comments and plot labels. Preserve the objective's sign
convention, not these variable names or final print labels (lines 81–115).[1]

### Executed checks of the unchanged function

These are synthetic numeric inputs, not fitted SEM2 trajectories. The source
blob was verified before import; `main()` and its CSV workflow were not run.

| Input/check | Observed output |
|---|---|
| Sloped line with zero intercept 0.2, both directions | Intercepts approximately 0.2; each direction retained its three-point core rather than necessarily accepting the whole exact line. |
| `x=(1,2,3,4)`, `y=(1,2,3,100)`, `high=False`, threshold 1.5 | Candidate's absolute internal studentized residual approximately 1.414214; all four points accepted; intercept **-48**. |
| One point | `NaN` intercept and an unfitted model. |
| Two points, `x=(1,2)`, `y=(2,4)` | Fitted line with intercept approximately zero, without a reliability warning. |
| Four repeated penalties, all x equal to 1, y from 1 to 4 | Intercept 2.5; no distinct-x validation. |
| Five zero responses at distinct penalties | Intercept zero, only the three-point core retained; runtime warning from division in the studentized residual. |
| A `NaN` response in the initial core | `ValueError: Input y contains NaN.` |

The extreme fourth-point check shows why merely adjusting the old threshold is
not the selected fix. It does not validate the replacement or establish a
universal failure rate for the old code.

## 2. Recommended approach 2, step by step

All detailed rules in this section remain recommendations. Their purpose is to
let the team review an explicit algorithm, rather than phrases such as
"sufficiently linear" without an operational meaning.

### A. Prepare each endpoint's points

- **Which points are eligible, and what does contiguity mean when runs fail?**

  **Action:** Split by sign (**agreed**). Exclude zero, reject duplicate penalties,
  and allow warned failed-run gaps but no skipped usable interior points (**proposed**).

  **Alternatives considered:** Averaging repeated fits or deleting isolated usable
  points; neither is recommended in the draft.

**Detailed explanation**

- Use positive penalties for the lower endpoint and negative penalties for the
  upper endpoint. Sort each side by increasing absolute penalty, while retaining
  the original signed x values for every regression.
- Keep zero-penalty runs in diagnostics but exclude them from both regressions.
- Require one run record per distinct supplied penalty for v0.1. Reject duplicate
  penalty specifications before training rather than silently averaging stochastic
  fits or pretending repeats provide distinct-x support.
- Omit failed or non-finite run results from regression, preserving their reasons
  and original positions. A finite result with a training warning remains visible
  and usable with that warning; do not invent a new convergence cutoff here.
- Define contiguity on the remaining sorted usable points. A window may bridge
  an omitted run, but warn if the selected interval does so. It must contain
  **every usable point** between its endpoints; no isolated finite-point deletion.

This sorting is postprocessing only. It must not silently change the recorded
training order or the meaning of an explicitly chosen warm start.

### B. Enumerate and fit all eligible windows

- **How much support is needed, and how should windows be fitted and scored?**

  **Action (proposed):** Require three distinct penalties; use unweighted OLS and
  leave-one-out prediction error; record intercept sensitivity and window geometry.

  **Alternatives considered:** A trusted initial core, the old internal
  studentized-residual check, and pure R-squared ranking; none is selected.

**Detailed explanation**

For each side, enumerate every contiguous window containing at least **three
distinct penalties**. No initial window is trusted, and either end of the curve
may be excluded. Fit an ordinary unweighted line with an intercept against signed
penalty. Center/scale the design internally for numerical conditioning, then
transform the zero prediction back to the original coordinates.

For each window record:

- its point identities, signed range, point count, and zero intercept;
- `E`: root-mean-squared leave-one-out prediction error. For each point, fit the
  line to the other points in that window, predict the held-out response, then
  take the square root of the mean of those squared errors;
- `D`: the larger absolute change in zero intercept after separately removing
  the nearest-zero and farthest-zero boundary point and refitting;
- `gap`: the smallest absolute penalty in the window;
- `span`: the largest minus smallest absolute penalty in the window.

Treat a rank-deficient or non-finite fit/score as uncomputable for that operation;
do not substitute a plausible score. Three-point windows are allowed, but have
weak support and receive a warning if selected. Boundary-deleted two-point fits
can provide a sensitivity diagnostic, not evidence of linearity on their own.

### C. Use one detrended scale per sign side

- **How should scatter and numerical tolerances be estimated?**

  **Action (proposed):** Use one adjacent-triple detrended scale per sign side,
  with a numerical floor and weak-scale warnings; the formula remains uncalibrated.

  **Alternatives considered:** Separate noise allowances for each window or an
  ordinary user-specified tolerance; neither is recommended.

**Detailed explanation**

Recommended starting convention: estimate scatter once from adjacent usable
triples on that side, and use the **same scale for all its candidate windows**.
For a consecutive triple `(x_left,y_left), (x_mid,y_mid), (x_right,y_right)`, form:

```text
w = (x_mid - x_left) / (x_right - x_left)
d = (y_mid - ((1-w)*y_left + w*y_right)) / sqrt(1 + (1-w)^2 + w^2)
sigma = median(abs(d) over the side's triples) / 0.6744897501960817
tau = max(sigma, numerical_roundoff_floor)
```

The contrast removes an affine trend and accommodates unequal penalty spacing.
Under independent equal-variance errors on a truly linear triple, the denominator
normalizes its error standard deviation; the median calibration is Gaussian.
**Those assumptions are motivation for a heuristic scale, not an established
noise model for neural fits.** Overlapping triples are dependent, curvature can
inflate the scale, and a small number of triples gives little information.
Warm-start dependence is another reason not to interpret these scores as tests.

Use a dtype-aware roundoff floor, recommended initially as the larger of the
smallest positive normal value and `64 * machine_epsilon * max(abs(y))` for that
side. This is a numerical safeguard, not a user-specified ATE tolerance. Guard
intermediate overflow/underflow and verify behavior under changes of units.

Warn when fewer than three usable triples support the scale. With only three
points, the single triple cannot independently distinguish noise from a bend;
returning a finite line must remain a warned result. The shared scale prevents
separate windows from each granting themselves a different noise allowance, but
it does **not** make the scale independent of the points being assessed.

This estimator is a proposed simplification for review, not a settled or
calibrated solution to the noise-versus-curvature problem.

### D. Decide which windows are comparable, then choose one

- **What is good enough, how do proximity and support compete, and when should stability warn?**

  **Action (proposed):** Apply quality/comparability thresholds, then prefer
  proximity and support; use intercept sensitivity for warnings, not selection.

  **Alternatives considered:** Pure R-squared ranking or a weighted score combining
  quality, proximity, support, and stability; neither is recommended.

**Detailed explanation**

Recommended initial constants and deterministic ordering:

1. Call a scorable window provisionally adequate when `E <= 3*tau`.
2. Among adequate windows, find the smallest `E`, called `E_best`. Keep windows
   satisfying `E <= E_best + tau` as comparably good.
3. Among those, prefer the smallest `gap` (nearest edge to zero), then the largest
   point count, then the smallest `E`, then the largest `span`. Use original
   sorted window indices as the final deterministic tie-breaker.
4. Refit the selected window and return its zero intercept. Keep `D` as a
   diagnostic, not an extra weighted term in the selection score.
5. Warn if `D > 3*tau`, or if `gap > span` (a narrow region far from zero), or
   if a boundary refit needed for `D` cannot be computed. These warnings do not
   discard an otherwise computable endpoint.

The `3` multipliers and one-scale comparison band are **recommended development
starting values only**. They are not significance levels, guarantees of fit,
or numerical bound-recovery accuracy thresholds. Their values and the near-edge
notion of proximity remain open for team review. Pure R-squared ranking is not
recommended; a flat linear region is eligible.

### E. Apply the agreed failure policy with explicit fallback rules

- **Which unreliable fit should be returned, and when is an endpoint unavailable?**

  **Action:** Retain computable estimates with warnings (**agreed**); use the
  score-first, then support-first fallback ordering below (**proposed**).

  **Alternatives considered:** Switching fitting families, silently repairing crossed
  endpoints, or generating more training points; none is recommended or in scope.

**Detailed explanation**

- If no window is provisionally adequate but some have finite `E`, choose the
  smallest `E`, breaking ties by the same gap/support/span/index preferences.
  Return its intercept with a no-adequate-window warning and other applicable
  diagnostics. Do not silently switch to a different fitting family.
- If no leave-one-out-scored window is available, consider computable contiguous
  OLS fits with at least two distinct penalties. Prefer the largest point count,
  then smallest gap, then widest span, then sorted indices. Return the intercept
  with an unavailable-quality-score warning. A two-point fallback always warns
  that linearity cannot be assessed.
- If fewer than two distinct usable penalties remain, or no stable finite line
  and zero prediction can be computed, mark that endpoint unavailable. One
  endpoint can be available while the other is not; do not fabricate a closed
  interval from one endpoint.
- If both endpoints are computed but lower exceeds upper, retain their labels
  and warn. Do not silently swap, clip, or repair them.

No failure triggers automatic new training points or a resume prompt.

### F. Return the endpoint and its diagnostics

- **What should the selector return?**

  **Action (proposed fields):** Return the endpoint or unavailable reason, selected
  points, fit diagnostics, warnings, and the selection/fallback reason.

  **Alternative considered:** A legacy CSV-shaped interface; native in-memory
  results are already agreed, with serialization separate.

**Detailed explanation**

The proposed result contains the endpoint or unavailable reason; selected point
identities and values; slope and intercept; prediction error, shared scale,
stability, gap and span diagnostics; warnings; and the selection or fallback
reason. These are the fields proposed in the review table below, not an approved
result schema. Use the package's native result conventions rather than a CSV
schema. Sections B–E specify the calculations and warning behavior under review.

## 3. Open questions and recommended answers

These are all subquestions of the **automatic selector**. The sweep, initialization
modes, estimand meaning, and general failure policy are not open again.

| Question for team discussion | Recommended default for review |
|---|---|
| Minimum support for selection? | Three distinct penalties; warn on a selected three-point window. Two points only as a warned fallback. |
| What is a contiguous window when runs fail? | Contiguous among usable points; allow a failed-run gap but warn if the chosen interval spans it. Never skip an interior usable point. |
| Zero or duplicate penalties? | Zero only in diagnostics; reject duplicate penalty specifications rather than aggregate repeated fits. |
| Regression and score? | Unweighted OLS with intercept; raw leave-one-out prediction RMSE as the primary score, not R-squared. |
| How is the shared scatter scale estimated? | The adjacent-triple detrending formula in C, with a numerical floor and explicit weak-scale warnings. Its behavior under curvature/sparse points needs review. |
| What counts as acceptable/comparable? | Start with `E <= 3*tau`, then within `tau` of the best adequate score. Review/calibrate these internal constants on a few examples. |
| How do proximity and support compete? | Quality first; among comparable candidates, smallest near-edge gap, then most points, then error/span/index tie-breakers. No arbitrary weighted sum. |
| Does intercept sensitivity affect selection? | Initially a warning diagnostic only; boundary deletion and the gap/span warning as in D. |
| Which unreliable fit is returned? | Best prediction score when available; otherwise the largest computable contiguous OLS fit, with explicit fallback warnings as in E. |
| What does the selector return? | Endpoint or unavailable reason; selected point identities/values; slope/intercept; prediction error, shared scale, stability/gap/span diagnostics; warnings and selection/fallback reason. Use native result conventions, not a CSV schema. |

## 4. Small implementation check set after agreement

- **What should be checked once the selector rules are agreed?**

  **Action (proposed):** Run a small set of checks covering selected points,
  intercepts, weak inputs, ranking, rescaling, and warnings.

  **Alternatives:** No separate test strategy is recorded here. The old-code probes
  above are evidence about that code, not validation of its replacement.

**Detailed explanation**

Check the selected points and intercept, not just a fit score:

- A clean sloped line and a flat line, with noiseless numerical handling.
- A linear middle region with poor points at both ends, without a trusted core.
- Weak inputs: too few distinct penalties, failed/non-finite runs, duplicates,
  and a side that cannot return an endpoint.
- Comparable windows with different proximity/support, including deterministic
  tie-breaking, modest noise changes, and response/penalty-unit rescaling.
- A narrow window far from zero and crossed endpoints, checking warnings rather
  than silently repairing the result.

Keep this proportional to the small postprocessing feature. No new selector was
implemented or benchmarked for this handoff, and no SEM2 fitting or package-wide
tests were run. The completed execution evidence in section 1 concerns only the
unchanged experimental function on explicitly synthetic inputs.

## Sources

[1] [Inspected estimate_ate_bound branch source: experiments/est_ate_bound.py](https://github.com/causal-design/sem2/blob/6143748181bcf6abbaf8d2cfe88c3ac323d3758a/experiments/est_ate_bound.py)
