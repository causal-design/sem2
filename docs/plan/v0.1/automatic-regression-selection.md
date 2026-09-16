# Automatic regression selection: design discussion

Companion to [Identification checks and partial-identification results](identification-and-partial-identification.md).

- **Decision needed:** Which fitted penalty points should we use to draw a straight
  line and extrapolate the ATE to penalty zero?
- **Chosen approach:** Approach 2, an exhaustive search over contiguous windows.
  Compare all eligible stretches of consecutive points rather than trusting a
  preselected starting group. Among similarly good, sufficiently supported
  straight-line fits, prefer one nearer zero.
- **Alternatives considered:** Growing a region from an arbitrary starting group
  (approach 1, rejected), or robust/consensus fitting (approach 3, not selected for
  v0.1).

The approach is agreed; its detailed rules are **still proposals**. Implementation
remains paused. The [technical reference](automatic-regression-selection-technical.md)
preserves the earlier code review, calculations, and numerical recommendations.

## 1. High-level algorithm

The input is a set of user-supplied penalties and their fitted ATE values. A
*window* is a consecutive stretch of these points. The usable region may lie in
the middle: fits very close to zero can be poor, while points farther away may
no longer follow a line. The proposed algorithm is:

1. **Separate the two sides.** Positive penalties supply the lower estimate;
   negative penalties supply the upper estimate. Regress on signed penalty,
   not its logarithm.
2. **Consider all eligible windows.** Order each side by distance from zero and
   examine every consecutive stretch with enough distinct penalties. Allow
   either end of the curve to be excluded, but no skipped usable interior point.
   Three points is the proposed minimum.
3. **Fit and assess a straight line.** Fit an ordinary unweighted regression in
   each window. Assess it by leaving out each point in turn and checking how
   well the remaining points predict it.
4. **Compare fit quality.** Judge prediction errors against a common scatter
   estimate for that side. Identify acceptable and similarly good fits. The
   scatter estimate and tolerances are the main open question.
5. **Choose among comparable fits.** Prefer a window nearer zero, then one with
   more points. Proximity must not outweigh poor fit. Exact comparisons and
   tie-breakers remain proposed.
6. **Extrapolate and report.** Extend the chosen line to zero. Return the estimate,
   selected points, and warnings. If no window is acceptable, return a computable
   estimate with a warning; if computation is impossible, mark it unavailable.

This uses existing fitted points only; it neither generates penalties nor starts
more training. The initialization choices in the main handoff remain unchanged.

## 2. Interpretation and checks

A straight-looking region does not establish a sharp causal bound; the underlying
fits and causal assumptions still matter. A short window far from zero can give
an unstable extrapolation. We propose warning when removing either end point
substantially changes the zero prediction.

Check selected points and extrapolated values on a few straight, flat,
curved-end, sparse/failed-fit, and rescaled examples. Earlier experiments tested
the old method, not this replacement.

## 3. Open questions and recommended answers

Only these selector details remain open. The recommendations, including the
technical note's formulas and cutoffs, are **unapproved and unvalidated**.

- **Which points should be eligible?**

  **Recommendation:** Require three distinct penalties for assessing a window;
  use two only as a warned fallback. Keep zero only
  for diagnostics, reject duplicate penalty specifications, and allow a window
  to span failed runs with a warning. Do not skip usable interior points.

  **Alternative considered:** Combining repeated fits rather than rejecting
  duplicate specifications. The current proposal keeps this first version simple.

- **What counts as a sufficiently straight line?**

  **Recommendation:** Use the leave-one-out prediction errors described above
  and one shared scatter estimate per side. The draft estimates scatter after
  removing local straight-line trends; its formula and cutoffs still need review,
  especially with few points or curvature.

  **Alternative considered:** Ranking by R-squared alone; this can undervalue
  a genuinely flat region and is not recommended.

- **When should proximity to zero outweigh a slightly better fit?**

  **Recommendation:** First identify comparably good fits, then prefer the window
  whose nearest edge is closest to zero, followed by more points. The width of
  the comparison band and final tie-breakers remain open.

  **Alternative considered:** Combining all criteria in a weighted score; the
  draft instead uses an explicit order of preference.

- **How should sensitivity and unreliable fits affect the result?**

  **Recommendation:** Use sensitivity to issue warnings, not as another selection
  score. When no window passes, prefer the best prediction score; if no such score
  is available, use the computable window with the most points (at least two
  distinct penalties) and warn. Warn about narrow, distant windows and crossed endpoints
  rather than silently correcting them. Exact warning thresholds and fallback
  ordering still need agreement.

  **Alternative considered:** Including sensitivity directly in the selection
  score. The agreed policy of retaining computable estimates with warnings stays.

- **What information should accompany the estimate?**

  **Recommendation:** Show the selected points, fitted line, fit quality,
  sensitivity, warnings, and reason for the selection or unavailable result.
  These details should be available in the package's ordinary results; their
  precise contents remain proposed.
