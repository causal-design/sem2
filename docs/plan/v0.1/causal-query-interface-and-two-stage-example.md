# 4. Causal-query interface and the two-stage example

**Agreed F04/F32 direction for the [v0.1 plan](README.md).** The thread selected
Option B (extend the existing small query operators) and confirmed the two-stage
example below. Review this handoff for fidelity and acceptance coverage, not to
choose again between the earlier alternatives. This is documentation only; the
new signatures, sampler fixes, and fitting example are not implemented here.

## Agreed interface

The [implementation plan](../v0.1.md) separates preparatory changes from the
trainer's technical review and later query implementation. Reuse existing small
operators; proposed names are not permission to add unnecessary classes/methods.
v0.1 fitting and examples are scalar-only. Deliberate categorical scoring happens
in separate preprocessing before training, not inside queries.

Build on `MeanPotentialOutcome` and `ATE`, keeping the target separate from
training configuration. A query names its outcome and interventions and holds a
reference to the supplied model. It does not add global treatment/outcome roles
to the graph or freeze the model's parameters.

- For ordinary use, fit once and evaluate different queries without refitting.
- For query-specific partial-identification fitting, construct the target before
  optimization and retain its differentiable value as the model changes.
- Query evaluation must not silently start fitting, identification checks, or a
  bound search. The [identification and partial-identification handoff](identification-and-partial-identification.md)
  governs those separate operations and their interpretation.

Specify one named scalar outcome per convenience query. Other scalar outcomes
can be queried from the same fit; this does not require a new vector-valued-node
framework. Each intervention dictionary maps user-declared variable names to
fixed values. Its keys are the treatment names; no additional positional list
is needed. Omitted variables follow their equations, an empty dictionary denotes
the natural model regime, and an explicit zero means zero. Never infer a
reference value or copy missing assignments from the other regime.

These dictionaries belong to the public query constructor. Validate and convert
them at that boundary into the [named-data contract](named-data-batches.md#named-data-contract)
used by the sampler. The low-level rejection of legacy dictionary interventions
does not prohibit these agreed user-facing calls.

For a contrast, evaluate the **alternative mean minus the reference mean**.
Allow user-selected levels, including nonbinary comparisons such as `2` versus
`0`; do not hard-code the original 1-versus-0 comparison. Both dictionaries
specify complete regimes by the omission rule above. For the fixed two-stage
comparison, explicitly name both treatments in both dictionaries.

For the ordinary operators below, use means and additive mean contrasts over the fitted model's implied
population, not implicit training-row averaging. For a binary endpoint explicitly
interpreted as an event indicator, the mean is a risk and the contrast is a risk
difference. Preserve the [general categorical-meaning requirements](identification-and-partial-identification.md#general-design-requirements-for-variables-and-estimands):
internal state codes alone do not supply a meaningful numerical outcome. Warn
and suggest a distributional estimand or deliberate numerical conversion when a
categorical outcome lacks that interpretation. This handoff does not settle the
separate conversion-helper design.

The [meeting-decision list](README.md#next-steps-and-review-decisions) records the
remaining failure/result behavior for an unscored categorical mean and the scope
of the suggested probability contrasts. Do not infer a numeric-code fallback or
an additional query family from this warning requirement.

Keep `(value, info)` as the small result convention:

- `value`: a scalar PyTorch tensor, retaining gradients when supported;
- `info`: the outcome and intervention assignments, the component means for a
  contrast, and the numerical method, sample count, and seed used.

Continuous evaluation averages simulated outcomes. Finite-discrete evaluation
weights outcome values by state probabilities and integrates over any remaining
exogenous draws; a probability table is not a batch of outcome samples. State
when integration remains Monte Carlo. A point query does not automatically return
a confidence interval, an identification verdict, or sharp bounds.

### Kang–Schafer ATT example: additional bounded target

The September 17 decision selects a Kang–Schafer adaptation for **ATT**, the
average treatment effect on the factually treated, with observed factual outcomes.
There is no separate missing-data case study. Ordinary `ATE` remains a population
contrast; its existing averaging must not be silently redefined.

Before [example 9b](../v0.1.md#9b-kangschafer-adapted-to-att), specify the adapted
potential-outcome generator, observed adjustment covariates and overlap assumptions,
the treated target distribution, and an independent reference for that same target.
Decide explicitly whether averaging uses observed treated covariates or a fitted
treated-population law. Conditioning on factual treatment is not intervention;
conditioning after setting treatment to treated does not recover ATT. Fitting only
treated rows is not a substitute for confounder adjustment using both arms.

Prefer a small example-local evaluation function; extend existing operators only
if the reviewed estimator genuinely needs it. No `ATT` class, arbitrary conditioning
API, general empirical-distribution system or counterfactual framework is authorized
by the example choice. Include a reviewed heterogeneous-effect check that separates
ATT from ATE; a constant-effect example alone cannot establish correct averaging.
Keep generator-only potential outcomes out of fitting. This is a target and review
boundary, not an implemented estimator or approval of a particular adaptation.

### Ordinary fitted-model calls

The following are **target-API sketches**, not executable calls in the present
package. `fitted` is an existing fitted SEM2 model, not a proposed fit-result
class. Exact import locations, keyword spellings, and whether the generalized
`ATE` name becomes `AverageCausalEffect` can be finalized during implementation
review without reopening this contract. The low-level public intervention name
follows the `do_intervention` direction in the [model–data handoff](minimal-model-data-interface.md#implementation-and-review-checks);
`set_nodes` is an existing implementation starting point, not the selected public
name.

```python
mean, info = MeanPotentialOutcome(
    fitted, outcome="Y", intervention={"A": 1},
).evaluate(n_samples=20_000, seed=17)

q = ATE(
    fitted, outcome="Y",
    alternative={"A": 1}, reference={"A": 0},
)
effect, info = q.evaluate(n_samples=20_000, seed=17)
```

Numerical evaluation settings are separate from training configuration; the
sample count and seed above are illustrations, not agreed defaults. Changing
these queries must not update fitted parameters.

Option A's simple functions were considered for brevity, and Option C's named
regime table for comparing many regimes. B reuses the existing operator-based
loss composition without adding a second public convention or a new hierarchy.
Neither a function facade nor a regime table is required for this milestone.

## Agreed two-stage example

Let `A1`, `L`, `A2`, and `Y` be observed binary variables, with an unobserved
binary common cause `U` of `L` and `Y`. The directed edges are:

```text
A1 -> L -> A2 -> Y
A1 -> Y
L  -> Y
U  -> L
U  -> Y
```

`A1` is independently randomized with probability 0.5. `L` is intermediate
information affected by `A1` and `U`. `A2` uses a fresh coin with probability
0.25 when `L=0` and 0.75 when `L=1`. Thus both treatments are randomized, but
second-stage assignment depends on observed history and has overlap in both
histories. All sampling innovations are mutually independent, including `U`
and the treatment-assignment coins. Assume consistency and no interference;
there is no missingness or censoring in this reference example.

Use `P(U=1)=0.5` and these conditional probabilities for Bernoulli draws. This
block defines the reference data generator, **not SEM2 formula syntax**:

```python
def p_l(a1, u):
    return 0.2 + 0.3*a1 + 0.3*u

def p_a2(l):
    return 0.25 + 0.5*l

def p_y(a1, a2, l, u):
    return 0.1 + 0.1*a1 + 0.2*a2 + 0.2*l + 0.1*u + 0.1*a1*a2
```

The interaction makes the second-stage effect depend on the first treatment;
two single-treatment effects cannot simply be added. Under each fixed regime,
regenerate `L` after setting `A1`, then fix `A2` and propagate to `Y`. Holding
`L` at its observed value, or intervening on `L` too, changes the question.

Write `mu(a1,a2)` for the population mean of `Y` under the joint intervention.

| Fixed regime | Mean Y |
|---|---:|
| `A1=0, A2=0` | 0.22 |
| `A1=1, A2=0` | 0.38 |
| `A1=0, A2=1` | 0.42 |
| `A1=1, A2=1` | 0.68 |

The primary contrast is **both stages versus neither**, `mu(1,1)-mu(0,0)=0.46`.
A useful second query is **second-stage treatment at fixed first-stage treatment**,
`mu(1,1)-mu(1,0)=0.30`. These are population references, not promises of exact
recovery from finite-data fitting.

```python
q_joint = ATE(
    fitted_two_stage, outcome="Y",
    alternative={"A1": 1, "A2": 1},
    reference={"A1": 0, "A2": 0},
)
value, info = q_joint.evaluate(n_samples=20_000, seed=17)

q_second = ATE(
    fitted_two_stage, outcome="Y",
    alternative={"A1": 1, "A2": 1},
    reference={"A1": 1, "A2": 0},
)
value_second, info_second = q_second.evaluate(n_samples=20_000, seed=17)
```

Both calls use the same fitted model. Their syntax is illustrative on the same
terms as the ordinary calls above.

For an independent observed-data reference, use the sequential g-formula:

$$\mu(a_1,a_2)=\sum_l P(Y=1\mid A_1=a_1,L=l,A_2=a_2)P(L=l\mid A_1=a_1).$$

Under the stated sequential randomization assumptions, it agrees with structural
enumeration despite the unobserved `L`–`Y` confounding. The naive observed-group
comparison `E[Y|A1=1,A2=1]-E[Y|A1=0,A2=0]` is approximately 0.552174, not 0.46:
selection into the second-stage treatment group depends on `L`. This distinction
is why history-dependent randomization is preferable to two independent fair
coins for the example.

The assignment rule depends on history, but the queried regimes are fixed.
A policy that assigns `A2` from the newly simulated `L` would need a separate
policy interface; that is not required here.

## Implementation and acceptance checks

Start from the [scalar operators](../../../sem2/operators/scalar_operators.py),
[base sampler](../../../sem2/model/models/base.py),
[finite-state sampler](../../../sem2/model/models/discrete.py), and
[potential-outcome sampler](../../../sem2/operators/vector_operators.py).
Keep fixes local and separate implementation PRs from this specification.

1. **Query contract:** remove treatment/outcome dependence on `TrainingConfig`;
   test changing outcome, treatment names, levels, and contrast direction on the
   same fitted model without changing its parameters. Retain differentiability
   for query-specific fitting. Test the natural-regime omission rule and result
   metadata, with no silent training or search.
2. **Intervention semantics:** validate names and declared states before coercion;
   reject unknown, ambiguous, fractional-invalid, and out-of-support assignments
   rather than clipping or silently substituting a state. Construct values with
   appropriate dtype/device. Test equation replacement, unaffected mechanisms,
   joint assignments, regenerated intermediate variables, probability
   normalization, and equivalent named mechanisms with reordered parents.
   Preserve categorical meaning rather than averaging arbitrary codes.
3. **Known-model reference:** check all four regime means and both contrasts
   against independent structural enumeration and the observed-data g-formula.
   Keep simulation error distinct from fitting error. Match exogenous draws
   explicitly when using paired simulation checks; a reset global seed alone
   does not guarantee pairing if regimes consume random draws differently.
4. **Observed-data fit:** generate a DataFrame containing only `A1`, `L`, `A2`,
   and `Y`, infer `U` as unobserved through the agreed model–data preparation, and
   fit through the package-owned discrete route. Do not reveal `U` or fix the
   unknown outcome mechanism at its truth. Inspect fitting loss/gradient history
   and compare queries to the same reference at human-reviewed tolerances.
   Deliver an executable Python example and independent tests, not a fourth
   notebook. The training budget, sample size, and recovery tolerance are
   implementation-review details, not constants approved by this document.

The earlier read-only review at source revision
`7c13d6cac1374da797c6efb84272bbdfe401fd05` found three relevant regression cases:
a binary intervention of 0.9 silently became 0; value 2 could yield zero total
probability; and changing declared parent order with equivalent named mechanisms
moved the sequential contrast from about 0.460 to 0.495. An integer-valued
continuous ATE intervention could also cause a dtype error. Treat these as
historical findings to reproduce and fix, not claims about every later branch.
The prior native finite-state calculation used analytically assigned mechanism
weights; it did **not** establish observed-data fit recovery. Neither that check
nor a passing existing test suite substitutes for the acceptance work above.

## Deferred extensions and review boundary

Dynamic/stochastic policy queries, general query-expression languages, named
regime result tables, ratios/quantiles, a new result hierarchy, and general
uncertainty estimation are not prerequisites for F04/F32. They are not assigned
to a subsequent numbered release by this agreement.

The interface family and two-stage example are settled. Reviewers should check
fidelity, scientific assumptions, and test coverage; exact API spellings and
numerical fitting tolerances remain for implementation review. Cross-document
questions about categorical scoring and failure behavior are collected in the
[parent plan's meeting list](README.md#next-steps-and-review-decisions), without
reopening the agreed interface or generator. The separate identification-selector
discussion is unchanged. This documentation does not authorize implementation,
merge, deployment, or release. Original meeting records remain historical
evidence; the accepted interface and example above remain the agreed direction.
