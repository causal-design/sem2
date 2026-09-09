# 1. Minimal model–data interface

**F02/F03 handoff for the [v0.1 plan](../v0.1.md).** This records the agreed
direction from Slack, with non-pandas input handling still under review. It is
an implementation specification, not a claim that the new APIs or parser fixes
already work.

**Review requested from:** Slack reviewers `U0AACS6EYJF` and `U0BEV10KU2X`.
Please check that this captures the agreement and that the acceptance checks are
sufficient for bounded implementation PRs. Import locations and keyword spellings
in the example are illustrative.

## Working direction

- **Use pandas, with direct name matching.** Prepared DataFrame columns name model
  variables. Users select the relevant raw columns; unexpected columns in the
  prepared table are errors. No user-supplied model–column mapping or public
  binding object is needed.
- **Use a thin preparation wrapper, not a new DataFrame system.** Ordinary users
  pass a pandas DataFrame directly to `fit`. Shared internal preparation performs
  conversion and validation without modifying the original DataFrame. It may use
  a private record for aligned data and provenance; neither `SEMData(df)` nor a
  separate `normalize_column_names` call is compulsory. Do not subclass pandas or
  require a richer dataset schema for this work. DataFrame support is agreed;
  rejecting versus automatically converting other tabular inputs is not yet decided.
- **Infer observation status jointly from model and data.** Supplied columns are
  observed; remaining user-declared model variables are unobserved. No separate
  `unobs(...)` or observed-variable declaration. Compiler-generated helpers are
  excluded. Missing cells do not change a variable's status. Unsupported
  model/data/objective combinations fail clearly.
- **v0.1 missing-data policy: complete cases only.** Fit rows complete across all
  selected observed variables. Report input, retained, and dropped row counts;
  never drop rows silently. Unselected metadata must not affect row selection.
  Fail if no complete rows remain. More advanced missing-data handling is deferred
  until **after v0.2**. Complete-case fitting is not a guarantee against
  missingness-related selection bias.
- **Provide automatic column-name conversion.** Produce unique, valid Python
  identifiers, excluding keywords. Preserve already-valid names; handle spaces,
  punctuation, leading digits, keywords, and collisions deterministically without
  merging columns. Report original and converted names; an optional preview can
  help users write formulas, but is not a required processing step. Reject
  ambiguous duplicate source labels and never guess an ambiguous model alignment.
  This is automatic preprocessing/provenance, not a user-authored mapping.
- **Keep fitting internals small.** Derive an ordered observed-name list, its
  name-to-column-index dictionary, and an aligned observation tensor. Validate
  values, shape, dtype, and device; convert once before optimization. Retain the
  name-conversion and row-selection report. The model, objective, and trainer
  retain their separate jobs.
- **Make `fit` a convenience layer over one canonical Trainer.** Both entry routes
  use the same preparation, objective handling, training loop, and result reporting.
  Start from the [package trainer](../../../sem2/sem2trainer.py); consolidate the
  [experimental duplicate](../../../experiments/trainer.py) rather than introduce
  another fitting engine. Return the fit result with preparation and training
  diagnostics. No new checkpoint/continuation framework is required here.

## Intercepts and internal helper names

**Estimate linear-equation intercepts when no numeric constant term is supplied.**
A standalone numeric constant fixes the intercept at that value, replacing the
estimated intercept. In particular, a standalone `0` means no intercept (fixed
at zero); `-1` means an intercept fixed at minus one, not intercept removal.
Use “intercept,” not “bias,” publicly, without requiring a `node_cfgs` block.

| Formula | Coefficient of `x` | Intercept |
|---|---|---|
| `y <- x` | Estimated | Estimated |
| `y <- x + 0` | Estimated | Fixed at 0 |
| `y <- x + 1` | Estimated | Fixed at 1 |
| `y <- x - 1` | Estimated | Fixed at -1 |
| `y <- -1*x` | Fixed at -1 | Estimated |
| `y <- -1*x-1` | Fixed at -1 | Fixed at -1 |

Whitespace does not change these meanings. Default estimation is already
[the linear-node behavior](../../../sem2/model/components/nodes.py#L134-L145);
numeric fixed-intercept syntax still needs implementation. This rule also governs
the outer linear equation in a mixed expression; it must not silently disable
offsets inside a neural network or introduce new neural centering rules.

**Keep the existing expansion and correct its semantics.** Reuse the graph/node
machinery rather than require composite mechanisms or user-written intermediates.
The agreed expansion of `y <- n(x) + z` is:

```text
y:0 <- n(x)
y   <- 1*y:0 + z
```

The [current mixed-sum expansion](../../../sem2/model/construction/formula_parser.py#L374-L388)
omits `1*`; fix that omission. The generated coefficient must be fixed at one,
not merely initialized to one; ordinary bare predictors such as `z` remain learned.
Add this coefficient only for lifted summands in linear sums. Nested network
inputs stay inputs: `n(n(x), z)` becomes a helper feeding `n(y:0, z)`.

**Keep helper identities private and collision-safe.** The positional labels can
stay: `n(x) + z` needs `y:0`; `z + n(x)` needs `y:1`; two nonlinear summands need
distinct helpers. Nested labels can extend the path, such as `y:1:0`. Reserve
the generated namespace (for example `:`) against user-declared model names and
reject duplicate definitions; positions alone do not ensure global uniqueness.
Do not merge repeated `n(x)` occurrences and inadvertently share their parameters.

Helpers are computation nodes, not data-column indices or user-declared latent
causes. Exclude them from ordinary observation matching, variable reporting, and
public intervention targets; users must not need to configure them by generated
name. Name-to-index lookup is separate—the [DAG already provides it](../../../sem2/model/construction/dag.py#L79-L82).

## Small end-to-end example

This target-API example uses default estimated intercepts. The new `fit` export
and objective-keyword spelling are illustrative; this is not an executed fit.

```python
import pandas as pd
from sem2 import fit  # new convenience entry point; illustrative export
from sem2.model import SEM2Linear

raw = pd.DataFrame({
    "measured x": [0.0, 1.0, 2.0, 3.0],
    "measured y": [1.0, 3.0, None, 7.0],
})
model = SEM2Linear(
    "ex, ey ~ Normal(0,1); "
    "measured_x <- 1*ex; measured_y <- measured_x + 1*ey"
)
result = fit(model, raw, objective="gaussian_nll")  # preparation is internal
```

Expected interface behavior: display the two renamings; use three complete rows
and report one dropped row. Both measured variables are observed; `ex` and `ey`
are unobserved. This illustrates the interface, not an executed fit or recovery test.

## Implementation and review checks

1. **Data preparation:** implement the shared wrapper and reports. Test reordered
   columns, valid-name preservation, punctuation/keywords/collisions, duplicate
   labels, correct row counts, no complete rows, and no mutation of the input.
   A different observed subset must not change the structural model; unknown
   columns must fail rather than disappear through silent intersection.
2. **Parser and nodes:** implement fixed helper coefficients and numeric fixed intercepts.
   Test single, multiple, repeated, and nested nonlinear terms; collisions;
   estimated versus zero/positive/negative fixed intercepts; negative fixed slopes;
   and whitespace variants. Compare outputs and gradients with direct evaluation
   using matched parameters and intercepts. Verify that fixed intercepts do not
   change and effective helper coefficients stay one during optimization while
   neural parameters still receive gradients, and helpers stay out of public views.
3. **Trainer and objectives:** route `fit` and direct Trainer use through one
   implementation. Pass aligned names/data to every supported objective and remove
   hard-coded names or column positions. Check equivalent results and preparation
   reports with matched settings; return inspectable training diagnostics.
4. **Discrete values and query boundary:** require exactly one matching state;
   reject unknown/ambiguous matches and invalid codes before integer conversion.
   Use `do_intervention` as the user-facing intervention name rather than
   `set_nodes`. Keep broader query signatures and treatment/outcome/instrument
   roles in their separate interface work, not the generic data contract.

Review the specification and acceptance checks for omissions or correctness
issues, and settle the non-pandas input policy before implementing that boundary.
Implementation PRs must include real test output and the completed
construction → preparation → fitting example.
Documentation/data checks and isolated parser probes are not full SEM2 fitting
or gradient tests. This handoff makes no implementation, merge, or release claim.
