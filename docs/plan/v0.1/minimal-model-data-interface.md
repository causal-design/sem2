# 1. Minimal model–data interface

**F02/F03 handoff for the [v0.1 plan](../v0.1.md).** This records the latest
thread feedback. Public API spellings below are proposals, not implemented features.

## Working direction

- **Use pandas, with direct name matching.** Prepared DataFrame columns name model
  variables. Users select the relevant raw columns; unexpected columns in the
  prepared table are errors. No user-supplied model–column mapping or public
  binding object is needed.
- **Infer observation status jointly from model and data.** Supplied columns are
  observed; remaining user-declared model variables are unobserved. No separate
  `unobs(...)` or observed-variable declaration. Missing cells do not change a
  variable's status. Unsupported model/data/objective combinations fail clearly.
- **v0.1 missing-data policy: complete cases only.** Fit rows complete across all
  selected observed variables. Report input, retained, and dropped row counts;
  never drop rows silently. Unselected metadata must not affect row selection.
  Fail if no complete rows remain. More advanced missing-data handling is deferred
  until **after v0.2**. Complete-case fitting is not a guarantee against
  missingness-related selection bias.
- **Provide automatic column-name conversion.** Produce unique, valid Python
  identifiers, excluding keywords. Preserve already-valid names; handle spaces,
  punctuation, leading digits, keywords, and collisions deterministically without
  merging columns. Show original and converted names so users can write the model
  with the converted names. This is automatic preprocessing/provenance, not an
  additional user-authored mapping. Reject ambiguous duplicate source labels.
- **Keep fitting internals small.** Derive an ordered observed-name list, its
  name-to-column-index dictionary, and an aligned observation tensor. Validate
  values, shape, dtype, and device; convert once before optimization. The model,
  objective, and trainer retain their separate jobs. No richer schema is required.

## Intercepts and internal helper names

**Say “intercept,” not “bias,” in the user-facing SEM interface.** Ordinary linear
examples should estimate intercepts without a `node_cfgs` block; that is already
[the linear-node default](../../../sem2/model/components/nodes.py#L134-L145).
The earlier zero-intercept example simply matched its zero-mean data generator;
it was not a general SEM requirement. Provide concise intercept control rather
than requiring per-node configuration objects. A formula-level `+1` / `+0`
convention is one proposal to confirm, **not current syntax**.

**`y:0` is a parser temporary, not a data-column index.** The current
[formula parser](../../../sem2/model/construction/formula_parser.py#L342-L388)
introduces nodes for intermediate nonlinear expressions, such as the `n(x)` part
of `y <- n(x) + z`. These helpers must stay internal, not appear as user-declared
latent variables or require matching data columns. A dictionary is sufficient for
name-to-index lookup—the [DAG already has one](../../../sem2/model/construction/dag.py#L79-L82)—
but it serves a different purpose from representing intermediate computations.
F02/F03 does not require redesigning that parser representation.

## Small end-to-end example

`normalize_column_names` and `fit` below are **proposed entry points**. Their exact
signatures remain to be agreed. The example uses default estimated intercepts.

```python
import pandas as pd
from sem2.model import SEM2Linear

raw = pd.DataFrame({
    "measured x": [0.0, 1.0, 2.0, 3.0],
    "measured y": [1.0, 3.0, None, 7.0],
})
data = normalize_column_names(raw)  # proposed: measured_x, measured_y

model = SEM2Linear(
    "ex, ey ~ Normal(0,1); "
    "measured_x <- 1*ex; measured_y <- measured_x + 1*ey"
)
result = fit(model, data, objective="gaussian_nll")  # proposed
```

Expected interface behavior: display the two renamings; use three complete rows
and report one dropped row. Both measured variables are observed; `ex` and `ey`
are unobserved. This illustrates the interface, not an executed fit or recovery test.

## Implementation handoff and remaining choices

1. Implement the name converter, complete-case policy, reporting, and shared
   name validation. Confirm converter naming and collision rules before coding.
2. Pass derived names and aligned data through each supported fitting objective;
   remove assumptions about fixed column positions or hard-coded variable names.
3. Confirm the simpler intercept syntax and the proposed `set_nodes` →
   `do_intervention` rename. Keep treatment/outcome/instrument role arguments for
   later design-specific interfaces, not the generic data contract.
4. Fix discrete state matching: require exactly one matching state; reject unknown
   or ambiguous matches and invalid codes **before** integer conversion. Merely
   replacing `argmax` is insufficient.
5. Acceptance checks: column reordering preserves alignment; name conversion
   handles collisions without overwriting valid names; row counts and complete-case
   filtering are correct; all-missing/no-complete-row inputs fail clearly; another
   observed subset leaves the model unchanged; the example runs with estimated
   intercepts; invalid discrete states cannot silently select the first state.
