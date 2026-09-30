# SEM2 contributor and agent guidance

SEM2 is a research-stage PyTorch implementation of structural equation models. Read [`README.md`](README.md) for stable project context and [`PROJECT.md`](PROJECT.md) for the current technical state, limitations, and work plan. Inspect the source and tests before acting when either document is stale; fresh evidence takes precedence.

## Source map

- `sem2/model/construction/formula_parser.py`: formula grammar, validation, and lowering to node specifications.
- `sem2/model/construction/dag.py`: graph validation, topological order, observed-variable list, and inferred model type.
- `sem2/model/components/nodes.py`: exogenous, linear, continuous MLP, and discrete MLP nodes.
- `sem2/model/models/base.py`: general continuous `SEM2` construction and sampling.
- `sem2/model/models/linear.py`: scalar linear path coefficients and model-implied moments.
- `sem2/model/models/discrete.py`: finite-state probability calculation and discrete interventions.
- `sem2/operators/linear.py`: Gaussian likelihood and covariance-based linear objectives.
- `sem2/operators/scalar_operators.py`: experimental energy and causal-query operators.
- `sem2/operators/vector_operators.py`: potential-outcome sampling.
- `sem2/operators/discrete_energy_loss.py`: finite-discrete energy calculations.
- `sem2/sem2trainer.py`: current package trainer; do not assume it is the sole or stable fitting path.
- `tests/`: parser, construction, sampling, and linear-model tests.
- `experiments/` and `notebooks/`: research code and demonstrations; some imports and dependencies are stale.

## Current API conventions

Use the supported model imports:

```python
from sem2.model import SEM2, SEM2Discrete, SEM2Linear, SEMType
```

Use public node configurations from `sem2.default_configs`, notably `LinearNodeConfig` and `MLPConfig`.

Formula statements are separated by semicolons:

- `u ~ Normal(0,1)` declares an exogenous PyTorch distribution.
- `y <- x + z` declares learned linear coefficients.
- `y <- 1*x + 0.5*z` fixes coefficients written explicitly.
- `y <- n(x,z)` declares a continuous MLP mechanism.
- `y <- d(x,z)` declares a finite discrete mechanism.
- `unobs(u)` marks variables as unobserved in the current implementation.

Call `model.sample(...)`; `forward(...)` deliberately raises `NotImplementedError`. Continuous sampling returns a node-to-tensor dictionary plus an auxiliary dictionary. Discrete sampling returns probabilities over `model.all_states` plus state metadata.

Pass `set_nodes` as a node-name-to-tensor dictionary with shape `(batch_size, node.out_dim)` for each value. Sampling updates cached node values and draws from the PyTorch RNG; it is not a pure inspection operation.

Pass node configurations through `node_cfgs`, keyed by the left-hand-side node name. Construction consumes entries from that dictionary; pass a copy if the caller needs to retain it. Treat tensor column order as unsafe unless it is checked against named model variables.

These conventions describe the current default-branch API. Named records, node-owned support, fixed formula intercepts, DataFrame preparation, and explicit objective composition are migration work, not capabilities to infer from planning documents or an unmerged implementation. Inspect the exact branch before documenting its API.

## Design conformance

Before changing model, data, or operator boundaries, read the agreed design direction in [`PROJECT.md`](PROJECT.md). Preserve compatibility with existing formulas, require named variable mappings at data boundaries, and keep application-specific roles out of graph-generic components.

For v0.1 work, start with the [implementation plan](docs/plan/v0.1.md) and its linked interface specifications, then check [`PROJECT.md`](PROJECT.md) for the current baseline and review status. Distinguish the agreed target from inspected behavior. Keep plan review separate from implementation review; an earlier task, prototype, or plan approval does not authorize the next unit. Do not infer quantitative meaning from numeric storage or arithmetic on category codes.

## Capability and claim discipline

Consult [`PROJECT.md`](PROJECT.md) before describing what the package currently supports. Claims in code, documentation, and pull requests must match inspected behavior. When functionality is incomplete, fail early with a precise message or label it experimental. Do not broaden a capability claim from a notebook, meeting report, or unmerged branch.

Keep two causal boundaries explicit: `set_nodes` computes an intervention under the supplied model, not evidence that the graph is valid or an effect is identified; and numerical extrema from a partial-identification search are not sharp bounds without fit tolerances, restarts, diagnostics, and an identification argument or independent reference.

## Development and verification

Install the editable package and run the suite from the repository root:

```bash
python -m pip install -e .
python -m pytest
```

For code changes:

1. extend suitable existing tests or add a few focused cases for distinct contract changes; respect an explicit request to defer new tests while still running relevant existing checks;
2. run the smallest relevant tests while iterating;
3. run the full suite before publication;
4. run `git diff --check`; and
5. report warnings, skipped checks, and missing dependencies rather than treating them as passes.

Estimator and operator changes should test analytical values, gradients, tensor axes, batch shapes, devices, reproducibility, and failure cases. Intervention tests should use explicit node names and state the model assumptions. Optimizer-based examples should report convergence information rather than only an iteration count.

For documentation changes, verify every repository-relative link and execute retained code examples when the environment has the declared dependencies.

## Documentation policy

This policy applies equally to human contributors and coding agents. Document every new or behaviorally changed public class, function, and method, plus internal helpers whose contracts or algorithms matter to correctness. Reconcile existing documentation when its behavior changes; a usage example alone is not API documentation.

- **In the source:** use concise NumPy-style docstrings immediately inside the class/function. State purpose, parameter and output meanings, relevant types/defaults, tensor shapes and axis meanings, essential assumptions, restrictions, side effects, randomness, and important errors as applicable. Explain constructor arguments and important attributes for classes. Do not merely repeat the name or hide a necessary precondition in an external chapter.
- **In `docs/`:** keep architecture, mathematical definitions and derivations, statistical assumptions, equation-to-code explanations, end-to-end workflows, and extended examples. Cross-link relevant API and explanatory text. Keep one authoritative explanation rather than copying it into several docstrings.
- **Keep it proportionate:** no arbitrary minimum length or compulsory long example/extended-summary section for every small helper. Separate documentation-only edits from behavioral edits in commits where practical, but review their consistency together in the same PR.
- **Keep it truthful:** describe inspected behavior, not intended behavior. Reviewed mathematics and API decisions define the intended contract; agent-written code, tests, and prose do not silently become its specification. Flag a conflict for a human decision rather than documenting a bug as the intended result.

For the current base `SEM2.sample`, a docstring could read as follows (docstring only, not a new implementation or the discrete subclass contract):

```python
"""Draw node values in parent-before-child order, with optional interventions.

Parameters
----------
batch_size : int, default 1
    Number of rows to draw.
set_nodes : dict of str to torch.Tensor, optional
    Values replacing named node equations. Each tensor must have shape
    (batch_size, node.out_dim) and valid values on the model's device.

Returns
-------
samples : dict of str to torch.Tensor
    Node values, each shaped (batch_size, node.out_dim).
auxiliary : dict
    Empty in this implementation.

Notes
-----
Uses the PyTorch RNG and updates cached node values. Interventions are
model calculations, not evidence of causal identification.
"""
```

### Automated checks and human review

The enforcement design is source-level docstring validation plus a strict documentation build and required human review. Consult [the implementation status](PROJECT.md#documentation-policy-and-enforcement) before claiming any check is configured or passing.

1. Use a small selected set of `numpydoc` checks for missing docstrings and missing, obsolete, or undescribed parameters/returns. Check source coverage as well as generated pages so omitted API objects cannot escape validation. Scope exceptions explicitly; do not use formatting rules to demand filler.
2. Build the Sphinx API reference and explanatory pages with warnings treated as errors and unresolved cross-references reported. Local pre-commit checks are useful feedback, not a substitute for CI checks on the PR. Confirm actual configuration and commands before claiming a build.
3. In each PR, identify changed interfaces/behavior, the updated documentation (or why none was needed), and the checks actually executed, including limitations. Human reviewers check semantic and mathematical accuracy, shapes, assumptions, completeness, and concision. A second AI review is advisory, not approval.
4. Require passing documentation checks and approval from designated human reviewers before default-branch integration; require renewed review after substantive changes. Protect policy files, checker configuration, exclusions, workflows, and ownership rules with human review too. Agents must not weaken checks to make their changes pass, approve their own work, or bypass review requirements. Administrative configuration requires a separate authorized change; writing this rule does not enable it.
5. Account for every review comment and follow-up, including comments on outdated lines. Link changes and executed evidence back to the request; distinguish answered, implemented-awaiting-review, blocked, and explicitly deferred items. Leave thread resolution and merge to the human reviewer.

Reference behavior: [numpydoc validation](https://numpydoc.readthedocs.io/en/latest/validation.html), [Sphinx build options](https://www.sphinx-doc.org/en/master/man/sphinx-build.html), and [GitHub branch protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/managing-a-branch-protection-rule).

## Research integration

Bring external methods into bounded review branches with:

- the exact objective and assumptions;
- dependencies and configuration;
- supported model classes, tensor shapes, and devices;
- deterministic examples with seeds;
- analytical or independent-reference tests;
- benchmark acceptance criteria; and
- a clear separation between empirical observations and package guarantees.

Keep generic infrastructure separate from application-specific workflows. In particular, binary instrumental-variable adapters should be named as such rather than hidden inside general losses.

## Repository memory

- `README.md` contains stable, user-facing context and the supported entry points.
- `PROJECT.md` records current capabilities, open problems, evidence, and the agreed order of work.
- `AGENTS.md` contains durable contribution and verification guidance.
- `docs/meetings/` contains dated historical records; do not rewrite them to represent later implementation status.

Update the relevant memory file when an interface, capability, limitation, or agreed priority changes. Preserve evidence and do not invent project facts. Current code and executed tests outrank stale prose.
