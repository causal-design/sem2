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
- `sem2/sem2trainer.py`: current package trainer; do not assume it is the sole or stable fitting path.
- `tests/`: parser, construction, sampling, and linear-model tests.
- `experiments/` and `notebooks/`: research code and demonstrations; some imports and dependencies are stale.

`DESIGN.md` describes an older architecture. Use it as historical context, not as the current API specification.

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

Pass node configurations through `node_cfgs`, keyed by the left-hand-side node name. Treat tensor column order as unsafe unless it is checked against named model variables.

## Design direction

Keep these distinctions explicit in new work:

- Exogenous or endogenous status belongs to the structural model.
- Observed or latent status belongs to a dataset's view of that model.
- Treatment, outcome, and intervention roles belong to a statistical or causal query.
- A graph-generic operator must not silently encode binary-IV names or column positions.

The current `unobs(...)` syntax combines structural and observation concerns for backward compatibility. New data-layer work should move toward an explicit dataset schema and named column-to-node mapping without silently breaking existing formulas.

## Capability boundaries

Claims in code, documentation, and pull requests must match inspected behavior:

- Formula parsing, DAG construction, continuous sampling, linear moment algebra, Gaussian linear likelihood, and finite-state model probabilities are implemented.
- The generic continuous energy path and package trainer are experimental and need direct end-to-end tests.
- The finite discrete fitting workflow is specialized and incomplete.
- Mixed discrete/continuous fitting is not supported end to end.
- MMD and PFN posterior approximation are not implemented in the package.
- Sinkhorn and remaining discrete research implementations discussed by the team are external until reviewed and merged.
- `set_nodes` computes an intervention under the supplied model. It does not establish causal identification or validate the graph.
- Numerical extrema from a partial-identification search are not sharp bounds without fit tolerances, restarts, diagnostics, and an identification argument or independent reference.

When functionality is incomplete, fail early with a precise message or label it experimental. Do not broaden a capability claim from a notebook, meeting report, or unmerged branch.

## Development and verification

Install the editable package and run the suite from the repository root:

```bash
python -m pip install -e .
python -m pytest
```

For code changes:

1. add focused tests next to the behavior being changed;
2. run the smallest relevant tests while iterating;
3. run the full suite before publication;
4. run `git diff --check`; and
5. report warnings, skipped checks, and missing dependencies rather than treating them as passes.

Estimator and operator changes should test analytical values, gradients, tensor axes, batch shapes, devices, reproducibility, and failure cases. Intervention tests should use explicit node names and state the model assumptions. Optimizer-based examples should report convergence information rather than only an iteration count.

For documentation changes, verify every repository-relative link and execute retained code examples when the environment has the declared dependencies.

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
