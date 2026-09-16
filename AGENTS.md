# SEM2 contributor and agent guidance

SEM2 is a research-stage PyTorch implementation of structural equation models. Read [`README.md`](README.md) for stable project context and [`PROJECT.md`](PROJECT.md) for the current technical state, limitations, and work plan. Inspect the source and tests before acting when either document is stale; fresh evidence takes precedence.

## Source map

- `sem2/model/construction/formula_parser.py`: formula grammar, validation, and lowering to node specifications.
- `sem2/model/construction/dag.py`: graph validation, topological order, observed-variable list, and inferred model type.
- `sem2/model/components/nodes.py`: exogenous, linear, continuous MLP, and discrete MLP nodes.
- `sem2/model/models/base.py`: general continuous `SEM2` construction and sampling.
- `sem2/model/models/linear.py`: scalar linear path coefficients and model-implied moments.
- `sem2/model/models/discrete.py`: finite-state probability calculation and discrete interventions.
- `sem2/data.py`: `VariableInfo`, `NamedBatch`, and `FiniteDistribution` records.
- `sem2/dataloader.py`: synthetic Dummy loader yielding named batches, not a production DataFrame parser.
- `sem2/operators/linear.py`: Gaussian likelihood and covariance-based linear objectives.
- `sem2/operators/scalar_operators.py`: experimental energy and causal-query operators.
- `sem2/operators/vector_operators.py`: named potential-outcome results using the model samplers.
- `sem2/operators/discrete_energy_loss.py`: named observations versus finite-distribution predictions.
- `sem2/sem2trainer.py`: canonical trainer for the named-data integration; `experiments/trainer.py` delegates to it. Fitting remains experimental.
- `tests/`: parser, construction, sampling, and linear-model tests.
- `tests/test_named_data.py`: named alignment, record semantics, operator, and trainer regressions.
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

### Data and result conventions

- **Which forms should producers and consumers use?**

  **Chosen action:** Use `NamedBatch` for observations, generated draws, and interventions; use `FiniteDistribution` for finite-state probability results.

  **Alternative considered:** Preserve legacy tensor inputs, dictionary interventions, or old sampler returns through adapters. Do not restore those compatibility paths.

  **Explanation:** Call `model.sample(...)`; `forward(...)` deliberately raises `NotImplementedError`. `SEM2` and `SEM2Linear` return `(NamedBatch, auxiliary)`. `SEM2Discrete` returns `(FiniteDistribution, auxiliary)`, pairing named states with probabilities `[exogenous draws, states]`. Samplers and `PotentialOutcome` share these result records. Intervention provenance is `auxiliary["set_nodes"]`, a `NamedBatch` or `None`, not a node-to-tensor dictionary. The old raw `states`/`full_state` auxiliary payloads are not part of this interface.

- **How should operators interpret names, dimensions, and metadata?**

  **Chosen action:** Align by name, retain variable widths and metadata, and keep numerical meaning explicit.

  **Alternative considered:** Rely on column positions or treat all variables as scalar columns. Neither is a substitute for the named contract.

  **Explanation:** Use `subset(...)` when passing a batch onward and `select(...)` when a numerical calculation needs ordered tensor columns. Preserve shared descriptors through slicing and transfer. `VariableInfo.width` represents the contiguous block for an existing vector-valued node; scalar mean/ATE operators require scalar roles. `FiniteDistribution.marginal(...)` sums probability mass over omitted state coordinates; averaging the exogenous-draw axis is separate.

  Metadata describes stored representation, not a replacement for node-level support. Keep external labels, integer state codes, and deliberate numeric scores distinct. Do not infer categorical or quantitative meaning from dtype alone. Validate discrete values before coercion, and do not silently substitute scores for state codes expected by the finite-state engine. Ordinary tensor computations inside kernels, parameters, and scalar outputs remain appropriate.

  `DiscreteEnergyLoss(p).energy_score(observations, prediction)` and `.energy_dist(observations, prediction)` take a `NamedBatch` and a `FiniteDistribution`; the prediction owns its state table. Do not reintroduce the extra legacy batch axis or `p_true_mean` entry path. Keep observation and model sample counts separate in pair corrections.

Pass node configurations through `node_cfgs`, keyed by the left-hand-side node name. Use the [plan in PR #54](https://github.com/causal-design/sem2/pull/54) for the decisions and unresolved numeric-column discussion, and the [short code-change summary](docs/implementation/named-data-changes.md) for this implementation's scope.

## Design conformance

Before changing model, data, or operator boundaries, read the agreed design direction in [`PROJECT.md`](PROJECT.md). Preserve existing formulas, use the named-record contract at implemented data boundaries, and keep application-specific roles out of graph-generic components. The clean break concerns data formats, not unrelated formula or Python-version compatibility. The planned [DataFrame preparation layer](docs/plan/v0.1/minimal-model-data-interface.md) is separate work; do not claim that these low-level records implement its public fitting interface.

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

Keep generic infrastructure separate from application-specific workflows. In particular, the IV-specific discrete objective requires an explicit `TrainingConfig.instrument_node`; do not infer an instrument from column position or assume binary levels. Its role checks do not establish exogeneity or causal identification.

## Repository memory

- `README.md` contains stable, user-facing context and the supported entry points.
- `PROJECT.md` records current capabilities, open problems, evidence, and the agreed order of work.
- `AGENTS.md` contains durable contribution and verification guidance.
- `docs/meetings/` contains dated historical records; do not rewrite them to represent later implementation status.

Update the relevant memory file when an interface, capability, limitation, or agreed priority changes. Preserve evidence and do not invent project facts. Current code and executed tests outrank stale prose.
