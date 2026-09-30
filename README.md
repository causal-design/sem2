# SEM2

SEM2 is a research-stage PyTorch package for expressing structural equation models as formulas, compiling them into executable directed acyclic graphs, and fitting or querying the resulting models.

The project is designed around user-supplied causal structures rather than a fixed catalogue of study designs. It supports model-implied interventions, but observational fit alone does not establish that a graph is causal or that an effect is identified.

## Current status

See [`PROJECT.md`](PROJECT.md) for the maintained capability matrix, known limitations, design decisions, and work plan.

The [v0.1 implementation plan](docs/plan/v0.1.md) links the model–data, support/named-data, fitting/diagnostics, and identification specifications. These describe the target, not capabilities supplied by this checkout. The examples below use the current default-branch API; see `PROJECT.md` for the distinction between merged code, unmerged implementation work, and remaining reviews.

## Installation

SEM2 requires Python 3.10 or later.

```bash
python -m pip install -e .
```

The package dependencies are declared in [`pyproject.toml`](pyproject.toml). Use an editable checkout for development; interfaces remain research-stage and may change during reviewed development.

## Minimal example

```python
import torch
from sem2.default_configs import LinearNodeConfig
from sem2.model import SEM2Linear

torch.manual_seed(0)
model = SEM2Linear(
    "ex, ey ~ Normal(0,1); "
    "x <- 1*ex; "
    "y <- x + 1*ey; "
    "unobs(ex,ey)",
    node_cfgs={
        "x": LinearNodeConfig(use_bias=False),
        "y": LinearNodeConfig(use_bias=False),
    },
)

samples, _ = model.sample(batch_size=100)
print(samples["x"].shape)
print(samples["y"].shape)
print(model.cov_matrix(nodes=model.observed_vars))
```

This samples an initialized model; it does not fit the learned coefficient on `x`. Continuous/linear sampling returns a node-to-tensor dictionary plus an auxiliary dictionary. Each scalar node's tensor has shape `(batch_size, 1)`. Discrete sampling instead returns a probability tensor with separate exogenous-draw and enumerated-state axes, together with auxiliary state information; it is not a batch of realized category draws.

A model is sampled through `sample(...)`; calling the model directly is not part of the current API. Sampling consumes randomness and updates cached node values. The supported model imports are:

```python
from sem2.model import SEM2, SEM2Discrete, SEM2Linear, SEMType
```

## Formula language

Statements are separated by semicolons.

| Form | Meaning |
|---|---|
| `u ~ Normal(0,1)` | Declare an exogenous variable and its PyTorch distribution. |
| `y <- x + z` | Declare a linear structural equation with learned coefficients. |
| `y <- 1*x + 0.5*z` | Fix selected linear coefficients in the formula. |
| `y <- n(x,z)` | Use a configurable MLP for a continuous mechanism. |
| `y <- d(x,z)` | Use a finite discrete mechanism. |
| `unobs(u)` | Mark variables as unobserved in the current API. |

`unobs(...)` currently determines the model's observed-variable list. The agreed v0.1 target instead derives observation status from a supplied DataFrame; it is not implemented on this default-branch baseline. Use `LinearNodeConfig(use_bias=False)` to disable the current default linear bias; the planned standalone numeric-intercept syntax is not a current API guarantee.

Node configuration is passed through `node_cfgs`, keyed by the variable on the left-hand side of the formula. Construction consumes entries from this dictionary; pass a copy if retaining it. Public node configurations live in [`sem2/default_configs.py`](sem2/default_configs.py).

## Interventions and causal interpretation

`set_nodes` replaces selected node values while the remaining equations are evaluated. Supply valid tensors with one row per draw and the node's output width, on the model's device. Continuing the example above:

```python
import torch

n = 100
samples_do_x1, _ = model.sample(
    batch_size=n,
    set_nodes={"x": torch.ones(n, 1)},
)
```

This computes the intervention implied by the specified structural model. It does not validate the graph, exogenous assumptions, identification, or transportability. Causal results should state those assumptions and distinguish model calculations from conclusions supported by observed data.

## Repository layout

- [`sem2/model/construction/`](sem2/model/construction/) contains the formula parser, typed syntax, DAG construction, and model classification.
- [`sem2/model/components/`](sem2/model/components/) contains executable node and neural-network modules.
- [`sem2/model/models/`](sem2/model/models/) contains the general, linear, and finite discrete model classes.
- [`sem2/operators/`](sem2/operators/) contains likelihood, distributional-loss, and causal-query operators.
- [`sem2/sem2trainer.py`](sem2/sem2trainer.py) contains the current package trainer; it remains experimental.
- [`tests/`](tests/) contains the pytest suite.
- [`experiments/`](experiments/) and [`notebooks/`](notebooks/) contain research and demonstration code, some of which uses older APIs.
- [`docs/meetings/`](docs/meetings/) contains dated reviews and meeting records.
- [`docs/reports/`](docs/reports/) contains the project research reports.

## Testing

Run the full suite from the repository root:

```bash
python -m pytest
```

When changing an estimator or causal operator, extend focused checks for numerical values, gradients, tensor shapes, intervention semantics, and failure cases, reusing suitable existing tests. A passing parser or sampler test does not validate an end-to-end fitting claim.

## Documentation and contribution policy

Every new or behaviorally changed public class/function/method, and important internal helper, needs an accurate local docstring—not just a usage example. Keep purpose, inputs, outputs, shapes, and essential assumptions beside the code; put extended mathematics, architecture, and workflows in `docs/` and cross-link them. Update documentation in the same PR as the behavior, without forcing long examples or boilerplate onto trivial helpers.

The [contributor and agent policy](AGENTS.md#documentation-policy) defines the workflow: source-level docstring checks, a strict documentation build, evidence of the checks actually run, and human review of meaning. Changes to those checks or rules also require human review; AI-generated prose and passing checks are not proof of scientific correctness. See [enforcement status](PROJECT.md#documentation-policy-and-enforcement) for what is configured versus still to implement.

## Project records

- [`PROJECT.md`](PROJECT.md) is the canonical record of current work, known limitations, and priorities.
- [`AGENTS.md`](AGENTS.md) contains durable guidance for contributors and coding agents.
- [`docs/meetings/2026-08-26-meeting.md`](docs/meetings/2026-08-26-meeting.md) is the concise meeting record.
- [`docs/meetings/2026-08-26-ai-review-updated.md`](docs/meetings/2026-08-26-ai-review-updated.md) is the detailed review updated after that meeting.
- [`docs/meetings/2026-08-26-ai-review.md`](docs/meetings/2026-08-26-ai-review.md) preserves the original review that preceded the meeting update.
