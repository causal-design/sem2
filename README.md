# SEM2

SEM2 is a research-stage PyTorch package for expressing structural equation models as formulas, compiling them into executable directed acyclic graphs, and fitting or querying the resulting models.

The project is designed around user-supplied causal structures rather than a fixed catalogue of study designs. It supports model-implied interventions, but observational fit alone does not establish that a graph is causal or that an effect is identified.

## Current status

See [`PROJECT.md`](PROJECT.md) for the maintained capability matrix, known limitations, design decisions, and work plan.

## Installation

SEM2 requires Python 3.10 or later.

```bash
python -m pip install -e .
```

The package dependencies are declared in [`pyproject.toml`](pyproject.toml). The repository does not yet publish a stable release, so development should use an editable checkout.

## Minimal example

```python
from sem2.default_configs import LinearNodeConfig
from sem2.model import SEM2Linear

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
print(samples.select("x").shape)
print(samples.select("y").shape)
print(model.cov_matrix(nodes=model.observed_vars))
```

A model is sampled through `sample(...)`; calling the model directly is not part of the current API. The supported model imports are:

```python
from sem2.model import SEM2, SEM2Discrete, SEM2Linear, SEMType
```

## Named data and sampling results

- **Which format carries observations, interventions, and sampling results?**

  **Chosen action:** Use `NamedBatch` for data rows and `FiniteDistribution` for finite-state probability results.

  **Alternative considered:** Retain positional tensors and dictionary data interfaces. These are no longer accepted at the implemented data boundaries.

  **Explanation:** `SEM2.sample` and `SEM2Linear.sample` return `(NamedBatch, auxiliary)`. A batch pairs its `.values` tensor with `.variables` metadata and `.names`. Use `select("x", "y")` to obtain tensor columns in that order, or `subset("x", "y")` to retain a named batch. A variable can occupy several contiguous columns, recorded by `VariableInfo.width`.

  `SEM2Discrete.sample` instead returns `(FiniteDistribution, auxiliary)`. Its `.states` is a `NamedBatch`, and `.probabilities` has axes `[exogenous draws, states]`. `marginal(...)` combines state probabilities for selected variables; averaging over exogenous draws is a separate operation. Both sampler paths accept a `NamedBatch` for `set_nodes` and record it in `auxiliary["set_nodes"]` (or `None` when there is no intervention).

  These records are the current low-level interface, not the planned DataFrame preparation API. Metadata distinguishes category labels, stored state codes, and quantitative meaning; integer dtype alone does not establish meaningful scores. The [plan in PR #54](https://github.com/causal-design/sem2/pull/54) records the decisions and unresolved numeric-column policy. A [short code-change summary](docs/implementation/named-data-changes.md) describes this implementation.

The [Dummy training example](experiments/trainer.py) uses the canonical package trainer:

```bash
python -m experiments.trainer
```

Its short run demonstrates named-data integration, not convergence or validated causal bounds.

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

`unobs(...)` currently determines the model's observed-variable list. See [`PROJECT.md`](PROJECT.md) for the proposed separation between structural formulas and dataset-specific observation status.

Node configuration is passed through `node_cfgs`, keyed by the variable on the left-hand side of the formula. Public node configurations live in [`sem2/default_configs.py`](sem2/default_configs.py).

## Interventions and causal interpretation

`set_nodes` replaces selected node values while the remaining equations are evaluated:

```python
import torch
from sem2.data import NamedBatch

n = 100
intervention = NamedBatch(torch.ones(n, 1), samples.subset("x").variables)
samples_do_x1, _ = model.sample(
    batch_size=n,
    set_nodes=intervention,
)
torch.testing.assert_close(samples_do_x1.select("x"), torch.ones(n, 1))
```

This computes the intervention implied by the specified structural model. It does not validate the graph, exogenous assumptions, identification, or transportability. Causal results should state those assumptions and distinguish model calculations from conclusions supported by observed data.

## Repository layout

- [`sem2/model/construction/`](sem2/model/construction/) contains the formula parser, typed syntax, DAG construction, and model classification.
- [`sem2/model/components/`](sem2/model/components/) contains executable node and neural-network modules.
- [`sem2/model/models/`](sem2/model/models/) contains the general, linear, and finite discrete model classes.
- [`sem2/data.py`](sem2/data.py) contains the named data and finite-distribution records; [`sem2/dataloader.py`](sem2/dataloader.py) supplies the synthetic Dummy loader.
- [`sem2/operators/`](sem2/operators/) contains likelihood, distributional-loss, and causal-query operators.
- [`sem2/sem2trainer.py`](sem2/sem2trainer.py) contains the canonical trainer used by the named-data example; fitting remains experimental.
- [`tests/`](tests/) contains the pytest suite.
- [`experiments/`](experiments/) and [`notebooks/`](notebooks/) contain research and demonstration code, some of which uses older APIs.
- [`docs/meetings/`](docs/meetings/) contains dated reviews and meeting records.
- [`docs/reports/`](docs/reports/) contains the project research reports.

## Testing

Run the full suite from the repository root:

```bash
python -m pytest
```

When changing an estimator or causal operator, add tests for numerical values, gradients, tensor shapes, intervention semantics, and failure cases. A passing parser or sampler test does not validate an end-to-end fitting claim.

## Project records

- [`PROJECT.md`](PROJECT.md) is the canonical record of current work, known limitations, and priorities.
- [`AGENTS.md`](AGENTS.md) contains durable guidance for contributors and coding agents.
- [`docs/meetings/2026-08-26-meeting.md`](docs/meetings/2026-08-26-meeting.md) is the concise meeting record.
- [`docs/meetings/2026-08-26-ai-review-updated.md`](docs/meetings/2026-08-26-ai-review-updated.md) is the detailed review updated after that meeting.
- [`docs/meetings/2026-08-26-ai-review.md`](docs/meetings/2026-08-26-ai-review.md) preserves the original review that preceded the meeting update.
