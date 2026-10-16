# SEM2

SEM2 is a research-stage PyTorch package for expressing structural equation models as formulas, compiling them into executable directed acyclic graphs, and fitting or querying the resulting models.

The project is designed around user-supplied causal structures rather than a fixed catalogue of study designs. It supports model-implied interventions, but observational fit alone does not establish that a graph is causal or that an effect is identified.

## Current status

See [`PROJECT.md`](PROJECT.md) for the maintained capability matrix, known limitations, design decisions, and work plan.

Detailed plans and research records are maintained in the private [SEM2 documentation repository](https://github.com/causal-design/sem2-docs). Installing, using, and testing this package does not require access to it.

## Installation

SEM2 requires Python 3.11 or later.

```bash
python -m pip install -e .
```

The package dependencies are declared in [`pyproject.toml`](pyproject.toml). The repository does not yet publish a stable release, so development should use an editable checkout.

## Minimal example

```python
from sem2.default_configs import LinearNodeConfig
from sem2.model.models.linear import SEM2Linear

model = SEM2Linear(
    "ex, ey ~ Normal(0,1); "
    "x <- 1*ex; "
    "y <- x + 1*ey",
    node_cfgs={
        "x": LinearNodeConfig(use_intercept=False),
        "y": LinearNodeConfig(use_intercept=False),
    },
)

samples, _ = model.sample(batch_size=100)
print(samples.select("x").shape)
print(samples.select("y").shape)
print(model.cov_matrix(nodes=["x", "y"]))
```

A model is sampled through `sample(...)`; calling the model directly is not part of the current API. The supported model imports are:

```python
from sem2.model.models.base import SEM2
from sem2.model.models.discrete import SEM2Discrete
from sem2.model.models.linear import SEM2Linear
from sem2.model.construction.sem_types import SEMType
```

## Formula language

Statements are separated by semicolons.

| Form | Meaning |
|---|---|
| `u ~ Normal(0,1)` | Declare an exogenous variable and its PyTorch distribution. |
| `y <- x + z` | Declare a linear structural equation with learned coefficients. |
| `y <- 1*x + 0.5*z` | Fix selected linear coefficients in the formula. |
| `y <- n(x,z)` | Use a configurable MLP for a continuous mechanism. |
| `y <- d(x,z; levels=2)` | Use a finite discrete mechanism with two states. |

Observed variables are determined by the data supplied to fitting and operators, rather than an `unobs(...)` clause in the structural formula. Select covariance variables explicitly by name, as in the example above.

Node configuration is passed through `node_cfgs`, keyed by the variable on the left-hand side of the formula. Public node configurations live in [`sem2/default_configs.py`](sem2/default_configs.py).

## Interventions and causal interpretation

`set_nodes` replaces selected node values while the remaining equations are evaluated:

```python
import torch
from sem2.data.records import DataBatch

n = 100
samples_do_x1, _ = model.sample(
    batch_size=n,
    set_nodes=DataBatch(torch.ones(n, 1), ["x"]),
)
```

This computes the intervention implied by the specified structural model. It does not validate the graph, exogenous assumptions, identification, or transportability. Causal results should state those assumptions and distinguish model calculations from conclusions supported by observed data.

## Repository layout

- [`sem2/model/construction/`](sem2/model/construction/) contains the formula parser, typed syntax, DAG construction, and model classification.
- [`sem2/model/components/`](sem2/model/components/) contains executable node and neural-network modules.
- [`sem2/model/models/`](sem2/model/models/) contains the general, linear, and finite discrete model classes.
- [`sem2/operators/`](sem2/operators/) contains likelihood, distributional-loss, and causal-query operators.
- [`sem2/trainer/trainer.py`](sem2/trainer/trainer.py) contains the package trainer.
- [`tests/`](tests/) contains the pytest suite.
- [`experiments/`](experiments/) and [`notebooks/`](notebooks/) contain research and demonstration code, some of which uses older APIs.

## Testing

Run the full suite from the repository root:

```bash
python -m pytest
```

When changing an estimator or causal operator, add tests for numerical values, gradients, tensor shapes, intervention semantics, and failure cases. A passing parser or sampler test does not validate an end-to-end fitting claim.

## Project records

- [`PROJECT.md`](PROJECT.md) is the canonical record of current work, known limitations, and priorities.
- [`AGENTS.md`](AGENTS.md) contains durable guidance for contributors and coding agents.
- Meeting records, reviews, reports, and detailed plans are kept in the private documentation repository linked above.

## License

SEM2 code is licensed under the GNU General Public License, version 3 only (`GPL-3.0-only`). See [`LICENSE`](LICENSE) for the full terms. This declaration does not license the separately maintained private research records.
