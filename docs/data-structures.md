# Data structures and model nodes

A **pandas DataFrame** holds observations. A **model node** describes how one
variable is generated. The small records between them carry named tensors and
metadata; they are not additional models.

## What each object represents

| Object | Contents and role | Implementation |
|---|---|---|
| `pandas.DataFrame` | User-facing table: rows are observations, columns are named scalar variables. It can retain categorical labels, declared levels and ordering. | Consumed by [`prepare_data`](../sem2/data/preparation.py) and [`SEM2Trainer`](../sem2/trainer/trainer.py). |
| `VariableInfo` | One variable's name, component dimension, categorical status, label encoding, optional numerical scores and ordering. No observations or mechanism. | [`sem2/data/records.py`](../sem2/data/records.py) |
| `DataBatch` | A two-dimensional tensor `values` plus ordered `VariableInfo` objects. Shape: `[rows, sum(variable.dim)]`. | [`sem2/data/records.py`](../sem2/data/records.py) |
| `ScalarResult` | One named zero-dimensional tensor `value`, with optional associated variable metadata. Used for a loss or scalar query result. | [`sem2/data/records.py`](../sem2/data/records.py) |
| `SEM2Node` | A graph component: name, parents, structural mechanism or exogenous distribution, output dimension and current computed value. It is a PyTorch module; trainable mechanisms own parameters. | [`sem2/model/components/nodes.py`](../sem2/model/components/nodes.py) |
| `RandomVariableSupport` | A node's declared domain: a membership constraint and, when available, an enumeration. It is neither the observed unique values nor category scores. | [`node.support`](../sem2/model/components/nodes.py) |
| `DiscreteJointDistribution` | A finite joint state table (`DataBatch`) and probabilities indexed by exogenous draw and state. A computed probability law, not observations or a graph node. | [`sem2/model/distributions.py`](../sem2/model/distributions.py) |

A `SEM2`, `SEM2Linear` or `SEM2Discrete` model assembles nodes into a graph;
`model.node_dict["y"]` is the node for `y`. A node's cached `value` belongs to a
particular evaluation and is not the training dataset. `VariableInfo("y")`
only describes a column's representation: it is not a reference to that node.
Their shared **name** provides the matching key, while the node supplies its
dimension and domain.

## From a DataFrame to fitting

1. Pass the model, DataFrame and explicit scalar objective to `SEM2Trainer`, or
   use `fit`. Ordinary fitting does **not** require constructing a `DataBatch`.
2. `prepare_data` copies the frame, normalizes column names, rejects unknown or
   helper names and name collisions, and drops rows missing any supplied column.
   It encodes categories, checks domains, and orders supplied columns by the
   model's topological order. It returns a `DataBatch` and a preparation report;
   it does not build a loader or fit parameters.
3. The trainer puts the values in a PyTorch `TensorDataset`/`DataLoader`, retaining
   metadata separately. Each tensor minibatch is wrapped as a `DataBatch` for
   `objective.evaluate(batch)`.
4. A scalar objective returns `(ScalarResult, metrics)`. The trainer differentiates
   `result.value` and updates the same model. Metadata is not optimized.

Which variables are observed is determined by this frame–model pairing: the
preparation report lists supplied and absent model variables. It is not a new
property written onto each node. DataFrame fitting is currently scalar-only;
`VariableInfo.dim` also preserves existing low-level vector blocks. A scalar
categorical variable with three levels still has `dim=1`, not `dim=3`.

This runnable example shows the boundary without running an optimizer:

```python
import pandas as pd
import torch

from sem2.data.preparation import prepare_data
from sem2.data.records import DataBatch, ScalarResult
from sem2.model.models.linear import SEM2Linear
from sem2.operators.divergence import WeightedFrobeniusDistance

model = SEM2Linear("u,v ~ Normal(0,1); x <- 1*u + 0; y <- x + 1*v + 0")
frame = pd.DataFrame({"y": [1., 2., 3.], "x": [0., 1., 2.]})
batch, report = prepare_data(model, frame)
assert batch.names == ("x", "y")  # Matched by name, not incoming column position.
assert batch.values.shape == (3, 2)
node = model.node_dict["y"]        # Mechanism with parameters and declared support.
info = batch.variables[1]         # Representation metadata, not the mechanism.
assert info.name == node.name and info.dim == node.out_dim

objective = WeightedFrobeniusDistance(
    model, observed_vars=["x", "y"], weight=torch.tensor([[1., .5], [.5, 1.]]),
)
result, metrics = objective.evaluate(batch)
assert isinstance(result, ScalarResult) and result.value.ndim == 0
result.value.backward()           # The scalar retains its gradient graph.
```

`WeightedFrobeniusDistance` obtains the model covariance from
[`Covariance`](../sem2/operators/covariance.py), selects the named observed block,
and compares it with the batch covariance. `Covariance.evaluate()` itself
returns a **matrix tensor** and ordered variable metadata, not a `DataBatch` or
`ScalarResult`. The discrepancy is a weighted **sum of squared** residuals.
Covariance matching does not fit means or intercepts. See the complete
[Gaussian fitting example](../experiments/trainer.py) for true, initial and
fitted models, empirical covariance and an independent numerical reference.

## The same batch layout, different meanings for rows

- **Observed values:** `prepare_data` returns complete observations.
- **Generated values:** `SEM2.sample` and `SEM2Linear.sample` return
  `(DataBatch, auxiliary)`. Columns cover user-declared nodes; select the observed
  subset explicitly rather than assuming latent variables were excluded.
- **Interventions:** `set_nodes` is a `DataBatch` of fixed named values, with one
  row per requested draw. The sampler replaces those node equations and retains
  the intervention batch in `auxiliary["interventions"]`. Callers supply valid
  stored values; this low-level input does not encode labels or broadcast rows.
- **Enumerated states:** `DiscreteJointDistribution.states` is a `DataBatch`
  whose rows are possible joint values, not draws or empirical counts.

Continuing the example:

```python
fixed = DataBatch(torch.ones(3, 1), ["x"])
draws, auxiliary = model.sample(3, set_nodes=fixed)
observed_draws = draws.subset(["x", "y"])  # DataBatch: values and metadata.
y_values = observed_draws.select("y")      # Tensor of shape [3, 1].
assert torch.equal(observed_draws.select("x"), fixed.values)
assert auxiliary["interventions"] is fixed
```

Names in `DataBatch(values, ["x", "y"])` become scalar `VariableInfo` objects
internally. Explicit metadata is needed only for non-default dimensions or
representation details. Row slicing, named subsetting and device transfer
preserve metadata and tensor gradient paths; `select` deliberately returns a
plain tensor. A `ScalarResult` has no row axis, so it does not inherit these
batch operations.

## Categories and probability laws

Declare labelled categories with `pd.Categorical`, including unused levels and
optional ordering. Preparation records labels-to-codes in `VariableInfo.encoding`;
it does not standardize values or choose quantitative scores. `numeric_mapping`
is separate. The [`databatch_to_tensor`](../sem2/data/conversion.py) helper applies
supplied scores without replacing stored identities; its no-score categorical
fallback currently warns and uses codes. Not every numerical operator calls that
helper: covariance matching currently uses stored values directly. Do not treat
category ordering or integer codes as meaningful numerical spacing.

Preparation preserves metadata on its returned batch; it does not install
pandas labels or mappings on model nodes. Nor does constructing a new DataFrame
from a batch automatically decode its categories.

`SEM2Discrete.sample(M)` returns `(DiscreteJointDistribution, auxiliary)`:
`states.values` has shape `[K, D]`, and `probabilities` has shape `[M, K]`.
The state table enumerates the discrete endogenous variables; each probability
row is conditional on one exogenous draw. These are **not M categorical draws**.
`marginalize(names)` combines equal projected states by summing their
probabilities while retaining all M rows. Averaging those rows is a separate
Monte Carlo integration over exogenous inputs. Both operations can retain
probability gradients. Intervention belongs to the model's sampling operation,
not to state-table selection or conditioning of this result.
