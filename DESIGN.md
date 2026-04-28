# SEM2: Structural Equation Modeling in PyTorch

## What Is SEM2?

SEM2 is a **probabilistic programming library** for defining and fitting Structural Equation Models (SEM) in PyTorch. You write a model as a string of formulas, and SEM2 builds a PyTorch module that you can sample from, fit to data, and inspect.

**Use case:** Suppose you believe `y` depends on `x` and a random noise `e`, where `x` itself depends on a latent factor `f` and noise. You can write:

```
f ~ Normal(0,1)
x <- f + e1
y <- 2*x + e2
```

SEM2 turns this into a PyTorch model where you can:
- Sample from the model (`model(n)` generates `n` samples)
- Fit the model to real data via gradient-based optimization
- Inspect learned weights (`model.layer['y'].weight_matrix()`)

---

## Core Concepts

### 1. Formula Syntax

Formulas are separated by semicolons (`;`).

| Symbol | Meaning | Example | Node Type |
|--------|---------|---------|-----------|
| `~` | Random variable (root node, no parents) | `u1 ~ Normal(0,1)` | `NodeDistribution` |
| `<-` | Deterministic relationship | `y <- x + z` | `NodeLinear` |
| `n(...)` | Neural network (nonlinear relationship) | `x <- n(u1,u2)` | `NodeNN` |
| `1*v` | Fixed weight (not learned, forced to 1) | `x <- 1*y + z` | `NodeLinear` |

**Distribution names** (`Normal`, `Exponential`, `Uniform`, etc.) come from `torch.distributions`.

**Multiple outputs on LHS** expand automatically:
```
x1, x2, x3 ~ Normal(0,1)
```
expands to three separate nodes.

### 2. Nodes

Each formula becomes a **node** in a directed acyclic graph (DAG):

- **`NodeDistribution`** — samples from a PyTorch distribution. No parents.
- **`NodeLinear`** — linear layer `y = Wx + b`. Supports fixed weights via `1*var` prefix.
- **`NodeNN`** — multi-layer perceptron (MLP) for nonlinear relationships.

All nodes inherit from `SEM2Node`, which extends `nn.Module`. They store:
- `value`: the computed tensor for the current forward pass
- `out_dim`: output dimension (for stacking in cat)
- `predecessors`: list of parent node names

### 3. The Graph

Nodes are connected by their dependencies. `y <- x + z` means `y` depends on `x` and `z`. SEM2 uses Python's `graphlib.TopologicalSorter` to compute the execution order so parents are always computed before children.

### 4. Forward Pass

When you call `model(batch_size)`:

1. Nodes are processed in topological order
2. For each node, collect parent values from the cache
3. Run the node's computation (sample / linear / MLP)
4. Store the result for children to use
5. Return all node values concatenated: shape `[batch_size, total_out_dim]`

You can inject observed values with `set_nodes={name: tensor}` to condition the model.

---

## Architecture

```
sem2.py
  - SEM2Node (ABC)          Base class for all nodes
    - NodeDistribution      torch.distributions sample
    - NodeLinear            Linear layer with optional fixed weights
    - NodeNN                MLP wrapper
  - SEM2 (nn.Module)        Main model class
    - _parse_formula        Parse formula string to list of nodes
    - _expand_lhs/rhs       Expand comma-lists and n() terms
    - forward               Run the model
  - nn_models.py
    - MLPConfig             Dataclass for MLP architecture
    - MLP                   PyTorch Sequential model
```

**Flow:** `SEM2.__init__` → `_parse_formula` → creates nodes → `TopologicalSorter.static_order()` → `node.initialize_model()` → ready to `forward()`.

---

## Usage Example

```python
from sem2 import SEM2, LinearConfig
import torch

# Define model
formula = 'u1 ~ Normal(0,1); u2 ~ Normal(0,1); y <- n(u1,u2)'
model = SEM2(formula)

# Sample from model
samples = model(100)  # shape: [100, 3]

# Or with custom MLP config
from nn_models import MLPConfig
config = MLPConfig(input_dim=2, hidden_dims=[64, 64], output_dim=1)
model = SEM2(formula, node_cfgs={'y': config})

# Inject observed values
observed = {'u1': torch.randn(100, 1)}
conditioned = model(100, set_nodes=observed)
```

---

## Loss Functions

SEM2 is just a sampler. You choose how to fit it. Common approaches (shown in `linear_sem.ipynb` and `nonlinear_example.ipynb`):

**Energy loss** (from `engression` package):
```python
loss, s1, s2 = energy_loss(real_data, model_samples)
```

**Maximum likelihood** (match covariance):
```python
model_cov = torch.cov(torch.t(model_samples))
loss = torch.log(torch.det(model_cov)) + torch.trace(real_cov @ torch.inverse(model_cov))
```

---

## Refactoring Proposal

The current code has the following issues planned for addressing:

1. **Private inheritance** — `SEM2Node` inherits from `graphlib._NodeInfo` (a private CPython class). This is fragile across Python versions.
2. **Magic strings** — Symbols like `~`, `<-`, `n(` are scattered throughout the code.
3. **God class** — `SEM2` mixes parsing, graph building, model registration, and forward pass.

See `DESIGN_refactor.md` for the detailed refactoring plan.

---

## Quick Reference

| Property | Access |
|----------|--------|
| Topological order | `list(model.static_order())` |
| Node parameters | `model.parameters()` |
| All layers | `model.layer` (ModuleDict) |
| Single node output | `model.layer['node_name'].value` |
| Weight matrix | `model.layer['node_name'].weight_matrix()` |
| Node out_dim | `model.layer['node_name'].out_dim` |