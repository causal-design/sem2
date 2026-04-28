# SEM2

Structural Equation Model implementation in PyTorch.

## Files
- `sem2.py` - Main SEM2 class (model definition, formula parsing, forward)
- `nn_models.py` - MLP/MLPConfig for neural network nodes
- `tests/` - pytest test suite
- `notebooks/` - Jupyter notebooks for experiments
- `data/` - Datasets (e.g., PoliticalDemocracy.csv)
- `experiments/` - Standalone scripts

## Testing
```bash
python -m pytest tests/ -v
```

## Key conventions
- Formula syntax: `x ~ Normal(0,1)` (distribution), `y <- x + z` (linear), `y <- n(x,z)` (MLP), `y <- 1*x` (fixed weight)
- Configs: `LinearConfig`, `MLPConfig` passed in `node_cfgs` dict keyed by LHS variable name
- Device: defaults to "cpu", pass `device="cuda"` to SEM2 constructor if needed
- Inject observed values via `model(n, set_nodes={"x": tensor})`
