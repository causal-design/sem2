# SEM2

Structural Equation Model implementation in PyTorch.

## Files
- `sem2.py` - Main SEM2 class (model definition, formula parsing, forward)
- `nn_models.py` - MLP/MLPConfig for neural network nodes

## Running
```bash
python sem2.py      # Runs main() example
jupyter notebook   # Open notebooks for experiments
```

## Testing
No formal test suite. Verify via notebooks or manual runs.

## Key conventions
- Formula syntax: `x ~ Normal(0,1)` (distribution), `y <- x + z` (linear), `y <- n(x,z)` (MLP)
- Configs: `LinearConfig`, `MLPConfig` passed in `node_cfgs` dict keyed by LHS variable name
- Device: defaults to "cpu", pass `device="cuda"` to SEM2 constructor if needed
