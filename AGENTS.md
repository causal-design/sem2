# Agent guide

## Start here

- Read [`README.md`](README.md) for usage and [`PROJECT.md`](PROJECT.md) for current work, limitations, and decisions.
- Inspect the relevant source and tests on the current branch. Plans and unmerged PRs are not evidence that an API exists.
- For v0.1 changes, follow the [implementation plan](docs/plan/v0.1.md) and its linked interface specifications. Work only on the requested unit.

## Source map

- `sem2/model/construction/`: formula parser and DAG construction.
- `sem2/model/components/`: executable nodes and neural networks.
- `sem2/model/models/`: base, linear, and finite-discrete models.
- `sem2/operators/`: likelihoods, distributional losses, and causal queries.
- `sem2/sem2trainer.py`: package trainer.
- `tests/`: regression tests; `experiments/` and `notebooks/`: research examples that may use older APIs.

## Coding

- Reuse existing components and conventions; avoid unrelated refactoring or new abstractions.
- Import models from `sem2.model` and node configurations from `sem2.default_configs`. Use `model.sample(...)`, not `forward(...)`.
- Keep graph structure, dataset observation status, and query roles separate. Do not hard-code application-specific node names in generic operators.
- Align variables by name. Distinguish category labels, storage codes, and numeric scores.
- Preserve existing formula behavior unless a change is agreed. Check callers when changing data or operator interfaces.
- Copy `node_cfgs` if it must be retained: construction consumes its entries. Sampling changes cached node values and consumes randomness.
- Flag conflicts between code and the agreed mathematical/API contract rather than silently choosing new behavior. Agent-written tests do not define that contract.
- Treat interventions as model calculations, not proof of identification; do not call numerical extrema sharp bounds without supporting evidence.

## Documentation policy

- Document new or changed public APIs and important internals with concise NumPy-style docstrings. Explain purpose, inputs, outputs, shapes, and essential assumptions or side effects where relevant.
- Put longer mathematics, architecture, and workflows in `docs/`; link rather than duplicate them. Do not pad simple helpers with boilerplate or compulsory examples.
- Update documentation with behavior in the same PR. Describe what the code does and label planned behavior as planned.
- Run configured docstring checks and the strict documentation build; do not weaken checks to make a change pass. See [enforcement status](PROJECT.md#documentation-policy-and-enforcement) for what is actually configured.

Use this format for internal helpers (illustrative, not an existing API); omit sections that do not apply:

```python
import torch


def _mean_over_samples(draws: torch.Tensor) -> torch.Tensor:
    """Average each variable over Monte Carlo samples.

    Parameters
    ----------
    draws : torch.Tensor
        Floating-point values of shape (n_samples, n_variables), n_samples > 0.

    Returns
    -------
    torch.Tensor
        Means of shape (n_variables,), preserving input dtype and device.
    """
    return draws.mean(dim=0)
```

## Verify

Run from the repository root:

```bash
python -m pip install -e .
python -m pytest tests -c pyproject.toml --rootdir=.
git diff --check
```

- For code changes, run focused tests while iterating and the full suite before handoff. Extend existing tests for distinct behavior changes; respect an explicit request to defer test additions.
- For numerical changes, check reference values, gradients, shapes, and relevant device/failure cases. Use fixed seeds and report convergence for optimizer-based examples.
- For documentation-only changes, check links and execute changed examples; a full code-suite rerun is not required.
- Report actual commands, results, warnings, and anything not run. Do not present earlier results as fresh checks.

## Handoff and records

- Summarize the change and verification in the PR. Account for every review comment, including follow-ups and outdated-line comments; leave acceptance, thread resolution, and merge to the human reviewer.
- Keep stable usage in `README.md`, current work in `PROJECT.md`, and durable agent instructions here. Update affected records within the authorized scope; preserve dated meeting/review records as history.
