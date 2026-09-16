# Named-data code changes

Implementation of the named-data portion of the [plan in PR #54](https://github.com/causal-design/sem2/pull/54).

- `sem2/data.py`: adds `VariableInfo`, `NamedBatch`, and `FiniteDistribution`, including named selection, metadata, vector widths, and finite marginalization.
- Loader/trainer: passes named batches from the Dummy loader through the canonical `SEM2Trainer`; adds explicit `instrument_node` configuration.
- Models/operators: standardizes observations, interventions, draws, and finite-distribution results; removes legacy tensor/dictionary data interfaces and aligns inputs by name.
- Numerical fixes: validates discrete values before coercion, separates observation/model sample counts, and casts integer inputs at the neural-network boundary.
- Examples/tests: migrates callers and covers alignment, gradients, invalid states, retired interfaces, result axes, and Dummy training.
- `README.md` and `AGENTS.md`: describe the implemented interfaces and link back to the separate plan.

The Python files are preserved byte-for-byte from the pre-split PR #54 revision `9e45e53503e5213e4bb6213b1e133544492ea8d3`. This split adds no new numerical behavior.

Full DataFrame preparation and category-to-score conversion remain future work. Support inspection is separate work in PR #52; this implementation does not import its changes. The ambiguous numeric-column policy remains open in the plan.

Run the demonstration with `python -m experiments.trainer`; run the tests with `python -m pytest`. The short demonstration is not a convergence or causal-identification claim.
