# Variable support and named data: implementation plan

**Documentation only.** [PR #54](https://github.com/causal-design/sem2/pull/54) records the decisions from the variable-support and categorical-to-numeric-conversion thread. It targets [PR #50](https://github.com/causal-design/sem2/pull/50), not `main`. Its source code is unchanged relative to that review base.

The named-data implementation is reviewed separately in [PR #57](https://github.com/causal-design/sem2/pull/57), with a short code-change summary. Support inspection is separate work in [PR #52](https://github.com/causal-design/sem2/pull/52). Approval of this plan does not approve either implementation. API names below describe the intended interfaces, not capabilities added by this documentation PR.

Each item gives the question or decision, a brief chosen action and considered alternative, then the detailed explanation. The unresolved numeric-column question is last. The [minimal model–data interface](minimal-model-data-interface.md) remains the specification for broader DataFrame preparation; this note does not replace it.

## Variable support and preparation

- **Where should a variable's domain be defined and inspected?**

  **Chosen action:** Keep support on the node, exposed as `node.support: RandomVariableSupport` with `constraint` and `values` only.

  **Alternative considered:** A `get_support()` method, the name `SupportInformation`, or an additional conservative flag. These were not retained.

  **Explanation:** Support describes the node's declared domain, not a guarantee that every value is attainable under the fitted parameters. Reuse PyTorch constraints rather than create a parallel constraint system. `values` provides an enumeration when available; `values=None` means unenumerated, not necessarily infinite. Batch metadata describes representation and must not redefine this domain. Support inspection does not itself convert data, validate every observation, or implement categorical-to-numeric scoring. Those responsibilities must remain explicit.

- **Where should model–data matching and conversion happen?**

  **Chosen action:** Prepare data once through a shared internal path, before batching or optimization.

  **Alternative considered:** Add a second public parser or require users to construct a new data wrapper. Keep ordinary public fitting DataFrame-based instead.

  **Explanation:** The agreed public fitting direction is a pandas DataFrame passed directly to the fitting interface, with shared preparation for direct Trainer use. Preparation handles column-name conversion and reporting, name matching, validation, complete-case selection with input/retained/dropped counts, and internal tensor conversion without modifying the caller's DataFrame. It must not repeat this work in each batch or operator. Follow the linked minimal-interface plan for the exact input and missing-data rules.

  The Dummy loader is the initial integration example, not a production DataFrame parser. Internal `VariableInfo` and `NamedBatch` records can preserve aligned names and representation metadata without becoming a mandatory public wrapper. Full preparation and category-to-score conversion remain later implementation work; they are not supplied by adding named records alone.

## Named-data contract

- **Should the new named format coexist with the old data formats?**

  **Chosen action:** Use named records throughout the scoped low-level interfaces; remove backward compatibility.

  **Alternative considered:** Preserve tensor inputs and old sampler returns, adding named wrappers only at the operator boundary. This was superseded by the requested clean break.

  **Explanation:** The implementation should carry `VariableInfo`, `NamedBatch`, and `FiniteDistribution` through the Dummy loader, canonical `SEM2Trainer`, model samplers, and every implemented operator. Selection, slicing, transfer, and finite marginalization must preserve metadata and autograd. This is an intentional breaking migration, not an optional wrapper alongside the old formats. It does not add an implementation of `PathAnalysis`.

  Low-level observations and interventions should require `NamedBatch`; positional tensors and dictionary interventions should be rejected rather than accepted through compatibility shims. Migrate tests and runnable Python examples, including the test helper's return-format switch. Ordinary tensors remain appropriate inside numerical calculations, parameters, and scalar outputs. This internal contract does not replace the future public DataFrame fitting interface.

- **How should generated samples differ from finite-state probability results?**

  **Chosen action:** Use `NamedBatch` for draws and `FiniteDistribution` for states paired with probabilities.

  **Alternative considered:** Retain dictionary samples or return raw probabilities with separate state metadata. Do not retain those parallel result formats.

  **Explanation:** The target contract is `NamedBatch` draws from `SEM2.sample` and `SEM2Linear.sample`, and a `FiniteDistribution` from `SEM2Discrete.sample` with named states and probabilities `[exogenous draws, states]`. Preserve the common outer `(result, auxiliary)` convention. Record intervention provenance as a `NamedBatch`; remove the old raw `states`/`full_state` auxiliary payloads.

  `PotentialOutcome` should use the same sampler results, with continuous queries still stopping at the target. Existing descriptors may preserve external-label encodings. State marginalization sums probability mass; integration over exogenous draws is a separate mean. Merely selecting state columns without combining the corresponding probabilities is not marginalization. The scoped implementation uses the existing `set_nodes` name; the broader user-facing intervention naming remains in the minimal-interface plan.

- **How should the named format retain existing vector-valued nodes?**

  **Chosen action:** Record each variable's contiguous column width.

  **Alternative considered:** Keep the initial scalar-column-only record. That would lose existing supported node dimensions.

  **Explanation:** Use `VariableInfo.width` to retain the contiguous columns of existing vector-valued nodes. Scalar mean/ATE queries must explicitly require scalar roles. This is a small extension of the named record, not a new general tensor-valued data framework. `select(...)` should return the requested tensor blocks in named order; `subset(...)` should retain their metadata as a `NamedBatch`. Names determine alignment rather than merely labelling a positional calculation.

## Fitting and numerical meaning

- **How should the discrete energy calculation receive observations and predictions?**

  **Chosen action:** Pass a `NamedBatch` and a `FiniteDistribution` that owns its state table.

  **Alternative considered:** Keep raw probability tensors, a separate state-table argument, the extra batch axis, or `p_true_mean`. Remove those parallel entry paths.

  **Explanation:** The target signatures are `DiscreteEnergyLoss(p).energy_score(observations, prediction)` and `.energy_dist(observations, prediction)`, with named observations and a finite-distribution prediction. Predictions own their state tables. Named alignment determines which observed columns correspond to prediction states.

  Keep observation and model sample counts separate in pair corrections. Validate discrete values before coercion: invalid fractional codes must not become valid states through integer conversion, and unmatched observations must not silently become state zero. Cast integer-only inputs at the neural-network boundary when required, without changing their external representation. Such a cast is a computational requirement, not a decision that category codes are meaningful quantitative scores.

- **How should an instrumental-variable role be selected?**

  **Chosen action:** Require explicit `TrainingConfig(instrument_node="Z")` for the IV-specific objective.

  **Alternative considered:** Infer the instrument from column position and assume binary levels. Do not retain those assumptions.

  **Explanation:** The scoped IV-specific discrete objective requires an exogenous discrete instrument and at least two observations per observed group. Neither variable names nor level counts should be inferred from column positions. Without an instrument, discrete fitting should use the named observed endogenous margin. This is not a general observational-conditioning interface, and role checks do not establish exogeneity or causal identification. Conditioning and intervention remain distinct operations.

- **Does an integer category code establish a quantitative score?**

  **Chosen action:** Keep category labels, internal state codes, and deliberate numeric scores distinct.

  **Alternative considered:** A numeric default versus an explicit declaration or mapping remains open; discuss it at the end rather than adopt it here.

  **Explanation:** Existing distance and mean calculations use stored values, but a category ID alone does not make arithmetic meaningful. `VariableInfo.encoding` records external labels and their stored integer state codes; it is not a category-to-score conversion policy. Finite or discrete variables need not be categorical, and dtype alone does not establish the variable's meaning. Node-level support remains authoritative.

  Deliberate scores can make arithmetic meaningful without turning finite-discrete fitting into a continuous approximation. Preserve the finite-state engine's internal codes and the distinction between coding and scoring. The present numerical operators' use of stored values must not be treated as an agreed default for ambiguous columns in future preparation. Neither this plan nor the scoped named-data implementation settles that policy or supplies a full conversion layer.

## Implementation sequence and acceptance

- **Which training path and checks should the implementation use?**

  **Chosen action:** Integrate through the canonical package trainer and verify the named-data flow end to end.

  **Alternative considered:** Keep a separate experimental optimization loop. Delegate the example to `SEM2Trainer` instead.

  **Explanation:** Implement the records first, pass them through the Dummy loader and trainer, migrate sampler and operator boundaries together, then update tests and runnable examples. Preserve numerical behavior while checking the explicit boundary changes. The demonstration should display metadata, assert equivalence under column reordering, perform a short optimization run, and show named intervention outcomes. A few optimizer updates demonstrate integration, not convergence or validated causal bounds.

  On the implementation branch, use `python -m experiments.trainer` (with `python -m experiments.train` as its equivalent entry point). Check named alignment and renaming, numerical references, gradients, invalid states, rejection of legacy inputs, vector blocks, result axes, and Dummy training. Run the full suite and the discrete/model integration demos; report dependency and whitespace checks, warnings, and the devices actually tested. These are acceptance requirements, not test results for code introduced by this documentation PR. Execution evidence belongs with the implementation PR.

- **What belongs in each review?**

  **Chosen action:** Keep PR #54 documentation-only; review the named-data code separately in PR #57.

  **Alternative considered:** Keep the plan and implementation together in PR #54. The requested split supersedes that arrangement.

  **Explanation:** Both PRs target PR #50's branch. This plan retains the decisions, alternatives, rationale, and open discussion. The implementation PR carries the changed Python files, the README/contributor guidance describing those implemented interfaces, and a minimal code-change summary. README and AGENTS changes in this planning PR only point to the plan and make its status explicit; their current-API examples remain consistent with the base code.

  Do not expand the named-data implementation into full DataFrame preparation, choose a category-to-score default, add dependencies, or import support PR #52's code as part of this split. Keep support inspection and future conversion work distinguishable from the implemented named-data portion. Plan approval, implementation approval, merge, and release remain separate decisions.

## Discussion: ambiguous numeric columns

- **When a numeric column could contain either quantitative values or category codes, what should preparation assume?**

  **Chosen action:** Leave the default undecided for team discussion; implement neither policy as part of this split.

  **Alternatives to discuss:** Treat numeric values as quantitative by default, or require an explicit declaration or mapping before using ambiguous values quantitatively.

  **Explanation:** A column stored as numbers may represent measurements, deliberate scores, or category labels encoded as integers. Storage dtype alone does not resolve that distinction. The discussion should decide what evidence preparation requires before values are used in means, distances, or other arithmetic, and what the user must supply when their meaning is ambiguous.

  Keep the settled distinctions intact: category identity is separate from its internal code and from an optional score; node-level support remains authoritative; and a score mapping does not by itself change finite-discrete fitting into continuous fitting. This is a question for the future preparation/conversion policy, not permission to add an implicit default to the named-data implementation.
