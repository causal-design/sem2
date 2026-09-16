# Variable support and named data: implementation plan

**Documentation only.** This specification carries forward [PR #54](https://github.com/causal-design/sem2/pull/54)'s variable-support and categorical-to-numeric-conversion decisions into the combined [v0.1 planning review](../v0.1.md) for [PR #50](https://github.com/causal-design/sem2/pull/50). Source code is unchanged relative to that review base.

The closed [named-data PR #57](https://github.com/causal-design/sem2/pull/57) and [support PR #52](https://github.com/causal-design/sem2/pull/52) are **unmerged implementation examples**, not approved implementations or dependencies of this plan. For concrete illustrations, see the [named-data change summary](https://github.com/causal-design/sem2/blob/e43d22854f18a116f382a3d30160290a267146a3/docs/implementation/named-data-changes.md) and [node-support implementation](https://github.com/causal-design/sem2/blob/d9897a0d334a647a62c90685220470db13ca5330/sem2/model/components/nodes.py) at their preserved revisions. Their code is not included here. API names below describe intended interfaces; implementation needs separate authorization and review.

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

  `PotentialOutcome` should use the same sampler results, with continuous queries still stopping at the target. Existing descriptors may preserve external-label encodings. State marginalization sums probability mass; integration over exogenous draws is a separate mean. Merely selecting state columns without combining the corresponding probabilities is not marginalization. The closed code example uses the existing `set_nodes` name; the v0.1 user-facing name remains `do_intervention`, as specified in the minimal-interface plan.

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

  **Explanation:** The IV-specific discrete objective requires an exogenous discrete instrument and at least two retained complete-case observations per declared instrument level, including levels absent from the data. Otherwise, stop before fitting and report affected levels and counts; do not silently drop or merge levels or add pseudo-counts. Check full-data eligibility separately from minibatch safety, following the [discrete-IV plan](identification-and-partial-identification.md#discrete-iv-example). Neither variable names nor level counts should be inferred from column positions. Without an instrument, discrete fitting should use the named observed endogenous margin. This is not a general observational-conditioning interface, and role checks do not establish exogeneity or causal identification. Conditioning and intervention remain distinct operations.

- **Does an integer category code establish a quantitative score?**

  **Chosen action:** Keep category labels, internal state codes, and deliberate numeric scores distinct.

  **Alternative considered:** A numeric default versus an explicit declaration or mapping remains open; discuss it at the end rather than adopt it here.

  **Explanation:** Existing distance and mean calculations use stored values, but a category ID alone does not make arithmetic meaningful. `VariableInfo.encoding` records external labels and their stored integer state codes; it is not a category-to-score conversion policy. Finite or discrete variables need not be categorical, and dtype alone does not establish the variable's meaning. Node-level support remains authoritative.

  Deliberate scores can make arithmetic meaningful without turning finite-discrete fitting into a continuous approximation. Preserve the finite-state engine's internal codes and the distinction between coding and scoring. The present numerical operators' use of stored values must not be treated as an agreed default for ambiguous columns in future preparation. Neither this plan nor the scoped named-data implementation settles that policy or supplies a full conversion layer.

  Follow the [general variable and estimand requirements](identification-and-partial-identification.md#general-design-requirements-for-variables-and-estimands): retain original categorical status, labels, mappings, and any user-supplied numerical interpretation after both encoding and deliberate conversion. For categorical outcomes without that interpretation, warn and suggest category-probability contrasts or deliberate scoring instead of silently using codes as ATE values. The remaining numeric-column question concerns ambiguous inputs, not permission to discard known categorical meaning.

## Implementation sequence and acceptance

- **Which training path and checks should the implementation use?**

  **Chosen action:** Integrate through the canonical package trainer under the [minimum fitting and diagnostics contract](minimum-fitting-interface-and-diagnostics.md), and verify the named-data flow end to end.

  **Alternative considered:** Keep a separate experimental optimization loop. Delegate the example to `SEM2Trainer` instead.

  **Explanation:** Implement the records first, pass them through the Dummy loader and trainer, migrate sampler and operator boundaries together, then update tests and runnable examples. Preserve numerical behavior while checking the explicit boundary changes. The demonstration should display metadata, assert equivalence under column reordering, perform a short optimization run, and show named intervention outcomes. A few optimizer updates demonstrate integration, not convergence or validated causal bounds.

  The closed named-data example provides `python -m experiments.trainer` (with `python -m experiments.train` as its equivalent entry point); these commands refer to that example's revision, not an implementation supplied by this plan. For a future implementation, check named alignment and renaming, numerical references, gradients, invalid states, absent/singleton declared instrument levels, rejection of legacy inputs, vector blocks, result axes, and Dummy training. Run the full suite and the discrete/model integration demos; report dependency and whitespace checks, warnings, and the devices actually tested. These are acceptance requirements, not test results for code introduced by this documentation PR. Execution evidence belongs with the future implementation review.

- **What belongs in each review?**

  **Chosen action:** Combine the support/data and trainer plans in one documentation-only PR targeting PR #50. Keep the closed code PRs as linked examples.

  **Alternative considered:** Keep plan and implementation together in PR #54. The earlier split and the current planning-only review supersede that arrangement.

  **Explanation:** This plan retains the decisions, alternatives, rationale, and open discussion from PR #54; the companion fitting document carries PR #56's trainer requirements. PRs #52 and #57 preserve code illustrations but are closed and unmerged. README and AGENTS changes in the combined review only point to the plan and make its status explicit; their current-API examples remain consistent with the base code.

  Do not import code, add dependencies, or choose a category-to-score default as part of this consolidation. Full DataFrame preparation and deliberate conversion remain v0.1 implementation work under their respective specifications; the closed examples do not supply them. Plan approval, implementation approval, merge, and release remain separate decisions.

## Discussion: ambiguous numeric columns

- **When a numeric column could contain either quantitative values or category codes, what should preparation assume?**

  **Chosen action:** Leave the default undecided for team discussion; implement neither policy as part of this planning review.

  **Alternatives to discuss:** Treat numeric values as quantitative by default, or require an explicit declaration or mapping before using ambiguous values quantitatively.

  **Explanation:** A column stored as numbers may represent measurements, deliberate scores, or category labels encoded as integers. Storage dtype alone does not resolve that distinction. The discussion should decide what evidence preparation requires before values are used in means, distances, or other arithmetic, and what the user must supply when their meaning is ambiguous.

  Keep the settled distinctions intact: category identity is separate from its internal code and from an optional score; node-level support remains authoritative; and a score mapping does not by itself change finite-discrete fitting into continuous fitting. This is a question for the future preparation/conversion policy, not permission to add an implicit default to the named-data implementation.
