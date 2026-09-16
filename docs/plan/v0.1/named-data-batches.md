# Named data: decisions and discussion

This note preserves the implementation explanation from [PR #54](https://github.com/causal-design/sem2/pull/54), organised for review. The implementation remains unchanged by this documentation follow-up. Each item gives the decision, a brief action and alternative, then the detailed explanation. The unresolved question is at the end.

The PR is a draft stacked on [PR #50](https://github.com/causal-design/sem2/pull/50), targeting `docs/product-and-implementation-plan`. The [minimal model–data interface](minimal-model-data-interface.md) remains the specification for the broader preparation work; this note describes the smaller implemented slice.

## Data contract

- **Should the new named format coexist with the old data formats?**

  **Chosen action:** Use named records throughout; remove backward compatibility.

  **Alternative considered:** Preserve tensor inputs and old sampler returns, adding named wrappers only at the operator boundary. This was superseded by the requested clean break.

  **Explanation:** Following review, this is an intentional breaking migration: named data is the sole contract, not an opt-in wrapper alongside the old formats.

  `VariableInfo`, `NamedBatch`, and `FiniteDistribution` carry identity and representation metadata through the Dummy loader, canonical `SEM2Trainer`, model samplers, and every implemented operator. Selection, slicing, transfer, and finite marginalization preserve metadata and autograd. `PathAnalysis` remains explicitly unimplemented.

  Observations and interventions must be `NamedBatch` objects. Positional tensors and dictionary interventions are rejected, not converted by compatibility shims. Tests and runnable Python examples use the new records; the test helper's return-format switch is removed. Ordinary tensors remain inside numerical calculations, parameters, and scalar outputs.

- **How should generated samples differ from finite-state probability results?**

  **Chosen action:** Use `NamedBatch` for draws and `FiniteDistribution` for states paired with probabilities.

  **Alternative considered:** Retain dictionary samples or return raw probabilities with separate state metadata. Those return formats were removed.

  **Explanation:** `SEM2.sample` and `SEM2Linear.sample` return `NamedBatch` draws. `SEM2Discrete.sample` returns `FiniteDistribution` with named states and probabilities `[exogenous draws, states]`. The common outer `(result, auxiliary)` convention remains. Intervention provenance is a `NamedBatch`; old raw `states`/`full_state` auxiliary payloads are removed.

  `PotentialOutcome` uses the same sampler results; continuous queries still stop at the target. Existing descriptors can preserve external-label encodings. State marginalization sums mass; exogenous integration is a separate mean. Selecting state columns without combining the corresponding probabilities is not marginalization.

- **How should the named format retain existing vector-valued nodes?**

  **Chosen action:** Record each variable's contiguous column width.

  **Alternative considered:** Keep the initial scalar-column-only record. That would lose existing supported node dimensions.

  **Explanation:** `VariableInfo.width` retains the contiguous columns of existing vector-valued nodes. Scalar mean/ATE queries explicitly require scalar roles. This is a small extension of the named record, not a new general tensor-valued data framework. `select(...)` returns the requested tensor blocks in named order; `subset(...)` retains their metadata as a `NamedBatch`.

## Fitting and numerical meaning

- **How should the discrete energy calculation receive observations and predictions?**

  **Chosen action:** Pass a `NamedBatch` and a `FiniteDistribution` that owns its state table.

  **Alternative considered:** Keep raw probability tensors, a separate state-table argument, the extra batch axis, or `p_true_mean`. Those entry paths were removed.

  **Explanation:** `DiscreteEnergyLoss(p).energy_score(observations, prediction)` and `.energy_dist(observations, prediction)` require `NamedBatch` observations and a `FiniteDistribution` prediction. Predictions own their state tables. Raw tensor inputs, the legacy extra batch axis, and `p_true_mean` are removed.

  The integration uses separate observation/model sample counts, validates states before coercion, and casts integer-only inputs at the neural boundary. Invalid fractional codes must not become valid states through integer conversion, and unmatched observations must not silently become state zero. Named alignment determines which observed columns correspond to the prediction's states.

- **How should an instrumental-variable role be selected?**

  **Chosen action:** Require explicit `TrainingConfig(instrument_node="Z")` for the IV-specific objective.

  **Alternative considered:** Infer the instrument from column position and assume binary levels. Those assumptions were removed.

  **Explanation:** `TrainingConfig(instrument_node="Z")` explicitly selects the IV-specific discrete objective: an exogenous discrete instrument and at least two observations per observed group are required. Names and level counts are not inferred from positions. With no instrument, discrete fitting uses the named observed endogenous margin. This is not general observational conditioning or evidence of identification.

- **Does an integer category code establish a quantitative score?**

  **Chosen action:** Keep category labels, internal state codes, and deliberate numeric scores distinct.

  **Alternative considered:** A numeric default versus an explicit declaration or mapping remains open; it is discussed at the end rather than adopted here.

  **Explanation:** Distances/means still use stored values: category IDs alone do not establish meaningful scores. `VariableInfo.encoding` records external labels and their stored integer state codes; it is not a category-to-score conversion policy. Finite or discrete variables need not be categorical, and dtype alone does not establish the variable's meaning. Representation metadata does not redefine a node's declared domain.

  Deliberate scores can make arithmetic meaningful without turning finite-discrete fitting into a continuous approximation. This PR does not implement that conversion or the full DataFrame preparation layer. The existing numerical operators' use of stored values is not a newly agreed default for ambiguous columns in future preparation.

## Runnable example and verification

- **Which training path and evidence should reviewers use?**

  **Chosen action:** Use the canonical package trainer, migrated examples, and the recorded tests.

  **Alternative considered:** Keep a separate experimental optimization loop. The example now delegates to `SEM2Trainer` instead.

  **Explanation:** Run the example from the repository root:

  ```bash
  python -m experiments.trainer
  # equivalent entry point:
  python -m experiments.train
  ```

  The example displays metadata, asserts reorder equivalence, performs four optimizer updates, and prints named intervention outcomes. This demonstrates integration, not convergence or validated causal bounds.

  The implementation verification recorded for commit `c09703964e8180bfdb0ec75fca1935b348ccc49e` was:

  - Full suite: 209 passed, the same 51 existing warnings (205 before the clean-break revision).
  - 19 focused cases cover alignment/renaming, numerical references, gradients, invalid states, legacy-input rejection, vector blocks, result axes, and Dummy training.
  - Both training entry points, `experiments.discrete_demo`, and `sem2.model.models._integrationTestDemos` executed successfully. Model demos emit specialization warnings.
  - `pip check` and staged whitespace checks passed. CPU verification only.

  These are the preserved implementation results, not a claim of GPU verification or statistical validation. Documentation-follow-up checks are reported separately in the PR.

## Scope and review base

- **Should this follow-up expand the implementation or change its review target?**

  **Chosen action:** Preserve the implementation and target; update the authorised documentation only.

  **Alternative considered:** Full DataFrame preparation and category-to-score conversion remain separate work, not additions to this PR.

  **Explanation:** There is no full DataFrame preparation layer, category-to-score policy, new dependency, or dependency on support PR #52 in this implementation. The original PR description noted that `README.md` and `AGENTS.md` still described the previous interfaces and that updating them awaited consent. This follow-up has that consent and updates those files; it does not change the implementation or settle the conversion policy.

  Earlier publication incorporated the already-merged `main` baseline from #49. `PROJECT.md` appears only for that inherited update; this follow-up does not edit it. The PR still targets #50's branch, remains draft, and is not merged. Merge and release decisions remain separate human review steps.

## Discussion: ambiguous numeric columns

- **When a numeric column could contain either quantitative values or category codes, what should preparation assume?**

  **Chosen action:** Leave the default undecided for team discussion; implement neither policy in this PR.

  **Alternatives to discuss:** Treat numeric values as quantitative by default, or require an explicit declaration or mapping before using ambiguous values quantitatively.

  **Explanation:** A column stored as numbers may represent measurements, deliberate scores, or category labels encoded as integers. Storage dtype alone does not resolve that distinction. The discussion should decide what evidence preparation requires before the values are used in means, distances, or other arithmetic, and what the user must supply when that meaning is ambiguous.

  Keep the settled distinctions intact: category identity is separate from its internal code and from an optional score; node-level support remains authoritative; and a score mapping does not by itself change finite-discrete fitting into continuous fitting. This discussion concerns the future preparation/conversion policy, not a new validation or scoring feature added to #54.
