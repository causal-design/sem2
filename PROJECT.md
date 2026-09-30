# SEM2 project status

**Last updated:** 30 September 2026

**Default-branch source baseline:** `main` at `ac3052b0a1ba08f228f935552e85605a82dde34e`.

This file records current capabilities, limitations, decisions, and next work. [`README.md`](README.md) gives stable user context; [`AGENTS.md`](AGENTS.md) gives durable contribution rules. Fresh source and executed evidence take precedence over stale prose. The [v0.1 product specification](docs/plan/v0.1/README.md) and [implementation plan](docs/plan/v0.1.md) define the target, not a claim that every planned feature is present.

## Current implementation on main

The parser/DAG builder, continuous and linear samplers, scalar linear moment calculations, Gaussian likelihood operator, and small finite-state probability engine are present. This does not establish a general end-to-end fitting workflow or release readiness.

| Area | Observed implementation and limits |
|---|---|
| Continuous/linear sampling | `sample()` returns a node-to-tensor dictionary plus an auxiliary dictionary. `set_nodes` accepts a dictionary of node names to valid tensors. Calling the model directly raises `NotImplementedError`. |
| Finite-discrete models | `SEM2Discrete` enumerates endogenous states and returns probabilities for each exogenous draw, with auxiliary state information. Enumeration grows as the product of category counts; keep examples small. This is not a batch of realized category draws. |
| Linear models/objectives | Scalar linear means/covariances and tensor-based Gaussian log-likelihood/covariance objectives are implemented. `LogLikelihood` returns mean **log**-likelihood, so a minimizing optimizer needs its negative. It is exact for linear Gaussian models and a moment-based pseudo-likelihood otherwise. |
| Observation status/data | `unobs(...)` supplies the model's observed-variable list. The default branch has no `sem2/data.py`, named records, or reusable DataFrame preparation path. Tensor column alignment remains a caller responsibility. |
| General fitting | The package trainer and a separate experimental trainer remain. No public `fit` entry point or systematic returned training history/final report is present. The package trainer's batch and objective interfaces remain experimental. |
| Energy/IV objectives | Continuous energy evaluation explicitly selects `Z`, `X`, `Y`; the discrete route uses the first column as binary `Z`. Continuous sample-axis/estimator handling needs review. These are not graph-generic fitting guarantees. |
| Query/combination | Potential-outcome operators and a config-driven weighted loss are present. `ATE` fixes a 1-versus-0 intervention contrast; arbitrary-level/joint public query interfaces and explicit operator-list composition are not current default-branch interfaces. |
| Remaining scope | Mixed-model fitting, a supplied joint empirical covariate distribution, graphical identification, native penalty sweeps/extrapolation, general uncertainty quantification, MMD and PFN inference are not completed package workflows. Optimizing a target alongside fit does not by itself provide sharp identified bounds. |

### Unmerged implementation and review status

[PR #60](https://github.com/causal-design/sem2/pull/60), inspected at `c1b219088af3c5bb54086572e567517818f9adce`, contains preparatory steps 1–5: formula/helper repairs, node-owned support, named records and caller migration, scalar DataFrame preparation, and explicit scalar operator–weight composition. Its actual record names include `DataBatch`, `FiniteDistribution`, and `ScalarResult`. These are **not merged main capabilities**.

The [30 September review instruction](https://github.com/causal-design/sem2/pull/60#issuecomment-5909025337) says no further updates to PR #60 are required: incorporate its comments in a new PR, then it can be closed. The current documentation update is a separate, three-file PR against `main`; it does not revise that implementation, claim all its comments resolved, or close PR #60. The replacement implementation must account for every review comment and follow-up. Later trainer development is separate work and is not included in this source baseline.

Keep the reviewed implementation distinct from the intended design. Its incomplete observation-status transition, categorical scoring behavior, fixed 1-versus-0 convenience contrast, and remaining numerical/documentation issues still require review. Earlier inline explanations are not evidence that corrections have been implemented or accepted.

## Agreed direction

### Separate model, data, and query

The structural model specifies the graph, exogenous distributions, and mechanisms. Observation status belongs to the dataset relative to that model. Treatment, outcome, estimand, and intervention roles belong to queries/operators, not hard-coded node names or positional columns in generic components.

For v0.1, the agreed fitting input is a named pandas DataFrame of **scalar** variables, prepared internally with complete-case reporting. A categorical variable with several levels is still scalar. The richer dataset/schema interface is later work, not an unresolved prerequisite for this minimal contract. Distinguish category labels, stored integer codes, and deliberate numeric scores; numerical storage does not establish quantitative meaning. Scoring is separate preprocessing before training, with supplied meaning/provenance retained rather than guessed.

The target named-data and formula migrations are described in the [model–data handoff](docs/plan/v0.1/minimal-model-data-interface.md) and [support/data handoff](docs/plan/v0.1/named-data-batches.md). Do not advertise the named records, support properties, fixed numeric intercepts, or public fitting route as available on `main` until their implementation is reviewed and merged. Preserve valid existing formulas while making any intentionally retired boundary formats explicit.

### Keep the release scope and causal claims bounded

Use a graph supplied by the user. The agreed v0.1 examples are PoliticalDemocracy, Kang–Schafer adapted to the average treatment effect on the treated (ATT), the two-stage randomized example, and discrete IV. There is no separate missing-data case study. RHC/proximal/RDD demonstrations and a general empirical-covariate framework are not additions to this release scope.

Preserving the observed **joint** covariate distribution remains important: replacing dependent covariates with independent roots can change the target population. Its general implementation is deferred, not solved by scalar data preparation. Historical RHC discussions remain useful motivation rather than an active v0.1 acceptance requirement.

Distinguish fit to observed distributions, identification, target regularity, finite-sample uncertainty, and numerical optimization error. An intervention is a model calculation, not validation of a graph or identification. A numerical search endpoint is not automatically a sharp bound or confidence limit. Prioritize credible point estimation before general uncertainty machinery. Graph discovery and broad PFN amortization remain later research, requiring their own baselines, calibration and prior-shift evaluation.

### Use agents within a human-controlled process

Humans retain control of scientific meaning, architecture, accepted behavior, review and merge. Agents work on explicitly authorized bounded changes. Keep plan and implementation review distinct, reuse existing components, and justify new abstractions with a concrete requirement. Reviewed contracts—not agent-authored tests alone—define correctness. An earlier task, plan, or passing tests does not authorize the next implementation unit, merge, or deployment.

## Documentation policy and enforcement

**Agreed policy:** concise but sufficient local API contracts, automatically generated API reference, and longer mathematical/architectural/workflow explanations in `docs/`. Document important internals as well as public objects, update documentation with behavioral changes, and cross-link rather than duplicate. Do not require padding or long examples for trivial helpers. The canonical contributor rule, including a NumPy-style docstring example, is in [`AGENTS.md`](AGENTS.md#documentation-policy); [`README.md`](README.md#documentation-and-contribution-policy) provides the user-facing summary.

**Enforcement design:** selected `numpydoc` checks over source objects, a strict Sphinx build including cross-reference checks, PR evidence of what changed and what ran, and required human review. Protect policy files, workflows, checker configuration and exceptions with human review too. Agents must not disable checks or bypass approval. Human reviewers assess semantic/mathematical accuracy and concision; presence/format checks cannot establish these.

**Actual setup at this baseline:** no tracked Sphinx configuration, `numpydoc` validation configuration, pre-commit configuration, or GitHub workflow was found. Documentation tools are not declared in `pyproject.toml`. GitHub branch-protection/required-review settings were not inspected and must not be represented as enabled or disabled. This update changes only the three root Markdown files; it does not add enforcement configuration, update source docstrings, build a documentation site, or alter administrative settings.

**Remaining setup:** prepare the small API/page preview, agree selected checks and explicit exceptions, configure source coverage and the strict build, then have an authorized maintainer make CI checks and human approvals required. Sphinx with NumPy-style docstrings and `numpydoc` validation is the recommended implementation route; build/hosting details still need review. If existing omissions require phased adoption, record a reviewed scope/baseline rather than silently suppressing failures. The earlier plan's documentation-preview item remains relevant; the local-versus-external documentation policy itself is now recorded, not an open question.

## Current work and next reviews

1. **Replace, rather than keep updating, PR #60.** A separately authorized implementation revision must carry forward the applicable steps 1–5 work and account for every original comment/follow-up. Distinguish explanations, implemented changes awaiting review, decisions needed, and agreed deferrals. This documentation-only PR is not that code replacement.
2. **Implement documentation enforcement in a separately authorized change.** Follow the setup above and expand missing API/mathematical documentation alongside reviewed code. Do not claim this policy-only update supplies CI or branch protection.
3. **Review the canonical trainer separately.** Reconcile the package and experimental trainers and the public DataFrame path with the trainer handoff and later explicit decisions. Do not describe another branch's work as present on `main`; steps 1–5 alone do not complete the fitting interface.
4. **Resolve method choices before extending operators/examples.** Follow the [product decision agenda](docs/plan/v0.1/README.md#next-steps-and-review-decisions) for energy estimators, category distances, IV weights/batching, unscored categorical queries, and the concrete ATT adaptation. Current code is not approval of those choices. Use the small agreed example collection with independent references and actual execution.
5. **Keep identification/search separately controlled.** Preserve the existing pause and review requirements in the [identification handoff](docs/plan/v0.1/identification-and-partial-identification.md). Native user-supplied penalty sweeps and separate contiguous-window extrapolation are the target; selector details remain open. This documentation request does not restart those tasks.
6. **Finish bounded packaging/integration after the relevant units.** Dependency separation remains incomplete (`pytest` is still mandatory runtime; pandas is not yet declared on `main`). Retained examples/notebooks, the chosen documentation build, and installation need release-level verification. Integration and public release/deployment are separate human decisions.

Older contributor assignments remain in dated records, not as new active assignments. The local Sinkhorn implementation and additional discrete-paper code reported in August remain externally reported candidates; this pass does not verify their current status or schedule integration. New discrepancy families and general uncertainty/PFN work are outside the v0.1 implementation units.

## Evidence and uncertainty

- **Freshly inspected:** the default-branch model/operator/trainer source, package metadata, tracked documentation/configuration files, and PR #60's head and latest discussion. Current API guidance is anchored to `main`, not the older August baseline or an unmerged branch.
- **Execution evidence:** fresh checks for this three-file update are reported in its PR. PR #60's earlier reported suite/example results apply to its own implementation baseline, not to `main` or this update.
- **Not established by documentation or a passing suite:** statistical recovery across model families/seeds, general trainer robustness, sharpness of search bounds, comparative merits of alternative discrepancies, or future PFN calibration. GitHub enforcement settings are also unverified here.

## Supporting records

- [v0.1 product specification and open decisions](docs/plan/v0.1/README.md), [ordered implementation plan](docs/plan/v0.1.md), and their linked interface handoffs.
- [September 3/7 decision record](docs/meetings/2026-09-03-and-07/product-and-implementation-plan-updated.md), preserving meeting evidence and later clarifications.
- [August 26 meeting](docs/meetings/2026-08-26-meeting.md) and [August 28 meeting](docs/meetings/2026-08-28-meeting.md), including historical ownership and research directions.
- [Updated August technical review](docs/meetings/2026-08-26-ai-review-updated.md) and [original review](docs/meetings/2026-08-26-ai-review.md). Findings describe their dated source baselines, not current capability lists; preserve those records rather than rewriting them to match later code.
