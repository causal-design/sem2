# SEM2 product definition and implementation plan

**Status:** review draft; no priority or interface below is approved until the team confirms it<br>
**Draft date:** 29 August 2026<br>
**Repository baseline:** `main` at `315df9cfe5d6f1298f9d73ec574e08c87fa59426`

This document proposes what SEM2 should become and how to develop it. It begins with a consolidated feature inventory because the first team decision is not how to implement every idea, but which capabilities belong in the product and in what order.

The inventory primarily reflects the [26 August meeting](../2026-08-26-meeting.md), the [28 August meeting](../2026-08-28-meeting.md), and current GitHub issues. The [updated technical review](../2026-08-26-ai-review-updated.md) and research reports supply implementation evidence and additional candidates. Repository and paper evidence are used to clarify status; they do not turn an unconfirmed proposal into an agreed feature.

## 1. Desirable feature inventory for review

### 1.1 Proposed priority scale

| Priority | Meaning |
|---|---|
| **P0 — product prerequisite** | Required before SEM2 should claim a trustworthy end-to-end fit for its initial supported regimes. Work should be sequenced immediately after the team confirms the product definition. |
| **P1 — near-term product** | Important for the first useful research release, but can build on the P0 contracts and vertical slices. |
| **P2 — planned extension** | Desirable after the core paths are reliable, or dependent on a design decision or benchmark. |
| **P3 — research candidate** | Worth a bounded study, but not a commitment to the package API. |
| **Out of current scope** | Deliberately excluded from the next product stage unless the team changes the scope. |

Priority is based on dependency and product risk, not scientific importance. In particular, uncertainty quantification is scientifically important but depends on first having a reliable estimator, estimand, and diagnostic contract.

Source shorthand used below:

- **M26:** 26 August meeting;
- **M28:** 28 August meeting;
- **TR:** updated technical review;
- **R1/R2:** the two-part [MPhil report](../../reports/MPhilSC_BothReport_Srivathsan_Amruth_Report_2_Starts_at_Page_69.pdf);
- **GH:** linked GitHub issue.

### 1.2 Product, data, and API foundations

| ID | Proposed feature or decision | Priority | Why it is wanted and what would count as progress | Main sources |
|---|---|---:|---|---|
| **F01** | Define a public capability matrix and stability labels | P0 | Every model class, objective, query, and example should be labelled **stable**, **experimental**, **research-only**, or **unsupported**. Unsupported combinations should fail early rather than produce an ambiguous number. | M26, TR |
| **F02** | Introduce a named dataset view/schema | P0 | Map table columns to model nodes by name and record dtype, shape, support/cardinality, observed nodes, missing-value policy, weights, and data regime. Positional tensor order must not be an implicit contract. | M26, M28, TR |
| **F03** | Separate the structural model from observation status | P0 | The graph should describe exogenous variables and structural mechanisms; a dataset view should describe which nodes are observed. Keep `unobs(...)` working through an explicit compatibility path while a data-view API is introduced. | M26, TR |
| **F04** | Define a query/estimand object separate from the graph | P0 | Treatment, outcome, intervention values, target population, contrast, and estimand metadata belong to a query, not generic graph components or global training configuration. | M26, TR |
| **F05** | Define explicit observational and interventional regimes | P1 | A dataset view should state whether rows are observational or generated under an intervention and, if so, which values were fixed. | M26, TR |
| **F06** | Support multiple datasets and observed margins | P2 | Fit one structural model to several datasets that observe different node subsets or regimes, with explicit sample/loss weights and no silent dominance by dataset size or scale. | M26, TR |
| **F07** | Provide a supplied empirical joint distribution for covariates | P0 | Preserve dependence among high-dimensional adjustment covariates by resampling observed rows, rather than forcing users to declare unrelated scalar roots or train an unnecessary generator. The team must decide whether this is a model node, data-level sampler, or another component. | M28 |
| **F08** | Support multidimensional values at model-data boundaries | P1 | Shapes must be first-class and checked. The narrow initial requirement is a multidimensional exogenous covariate object; fully general vector-valued endogenous mechanisms can follow later. This raises the practical priority of the older multivariate-node request. | M28, [GH #40](https://github.com/causal-design/sem2/issues/40) |
| **F09** | Provide explicit, reproducible preprocessing adapters | P1 | Offer reusable categorical encoding, numeric conversion, and missing-value handling without silently choosing scientific assumptions. Learned transforms should be fit on declared training data, serializable, and reflected in the schema. | M28 |
| **F10** | Add a programmatic graph/model constructor | P2 | Keep the concise formula for small models, but avoid requiring very long strings for high-dimensional systems or generated designs. The programmatic and formula routes should lower to the same validated representation. | M26, TR |
| **F11** | Consider ADMG/noise-expansion convenience syntax | P3 | A bidirected-edge or automatic noise expansion could remove tedious explicit exogenous nodes. It requires clear semantics and must not obscure assumptions about latent dependence. | [GH #41](https://github.com/causal-design/sem2/issues/41) |
| **F12** | Resolve `RandomVariable` support semantics | P2 | Exogenous distribution support is local; endogenous support generally depends on the full graph. Decide whether support belongs only on tractable distributions, is optional metadata, or is inferred at model level. Avoid a misleading universal property. | [GH #33](https://github.com/causal-design/sem2/issues/33) |
| **F13** | Standardize result and dimension metadata | P0 | Sampling, finite-state probabilities, fitted parameters, queries, and diagnostics need typed/named outputs whose axes are explicit. In particular, distinguish rows, Monte Carlo draws, finite states, variables, and categories. | M26, TR |
| **F14** | Preserve backward compatibility deliberately | P1 | Existing formulas and supported imports should either keep working or receive a documented deprecation and migration path. Compatibility is a decision, not an accidental side effect of new abstractions. | M26, TR |

### 1.3 Fitting engines, objectives, and model classes

| ID | Proposed feature or decision | Priority | Why it is wanted and what would count as progress | Main sources |
|---|---|---:|---|---|
| **F15** | Make linear-Gaussian fitting the first stable route | P0 | Preserve exact means/covariances and Gaussian likelihood, add one supported trainer/fit entry point, and validate against analytical values and an external reference. | M26, TR, R2 |
| **F16** | Correct and specify continuous energy distance/score | P0 | Choose the exact finite-sample estimator, correct sample/feature axes, remove fixed `Z/X/Y` selection, test analytical cases and gradients, and define batching behavior. | M26, TR |
| **F17** | Consolidate training into one supported interface | P0 | Reconcile the package and experiment trainers. Support objective-specific closures, stopping, seed control, restart policy, traces, callbacks or structured progress, and a returned fit result. The existing trainer is a starting point, not yet the product contract. | M26, TR, [GH #20](https://github.com/causal-design/sem2/issues/20) |
| **F18** | Add fit and optimizer diagnostics | P0 | Report convergence status, objective decomposition, parameter change, gradient/step information where available, restarts, elapsed work, and post-fit model-versus-data checks. A fixed iteration count is not a convergence claim. | M26, M28, TR, R2 |
| **F19** | Make device, dtype, and random-state behavior explicit | P0 | Validate data/model device and dtype, move registered state correctly with `.to(...)`, expose deterministic seed handling, and test CPU plus any supported accelerators. Do not rely only on automatic device selection. | TR, [GH #29](https://github.com/causal-design/sem2/issues/29) |
| **F20** | Select fitting operators from declared capabilities | P2 | The package may recommend exact likelihood, moment fitting, or simulation-based discrepancy from model type and available data, but the selected objective and assumptions must remain visible and overridable. | M26, R2 |
| **F21** | Review and integrate Sinkhorn divergence | P1 | Bring the external implementation into a bounded PR with the exact divergence, regularization/debiasing convention, backend, shapes, gradients, device behavior, deterministic examples, and comparisons with energy score. | M26, TR, R2 |
| **F22** | Decide whether MMD belongs in the product | P3 | MMD is discussed but unimplemented and may have poor optimization geometry in the intended problems. Run a bounded benchmark before creating a public API; remove or clearly reject dormant configuration in the meantime. | M26, TR, R1 |
| **F23** | Benchmark projected or sliced discrepancies | P2 | Compare projected/sliced energy or Wasserstein variants in dimensions where ordinary energy distance is weak. Vary dimension, tails, sample size, projection count, misspecification, and compute budget. | M26, TR |
| **F24** | Decide whether the ICA/change-of-variables likelihood is a package feature | P2 | Reconcile the report implementation with current source, specify assumptions and identifiability, and compare with FastICA before integrating a stable operator. Otherwise retain it as research code. | TR, R2 |
| **F25** | Complete finite discrete fitting as a named regime | P1 | Integrate the remaining research code, define exact finite-state probability and likelihood/energy routes, and separate generic discrete fitting from the binary-IV application. | M26, TR, [GH #19](https://github.com/causal-design/sem2/issues/19), R1 |
| **F26** | Add state-space and memory guards | P1 | Before exact enumeration, report state count and estimated memory and reject infeasible cases. Make Monte Carlo integration over exogenous variables explicit. | TR, R1 |
| **F27** | Evaluate Gumbel-Softmax as an approximate discrete route | P2 | Implement only after temperature, gradient bias, training/evaluation semantics, and comparison with exact small problems are specified. Do not describe it as exact likelihood. | [GH #19](https://github.com/causal-design/sem2/issues/19), TR |
| **F28** | Keep “treat discrete as continuous” out of the ordinary API | Out of current scope | Developers explicitly decided not to expose this as the normal discrete solution; an advanced user can still choose a continuous mechanism deliberately. | [GH #19](https://github.com/causal-design/sem2/issues/19) |
| **F29** | Define mixed discrete/continuous semantics | P3 | Current classification does not yield an end-to-end fitter. Fail early now; undertake a design study only after continuous and finite-discrete paths are reliable. | M26, TR, R2 |
| **F30** | Add linear path analysis | P2 | Define path selection, path enumeration, fixed/learned coefficient semantics, and tests against hand-calculated path monomials before filling the current placeholder. | [GH #42](https://github.com/causal-design/sem2/issues/42) |
| **F31** | Improve nonlinear fitting efficiency | P2 | After correctness, benchmark GPU batching, loss approximations, early stopping, model-specific initialization, and reuse of observed-target calculations. Performance changes need numerical-equivalence tolerances. | R2 |

### 1.4 Causal queries, identification, and uncertainty

| ID | Proposed feature or decision | Priority | Why it is wanted and what would count as progress | Main sources |
|---|---|---:|---|---|
| **F32** | Test intervention semantics directly | P0 | Verify equation replacement, named node selection, shapes, common-random-number comparisons where appropriate, and invariance of unaffected mechanisms. State that computation under a graph does not validate the graph. | M26, TR |
| **F33** | Make assumptions and estimands inspectable | P1 | A query result should identify the supplied graph/model, intervention, observed margin, target population, contrast, and status of identification assumptions rather than returning an unqualified scalar. | M26, TR |
| **F34** | Build a reusable partial-identification workflow | P1 | Add signed penalty sweeps or constrained alternatives, continuation, fit tolerances, multistart optimization, branch diagnostics, and checks that reported extrema remain observationally compatible. | M26, TR, R1, [AAAI report](../../reports/AAAI_SEM2.pdf) |
| **F35** | Validate bounds against known references | P1 | Use binary IV and other small problems with analytical, linear-programming, or independently computed bounds. Numerical extrema are not “sharp bounds” without this evidence and an identification argument. | M26, TR, R1 |
| **F36** | Distinguish kinds of uncertainty in results and APIs | P1 | Keep sampling uncertainty, Monte Carlo error, optimization/restart variation, model/graph sensitivity, and structural non-identification separate. Also distinguish confidence regions, posterior regions, identified sets, and numerical sensitivity ranges. | M26, M28, TR |
| **F37** | Add linear-model uncertainty after the fit contract is stable | P2 | Evaluate Hessian/observed-information, delta-method, sandwich, and bootstrap routes against known linear-Gaussian behavior. The latest meeting places reliable point estimation first. | M28, TR, R2 |
| **F38** | Study nonlinear and causal-query uncertainty | P3 | A full refitting bootstrap is an immediate candidate; influence-function or one-step methods require explicit regularity conditions. Treat this as statistical research, not a generic wrapper around an unreliable fit. | M26, M28, TR, R1, R2 |
| **F39** | Add distributional causal summaries where justified | P2 | Fitted generative models can report interventional variances, quantiles, and tail probabilities as model-implied summaries. Do not present these as standard errors for a causal contrast. | R2 |

### 1.5 Evaluation, applications, documentation, and release quality

| ID | Proposed feature or decision | Priority | Why it is wanted and what would count as progress | Main sources |
|---|---|---:|---|---|
| **F40** | Establish a simulation suite with known targets | P0 | Test parameter, distribution, intervention, and causal-query recovery across seeds and sample sizes before using real applications as evidence of correctness. | M28, TR |
| **F41** | Maintain one executable vertical slice per claimed regime | P0 | A clean environment should run a linear-Gaussian fit, nonlinear-continuous fit, and finite-discrete fit only when each is claimed supported. Examples should fit before querying or intervening. | M26, TR |
| **F42** | Keep a reduced RHC smoke test | P1 | Use one or two covariates to exercise data preparation, fitting, and intervention. State that it is not a substantive estimate for the full high-dimensional RHC population. Resolve death-versus-survival outcome coding. | M28, [GH #47](https://github.com/causal-design/sem2/issues/47) |
| **F43** | Add a low-dimensional proximal/proxy example | P1 | Use a fixed graph with known theory or reference results to test proxy-variable semantics and hidden-confounding workflows. | M28, R1, R2 |
| **F44** | Retain bounded external-validation benchmarks | P2 | Where maintained, compare linear calibration with OLS, measurement models with lavaan, source separation with FastICA, and RDD with rdrobust. Specialist methods remain reference tools, not competitors SEM2 must always beat. | R2 |
| **F45** | Consolidate and make notebooks executable | P1 | Remove duplication only after owner review; update public imports and calls; declare example dependencies; record seeds, provenance, runtime, and status; run maintained notebooks in CI. | [GH #47](https://github.com/causal-design/sem2/issues/47) |
| **F46** | Complete user and API documentation | P1 | Keep the current overview, then document stable fit workflows, data/query contracts, diagnostics, supported regimes, failure modes, and causal interpretation. Generate API reference from inspected public interfaces, not aspirational classes. | [GH #38](https://github.com/causal-design/sem2/issues/38), M26, TR |
| **F47** | Make datasets and examples reproducible | P1 | Record source, license/terms, checksum where redistribution permits, transformation, seeds, environment, and numerical acceptance thresholds. CI should not depend on manual upload. | TR, [GH #47](https://github.com/causal-design/sem2/issues/47) |
| **F48** | Separate runtime, test, and example dependencies | P1 | `pytest` should not be a runtime requirement. Declare optional extras or a development lock for notebooks, plotting, external benchmarks, and research operators. | TR, [GH #47](https://github.com/causal-design/sem2/issues/47) |
| **F49** | Add normal open-source project metadata | P2 | Before public release, decide license, contribution guide, code of conduct if desired, versioning, supported Python/PyTorch versions, and release notes. | M28, TR |
| **F50** | Add CI for supported behavior | P0 | Run unit and integration tests, link checks, package installation, and reduced maintained examples. Add accelerator jobs only for devices the project claims to support. | M26, M28, TR |
| **F51** | Define benchmark and regression budgets | P2 | Record acceptable numerical error, runtime/memory ceilings, and stochastic variation for expensive fitting paths so optimization work does not silently change statistical behavior. | TR, R1, R2 |

### 1.6 Longer-horizon research candidates

| ID | Proposed feature or decision | Priority | Why it is wanted and what would count as progress | Main sources |
|---|---|---:|---|---|
| **F52** | Run a bounded PFN posterior study | P3 | Begin with a fixed supplied graph and transparent prior, targeting either a small parameter vector or one causal query. Compare calibration, coverage/proper score, prior shift, sample-size shift, compute, and failure modes with direct fitting and a high-quality posterior baseline. | M26, TR |
| **F53** | Consider graph-as-input amortization only after F52 | P3 | A PFN that consumes a supplied graph better matches SEM2 than graph discovery, but greatly expands the training distribution and validation burden. | M26, TR |
| **F54** | Keep graph discovery outside the current product | Out of current scope | The project currently aims to fit and query a graph supplied by the user. Learning graph structure raises distinct prior, equivalence, and identification problems. | M26, M28, TR |
| **F55** | Expand to more real policy and domain datasets | P3 | Add applications only when the core contracts can state assumptions, reproduce data, benchmark behavior, and distinguish a smoke test from estimator validation. | M28, R2 |

### 1.7 Completed foundations to preserve rather than redesign automatically

The inventory should not cause an agent to reopen completed architectural work without evidence. Current source already contains a robust parser and typed lowering, a separate DAG representation, named sampling outputs, a `SEM2Linear` class with exact moments and Gaussian likelihood, fixed-coefficient buffers that move with the model, and finite-state probability calculations. These correspond to work discussed in closed issues such as [#30](https://github.com/causal-design/sem2/issues/30), [#26](https://github.com/causal-design/sem2/issues/26), [#31](https://github.com/causal-design/sem2/issues/31), and [#29](https://github.com/causal-design/sem2/issues/29).

The next architecture should preserve these useful pieces unless a reviewed requirement demonstrates that they block the product. “AI can rewrite it” is not such a requirement.

### 1.8 Priority conflicts for the team to resolve

1. **Uncertainty timing.** R2 calls uncertainty the first priority; M28 explicitly prefers reliable point estimation and evaluation before a general confidence-interval interface. This draft follows the later meeting: uncertainty semantics are P1, linear implementations are P2, and general nonlinear uncertainty is P3.
2. **Multivariate nodes.** GH #40 originally called general multivariate nodes low priority. M28 identified an immediate need to preserve a high-dimensional covariate vector. This draft separates a narrow empirical multivariate input requirement (P0/P1) from fully general vector-valued mechanisms (P2 or later).
3. **Discrete strategy.** GH #19 records a decision not to present “pretend discrete is continuous” as the ordinary solution. Exact finite-state fitting and a tested Gumbel route remain distinct products with different semantics.
4. **Automatic operator choice.** R2 favors more automation. The meetings emphasize explicit assumptions. This draft makes transparent manual selection the initial contract and treats recommendation/automatic selection as P2.
5. **Broad framework versus reliable supported subset.** The long-term aim is broad, but the first release checkpoint should claim only regimes with an executable, tested fit and diagnostics.

## 2. Proposed software product

### 2.1 Product statement

SEM2 should be a research-grade Python library in which a user:

1. states a structural model or graph they are prepared to assume;
2. maps one or more named datasets and regimes to that model;
3. chooses or confirms a fitting objective supported for that model/data combination;
4. obtains a structured fit with numerical and model-fit diagnostics;
5. declares a statistical or causal query; and
6. receives a result whose assumptions, identification status, and uncertainty type are not overstated.

The differentiating value is a common structural representation across exact linear calculations, likelihood-based models, and differentiable generative fitting—not a claim that one generic optimizer should replace every specialized estimator.

### 2.2 Initial target users and tasks

The first target users are researchers and method developers who already have a scientifically supplied graph and need to prototype or compare structural models beyond a standard linear-Gaussian analysis. Initial tasks are:

- fit a linear-Gaussian SEM and inspect coefficients, moments, likelihood, and interventions;
- fit a small nonlinear continuous SEM by a verified distributional objective;
- fit a small finite-discrete SEM and evaluate interventions or a bounded partial-identification workflow;
- compare model-implied observed distributions with named data; and
- run reproducible simulations and reference comparisons.

The initial product is not a push-button causal-discovery tool, a guarantee that a supplied graph is correct, or a substitute for mature specialist software when only its standard estimand is required.

### 2.3 Conceptual product objects

The exact class names remain a design decision, but the product needs these separate concepts:

| Concept | Responsibility | Must not silently do |
|---|---|---|
| **Structural model specification** | Nodes, directed structure, exogenous laws or supplied samplers, and endogenous mechanisms | Decide which variables are observed in every dataset or assign treatment/outcome roles |
| **Dataset view/schema** | Named node mapping, shapes/types/support, observed margin, regime, weights, and explicit preprocessing | Infer column order, impute values, or declare interventions without a recorded policy |
| **Query/estimand** | Intervention, outcome, contrast, target population, and required assumptions | Treat a model-computed intervention as proof of identification |
| **Objective/operator** | Likelihood, moments, distributional discrepancy, causal functional, or a deliberate composition | Hide application-specific `Z/X/Y` conventions inside a generic class |
| **Fit plan/trainer** | Optimizer, stopping, seeds, batching, restarts, and supported model/objective compatibility | Return success solely because the loop exhausted its iterations |
| **Fit result** | Fitted state, objective history, diagnostics, warnings, provenance, and query evaluation entry point | Collapse all errors and uncertainty into one number |
| **Capability registry** | Supported combinations, stability level, required data properties, and failure messages | Advertise a feature merely because a notebook or external branch contains related code |

A future API might follow this flow, but the sketch is illustrative rather than an approved signature:

```python
model = Model.from_formula(...)
data = DatasetView.from_table(
    table,
    columns={"age": "age", "treatment": "A", "outcome": "Y"},
    regime=Observational(),
    missing=RejectMissing(),
)
query = AverageTreatmentEffect(treatment="A", outcome="Y", values=(0, 1))
plan = FitPlan(objective=EnergyDistance(), optimizer=..., seed=...)
result = fit(model, data, plan)
result.check_diagnostics()
estimate = result.evaluate(query)
```

The design review should choose names only after agreeing the responsibilities and compatibility rules.

### 2.4 Supported-regime policy

A regime becomes **stable** only when it has:

- an explicit model/data/query contract;
- early compatibility checks;
- objective-level numerical and gradient tests where applicable;
- an end-to-end recovery test with a known answer;
- convergence and fit diagnostics;
- one maintained example executable from a clean environment; and
- documentation that states assumptions and unsupported cases.

The proposed first stable sequence is linear Gaussian, nonlinear continuous, then small finite discrete. Partial identification can be a named experimental workflow on top of the finite-discrete layer until bound validation and diagnostics satisfy stronger criteria.

## 3. Proposed architecture boundaries

### 3.1 Preserve the current parser-to-model core

The current formula parser, typed statements, DAG construction, node modules, specialized model classes, and named sampling outputs are useful foundations. New work should add contracts around that core before attempting a rewrite.

### 3.2 Put data concerns at the boundary

A dataset view should validate and transform external data into named model values. Preprocessing should be explicit and reproducible:

- categorical levels and unknown-level behavior are recorded;
- numeric scaling and imputation are opt-in policies;
- transformations fit on data expose their learned state;
- the transformed schema retains links to source columns; and
- train/evaluation leakage is testable.

For empirical covariate sampling, the initial primitive should preserve complete observed rows and optional weights. It should define replacement, random-state, device, and missing-value behavior. It should not be described as a smooth density estimate.

### 3.3 Keep generic infrastructure free of design-specific roles

Binary IV, RDD, proximal-proxy, and RHC workflows should compose generic model, data, objective, trainer, and query components. If a design needs special grouping or bound tracing, expose a clearly named adapter or workflow rather than embedding assumptions in `EnergyLoss` or a global configuration.

### 3.4 Prefer explicit capability dispatch to accidental polymorphism

Model type detection can help select or reject operators, but every supported combination should have a declared contract. Mixed models, unavailable likelihoods, infeasible discrete state spaces, and non-differentiable routes should fail before training begins with a precise explanation and a suggested supported alternative where one exists.

### 3.5 Keep research integrations bounded

Sinkhorn, ICA, Gumbel-Softmax, PFNs, and new partial-identification algorithms should enter through review branches with their objective, assumptions, dependencies, tensor/device contract, deterministic example, independent reference, and benchmark criteria. Integration should not broaden package claims beyond the tested route.

## 4. Staged implementation plan

### Stage 0 — confirm the product before expanding it

**Work**

1. Review F01–F55 and mark each **accept**, **revise**, **defer**, or **reject**.
2. Confirm the first stable regimes and intended users.
3. Resolve the five priority conflicts in Section 1.8.
4. Agree conceptual responsibilities for model, data, query, objective, trainer, and result.
5. Convert accepted P0 items into bounded GitHub issues with dependencies and acceptance criteria.

**Exit criterion:** a team-approved product scope and ordered P0 backlog. No architecture-wide agent implementation should start before this gate.

### Stage 1 — establish the contracts with a linear-Gaussian vertical slice

**Work**

- implement the minimum named dataset view and query contract;
- define fit-plan, fit-result, diagnostics, and capability-check behavior;
- route the existing exact linear likelihood through the supported fit interface;
- add device/dtype/shape and named-column validation;
- validate with an analytical example and an external linear reference; and
- create the clean-environment example and CI job.

**Exit criterion:** a user can map named data to a linear-Gaussian model, fit it, inspect diagnostics, evaluate a declared intervention/query, and reproduce the result without a manual optimization loop.

### Stage 2 — make nonlinear continuous point fitting trustworthy

**Work**

- specify and repair energy distance;
- generate observed model tables from the dataset view rather than hard-coded names;
- support an empirical multidimensional covariate distribution;
- add trainer batching, stopping, trace, seed, and restart behavior needed by stochastic objectives;
- validate gradients and hand-checkable objective values;
- run simulation recovery across seeds and sample sizes; and
- retain a reduced RHC smoke test with explicit outcome coding and limitations.

**Exit criterion:** one small nonlinear continuous design recovers a known target within reviewed tolerances, reports convergence/fit diagnostics, and runs through the public interface.

### Stage 3 — complete the finite-discrete and partial-identification paths

**Work**

- review and integrate remaining discrete code;
- separate finite-state model calculations from binary-IV adapters;
- add state-space/memory guards and dimension metadata;
- implement a supported finite-discrete fit with reference tests;
- build penalty/constrained sweeps with fit tolerances, restarts, and branch diagnostics; and
- compare numerical bounds with known binary-IV or other exact references.

**Exit criterion:** a small finite-discrete example fits through the public interface, and the experimental partial-identification workflow reports a validated numerical range without calling it a sharp bound beyond the evidence.

### Stage 4 — integrate selected methods and strengthen the product surface

**Work**

- review Sinkhorn and benchmark it against energy distance;
- decide projected/sliced discrepancies, ICA, path analysis, and Gumbel-Softmax from bounded studies;
- consolidate notebooks and data provenance;
- complete user/API documentation and optional dependency groups;
- add performance and numerical-regression budgets; and
- address accepted P1/P2 support for multiple datasets or vector-valued mechanisms.

**Exit criterion:** accepted extensions meet the same contract and verification standard as the core, and rejected candidates leave no misleading configuration or capability claim.

### Stage 5 — uncertainty and research extensions

**Work**

- add and validate linear-model uncertainty first;
- study full-refit bootstrap and regularity for nonlinear causal queries;
- define any mixed-model project before implementation;
- run the bounded fixed-graph PFN study; and
- consider graph-as-input amortization only if the fixed-graph study is calibrated and useful.

**Exit criterion:** every uncertainty output says exactly what region or error source it represents; research prototypes remain separately labelled until they satisfy product gates.

## 5. Workflow for development with AI coding agents

### 5.1 Allocation of responsibility

Humans own:

- product scope and priorities;
- scientific assumptions and estimands;
- architecture decisions and public APIs;
- acceptance criteria and benchmark meaning;
- dependency and compatibility policy;
- review, merge, release, and capability claims.

Coding agents may:

- inspect the approved issue and current source;
- propose implementation alternatives and risks;
- draft a bounded plan;
- implement the accepted plan with tests and documentation;
- run verification and summarize evidence; and
- review a diff for consistency or missing tests.

An agent's plan, test, or review is evidence for human review, not approval. Agents must not merge, redesign the package, add broad dependencies, or expand scope merely because doing so appears convenient.

### 5.2 Required issue specification

Every implementation issue should include:

1. **Feature IDs and user outcome.** Link to this inventory and state what a user can do afterward.
2. **Current evidence.** Cite inspected code, failing/present tests, meeting decision, issue, or benchmark.
3. **In scope and out of scope.** Prevent adjacent cleanup from becoming an unbounded refactor.
4. **Behavioral contract.** Inputs, outputs, shapes, dtypes, devices, errors, determinism, and compatibility.
5. **Statistical contract.** Objective, assumptions, estimand, approximation, and what is not guaranteed.
6. **Acceptance tests.** Analytical/reference values, gradients, integration behavior, failures, and tolerances.
7. **Documentation change.** State which of `README.md`, `PROJECT.md`, `AGENTS.md`, API docs, or examples must change.
8. **Dependencies and migration.** Record new packages, deprecations, data changes, or performance costs requiring approval.

### 5.3 Change lifecycle

1. **Select an approved task.** Work begins from the prioritized backlog, not an agent-generated wishlist.
2. **Refresh evidence.** Inspect current source, tests, open PRs, and the exact branch baseline.
3. **Plan before code.** The agent proposes files, interfaces, alternatives, tests, migration, and risks. A human accepts or revises the plan.
4. **Use a bounded branch.** One coherent behavior change per branch; avoid architecture-wide batches.
5. **Implement against behavior.** Add focused tests with the code. Prefer the smallest change that satisfies the contract and preserves accepted architecture.
6. **Run layered verification.** Start with focused tests, then integration/reference checks, full suite, examples affected, link checks, and `git diff --check`.
7. **Perform adversarial review.** Ask why each new abstraction, file, option, and dependency is needed. Use a second agent for a structured review if useful, but require human resolution of substantive findings.
8. **Publish a reviewable PR.** The PR describes behavior, scientific assumptions, tests actually run, limitations, migration, and any unverified claims.
9. **Human review and merge.** At least one developer reviews architecture and code; a method owner reviews statistically substantive changes. Merge remains a human decision.
10. **Update maintained memory.** After behavior is accepted, update capability status and durable guidance without rewriting dated meeting/history files.

### 5.4 Review gates

| Gate | Required question | Typical evidence |
|---|---|---|
| **Product gate** | Is this an accepted feature at the right priority? | confirmed inventory item and issue |
| **Architecture gate** | Does it respect model/data/query/operator boundaries and avoid unnecessary abstractions? | accepted plan, focused diff, compatibility note |
| **Statistical gate** | Is the objective/query mathematically defined and are claims conditional on stated assumptions? | formula/reference, known-answer test, method-owner review |
| **Implementation gate** | Does behavior hold for values, gradients, axes, dtypes, devices, seeds, and failures? | automated tests and executed examples |
| **Integration gate** | Does it compose through the public fit/query path without breaking supported regimes? | vertical-slice and full-suite results |
| **Publication gate** | Is the PR understandable, bounded, documented, and honestly verified? | human review; no unresolved critical finding |

### 5.5 Rules intended to prevent agent-driven architecture drift

- Do not ask an agent to “implement all features” or “clean up the architecture.”
- Do not accept a new abstraction without at least two concrete uses or a product requirement that needs the extension point now.
- Do not duplicate trainer, schema, result, or operator hierarchies to avoid understanding existing code.
- Do not silently change formula semantics, node order, tensor axes, default estimands, or intervention behavior.
- Do not add fallback behavior that turns an unsupported combination into an apparently successful fit.
- Do not add optional research methods to the stable import surface before their capability status is decided.
- Do not treat generated documentation, comments, or type annotations as tests of behavior.
- Do not delete notebooks or research artifacts solely because they are stale; obtain owner review and preserve useful provenance.
- Do not use passing tests to claim identification, calibration, or causal validity that the tests do not establish.
- If a change cannot be explained succinctly by the implementing agent and a reviewing developer, reduce or split it before merge.

## 6. Verification strategy

### 6.1 Test layers

| Layer | Required coverage |
|---|---|
| **Syntax and construction** | formula/programmatic equivalence, clear errors, cycle/parent checks, compatibility migration |
| **Data contract** | named mapping, order independence, shape/type/support errors, missingness policies, regime and weight validation |
| **Numerical operator** | analytical values, unbiased/biased estimator convention, axes, gradients, dtype/device, batching |
| **Model sampling** | shapes, named outputs, seeds, interventions, empirical-row preservation, finite-state normalization |
| **Fit integration** | recovery from known synthetic data, convergence and deliberate non-convergence, restarts, saved diagnostics |
| **Causal query** | correct equation replacement, explicit nodes/values, known contrasts, assumptions metadata |
| **Partial identification** | known bounds, fit tolerances, monotonic/branch diagnostics where applicable, restart sensitivity |
| **Uncertainty** | coverage or reference behavior for the exact method claimed; separation from optimization and identification ranges |
| **Documentation/example** | clean-environment execution, declared dependencies, data provenance, current public API |
| **Regression/performance** | numerical tolerance, stochastic variation, runtime and memory for accepted benchmark sizes |

### 6.2 Claim discipline

Every PR and result should distinguish:

- **observed:** source behavior or executed output;
- **reference-supported:** agreement with an analytical or independent implementation;
- **meeting/issue intent:** a requested direction not yet implemented;
- **inferred design:** a proposed way to satisfy the intent; and
- **unverified:** performance, calibration, identification, or external behavior not established by the available checks.

A release should not convert “reported in a research report,” “available on a contributor branch,” or “implemented by an agent” into “supported by SEM2” without the relevant integration gate.

## 7. Decisions requested from the team

The team can review this draft efficiently by answering the following in order:

1. Is the product statement in Section 2 the intended product?
2. Which regimes must be stable in the first research release?
3. For every inventory row, is the decision **accept**, **revise**, **defer**, or **reject**?
4. Are the proposed P0 items truly prerequisites, and is any prerequisite missing?
5. Where should a supplied empirical joint covariate distribution live?
6. How much preprocessing should SEM2 own versus require from callers?
7. Is moving observation status out of the formula accepted, and what compatibility period is required?
8. Should explicit operator choice precede automatic selection?
9. Which discrete route is the first supported product: exact finite-state fitting, Gumbel-Softmax, or both with separate labels?
10. Which external implementations are ready enough for review branches?
11. What human reviews are mandatory for architecture and statistical-method changes?
12. After these answers, who owns each accepted P0 issue and its acceptance criteria?

Only after those decisions should the conceptual objects in Section 2 be turned into concrete classes and signatures.

## 8. Evidence boundaries and source index

Directly inspected for this draft:

- current `sem2/` source, package metadata, trainer, operators, and tests at the stated baseline;
- all open GitHub issues and the closed issues most relevant to the present architecture;
- the 26 and 28 August meeting records;
- the updated technical review; and
- the two-part MPhil report and discrete partial-identification paper stored under `docs/reports/`.

The priority assignments, conceptual API, stages, and AI workflow are proposals synthesizing those sources. They require team confirmation. Research-report results and contributor-reported external implementations are not treated as current package capabilities.

Key source links:

- [26 August meeting](../2026-08-26-meeting.md)
- [28 August meeting](../2026-08-28-meeting.md)
- [Updated technical review](../2026-08-26-ai-review-updated.md)
- [Current project status](../../../PROJECT.md)
- [MPhil reports](../../reports/MPhilSC_BothReport_Srivathsan_Amruth_Report_2_Starts_at_Page_69.pdf)
- [Discrete partial-identification paper](../../reports/AAAI_SEM2.pdf)
- GitHub issues [#19](https://github.com/causal-design/sem2/issues/19), [#33](https://github.com/causal-design/sem2/issues/33), [#38](https://github.com/causal-design/sem2/issues/38), [#40](https://github.com/causal-design/sem2/issues/40), [#41](https://github.com/causal-design/sem2/issues/41), [#42](https://github.com/causal-design/sem2/issues/42), and [#47](https://github.com/causal-design/sem2/issues/47)
