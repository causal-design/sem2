# SEM2: decisions on the 55-item proposal

**Status:** meeting decisions with subsequent Slack review clarifications.

This record preserves the decisions and evidence for all 55 items. The separate [v0.1 implementation plan](../../plan/v0.1.md) defines the intended package, examples, tests, and next steps. It replaces the former Sections 2–4 here rather than duplicating them.

Here, **high priority / v0.1** means the next package milestone; **medium priority / v0.2** means the following development round; **low priority** means no scheduled implementation. These version labels follow the request for this update, rather than the original draft's P0–P3 scale. They do not announce a public release or change the package version. A v0.2 research task can deliver a decision or experiment rather than a supported feature.

The tables distinguish an accepted item, a reduced scope, a deferral, and an unresolved question. Where a meeting said only "later", the proposed version assignment is marked explicitly. A skipped item is not silently approved. The original plan's detailed architecture and implementation stages were not adopted wholesale.[4]

## Reading the records

- [Initial product and implementation plan](product-and-implementation-plan.md): the original 55-item proposal.[1]
- [Foundations transcript](transcript-f01-f09.txt) and [supplied minutes](summary-f01-f09.md): F01 onwards, including the short graph-construction discussion at the end.[2]
- [Fitting transcript](transcript-f12-f27.txt) and [supplied minutes](summary-f12-f27.md): starts by saying F11 was skipped, then covers F12 through F31, despite the filename.[3]
- [Queries and planning transcript](transcript-f32-f55.txt) and [supplied minutes](summary-f32-f55.md): F32–F55, the revised user workflow, and next steps.[4]

The filenames and supplied minutes are preserved. This folder groups the records for 3 and 7 September without inferring a recording's date from its upload order. The transcripts, rather than the AI-generated minutes, support the meeting decisions below; subsequent review clarifications are identified separately. Citations identify the recording; F-numbers, timestamps where available, and quoted phrases locate the relevant passage. The Sources section also links to the inspected, pre-move versions.

### Review clarifications

The following decisions were supplied in the Slack follow-up to this draft and resolve the questions raised in the first review:

- **F02/F03:** keep minimal named-data mapping and observation-status separation in v0.1; implement the richer dataset schema in v0.2.
- **F28:** retain both the finite-discrete route and the base `SEM2` route. For deliberate numeric treatment of categorical variables in `SEM2`, provide an explicit conversion helper analogous in purpose to R's `as.numeric`. The helper's name and conversion rules still need design review; its inclusion is settled. This narrow helper belongs in v0.1, while general categorical preprocessing under F09 remains in v0.2.

### Points to confirm in this review

1. **F12 support semantics.** The team called this a high-priority design question but deferred the technical discussion. v0.1 needs that decision, not an agent-invented universal support property.[3]
2. **F10 interpretation.** The brief graph-editor discussion does not settle the original programmatic-constructor proposal. Neither is a v0.1 requirement.[2]
3. **Remaining design choices.** Review the small query/result representation, conversion helper, graph-plot preview, and sample documentation page through the [v0.1 next steps](../../plan/v0.1.md#next-steps-and-review-decisions). Confirm the v0.2 assignments labelled "proposed" below.[3][4]

## 1. Decisions on the 55 items

### 1.1 Model, data, and query foundations

| ID | Original item, shortened | Priority / milestone | Meeting decision and limit |
|---|---|---|---|
| F01 | Capability matrix and stability labels | Low / dropped for now | Drop the formal classification exercise at this early stage. Ask agents for focused changes instead; do not add a registry merely to label everything experimental. [2], 19:09–22:19 |
| F02 | Named dataset view/schema | High / v0.1 minimal; Medium / v0.2 richer schema | Keep minimal named column-to-node mapping in v0.1 and implement the richer dataset schema in v0.2. The [Slack review clarification](#review-clarifications) resolves the conflict between the detailed discussion and closing recap. Meeting background: [2], 22:19–27:27; [4], "Put data concerns at the boundary" |
| F03 | Model separate from observation status | High / v0.1 | Keep observed/latent status relative to the dataset rather than permanently in the structural model, using the minimal F02 boundary. The [Slack review clarification](#review-clarifications) settles its timing; the interface still needs design review. Meeting background: [2], 24:23–27:27; [4], closing architecture recap |
| F04 | Query/estimand separate from the graph | High / v0.1 | Accept. Users should choose arbitrary treatment/outcome names and query a fitted model without editing global training configuration. Query-specific fitting must still be possible for partial identification. No large query-object hierarchy was agreed. [2], 27:27–30:28; [4], product-workflow discussion |
| F05 | Observational/interventional data regimes | Low / later | Defer fitting from combined observational and experimental data. This does not defer computing interventions from a supplied model. [2], after 30:28 |
| F06 | Multiple datasets and observed margins | Low / later | Defer joint fitting across datasets or margins. One dataset/observational margin is enough for the next milestone. [2], F06 |
| F07 | Empirical joint covariate distribution | Medium / v0.2 | Useful, but not required to test the core now. Preserve covariate dependence; leave open whether empirical sampling belongs in model fitting, a sampler, or the data loader. [2], 32:26–44:06 |
| F08 | Multidimensional node values | Medium / v0.2 | Prioritize vector-valued nodes ahead of the empirical-distribution extension, but not as a v0.1 prerequisite. Do not assume a scalar node per covariate means those covariates are independent. [2], 43:11–F08 discussion |
| F09 | Preprocessing and categorical encoding | Medium / v0.2 general framework | General preprocessing and categorical encoding remain in v0.2. The explicit categorical-to-numeric helper requested under F28 is a narrow v0.1 exception, not approval of the full framework. [2], 46:52–50:46; [Slack review clarification](#review-clarifications) |
| F10 | Programmatic graph/model constructor | Low / unscheduled — **interpretation provisional** | The original item is a programmatic constructor, but the closing discussion refers to an existing graphical tool and calls it low priority. No explicit approval of the programmatic constructor is recorded. Keep it separate from F39's simple graph plot. [1][2], 52:22–52:34 |
| F11 | ADMG/noise-expansion convenience syntax | Low / skipped | The fitting recording explicitly says to skip F11. No implementation decision or release commitment is recorded. Internal DAG-to-ADMG projection for F33 is a separate requirement. [3], opening |
| F12 | `RandomVariable` support semantics | High / v0.1 **design decision first** | Escalate above the original P2: decide where support information belongs before new code makes inconsistent assumptions. Technical design is deferred to a focused discussion, not resolved by this plan. [3], opening F12 discussion |
| F13 | Result and dimension metadata | Medium / v0.2, subject to clarification | Ask the agent what specifically needs changing. The team saw possible documentation/convention benefits but did not approve the original typed-output redesign. Document the v0.1 outputs actually used; decide broader standardization separately. [3], F13 |
| F14 | Backward compatibility | Low / rejected at this stage | Do not build compatibility/deprecation machinery now. Necessary changes can break the early interface; retain useful code and update maintained examples rather than refactoring gratuitously. Reconsider once users rely on a stable version. [3], F14 |

### 1.2 Fitting and model classes

| ID | Original item, shortened | Priority / milestone | Meeting decision and limit |
|---|---|---|---|
| F15 | Reliable linear-Gaussian fitting | High / v0.1 | Use the existing linear-Gaussian route as a core example and serious regression test, not a reason to rewrite it. Validate against known values and the agreed external references. [3], F15; [4], F44 |
| F16 | Continuous energy distance/score | High / v0.1 | Inspect and explain the current problems, remove hardcoded node selections, and repair/stabilize the loss. Humans review the objective and proposed fix before implementation. [3], F16, including 08:39–08:45 |
| F17 | One supported trainer | High / v0.1 minimal; Medium / v0.2 extensions | Consolidate the package and experiment versions into one minimal trainer sufficient for the examples. Defer a large menu of stopping, restart, callback, and tuning features. [3], 11:28–15:25 |
| F18 | Fit and optimizer diagnostics | High / v0.1 minimal; Medium / v0.2 extensions | Report loss and gradient magnitude and provide a way to plot training progress. Humans inspect these; no automatic or certified convergence decision is required now. [3], 16:08–17:32 |
| F19 | Device, dtype, and random-state behavior | Medium / v0.2 broader work | Agent-run v0.1 tests target CPU and small correctness cases. Max can check GPU behavior manually. Defer GPU/HPC infrastructure and broad hardware testing, without introducing avoidable tensor mismatches. [3], 17:32–25:59 |
| F20 | Automatic fitting-operator selection | Low / later | Keep operator selection simple; automatic recommendation is not urgent. Existing checks against incompatible model/operator combinations need not be replaced with a new dispatch system. [3], 25:59–26:47 |
| F21 | Sinkhorn divergence | Medium / v0.2 | Defer integration until the core and its tests work. Then review and compare it with the existing loss. [3], 27:03–27:35 |
| F22 | MMD | Medium / v0.2 evaluation | Consider with Sinkhorn and the other losses in the next phase. This is an evaluation task, not a claim that MMD is already useful or must become public functionality. [3], 27:35–28:20 |
| F23 | Projected/sliced discrepancies | Medium / v0.2 evaluation | Compare these with energy and Sinkhorn after the core is ready. No large parameter sweep is part of v0.1. [3], discussion around 28:20 |
| F24 | ICA/change-of-variables likelihood | Low / research later | Earlier ICA experiments were not convincing. Revisit as an experiment rather than a required package feature or acceptance test. [3], before 29:15; [4], F44 |
| F25 | Finite-discrete fitting | High / v0.1 | Consolidate the available discrete work and test general finite-discrete cases, including non-binary variables. Keep binary-IV application assumptions out of generic code. [3], 29:15–30:44 |
| F26 | State-space and memory guards | High / v0.1, small change | Add a simple check before an infeasible enumeration/allocation. Do not turn this into a resource-management framework. [3], F26, around 31:23–31:24 |
| F27 | Gumbel-Softmax | Medium / v0.2 **proposed review slot** | Explicitly defer; exact finite-state work comes first. The meeting did not specify the release for a Gumbel route or approve its implementation. [3], 31:58–33:07 |
| F28 | Keep discrete-as-continuous out of the ordinary interface | High / v0.1; original exclusion superseded | Retain both the finite-discrete route and deliberate numeric treatment through base `SEM2`, using an explicit categorical-to-numeric conversion helper. The latter is a modeling choice, not equivalent to finite-discrete fitting. The [Slack review clarification](#review-clarifications) supersedes the original exclusion and resolves the transcript reversal. Meeting background: [3], 32:46–33:07 and following correction |
| F29 | Mixed discrete/continuous semantics | Medium / v0.2 **proposed design study** | Defer until the pure continuous and finite-discrete paths are reliable. No genuine mixed fitter was agreed for v0.1; a v0.2 study is not a promise to deliver one. [3], 33:07–35:07 |
| F30 | Linear path analysis | Medium / v0.2 | Add after the linear-Gaussian tests are established. Path-specific effects are useful but not a first-version requirement. [3], 35:10–35:57 |
| F31 | Nonlinear fitting efficiency | Medium / v0.2 | Defer performance work until correctness. The fitting recording discusses this just before the break, despite its filename ending at F27. [3], 36:00–36:17 |

### 1.3 Causal queries and identification

| ID | Original item, shortened | Priority / milestone | Meeting decision and limit |
|---|---|---|---|
| F32 | Intervention tests | High / v0.1 | Test interventions directly and support more than one treatment name. Add the proposed sequential randomized-trial example with two treatments and intermediate information potentially confounded with the outcome. [4], F32 |
| F33 | Inspectable assumptions and estimands | High / v0.1 ID check; Medium / v0.2 numerical experiment | Implement DAG-to-ADMG projection and ID for one observational margin. State the limits: no data fusion, latent-variable targets, or identification from additional proxy/parametric assumptions. Keep result representation small and subject to review. Defer the experiment checking whether numerical lower/upper estimates meet on identifiable examples. [4], F33 |
| F34 | Partial-identification workflow | High / v0.1 existing core; Medium / v0.2 robustness extensions | Keep basic partial identification in the first coherent version, reusing the invested work. Defer continuation, extensive multistart/penalty sweeps, and elaborate trainer diagnostics. Do not silently remove the existing workflow because the larger proposal was deferred. [4], F34 |
| F35 | Bounds against known references | High / v0.1 | Turn the discrete-IV example with independently computed linear-programming bounds into a regression test. Agree the reference problem, assumptions, and acceptable numerical error. [4], F35 |
| F36 | General uncertainty-result interfaces | Low / deferred | The team did not want the broad uncertainty-interface project now. This does not license describing an identified set or numerical range as a confidence interval. [4], F36 |
| F37 | Linear-model uncertainty | Medium / v0.2 **proposed slot** | Defer until point fitting is stable. Standard parametric formulas were discussed, but no method or release was selected. Review before committing v0.2 to an implementation. [4], between F36 and F38 |
| F38 | Nonlinear/causal-query uncertainty | Low / theory-led later work | Put this after reliable estimation and the required theoretical work. Do not add generic bootstrap/influence-function machinery merely because it appeared in the initial plan. [4], F38 |
| F39 | Distributional summaries; discussion redirected to graph plotting | High / v0.1 graph preview, conditional; Low / original summaries | The discussion did not adopt the variance/quantile/tail-summary feature. Instead, ask the agent for a simple plot of the supplied graph and include it if the team likes it. This is a static inspection aid, not an interactive graph editor. [4], F39 |

### 1.4 Tests, examples, documentation, and packaging

| ID | Original item, shortened | Priority / milestone | Meeting decision and limit |
|---|---|---|---|
| F40 | Simulations with known targets | High / v0.1 | Build a small collection around the agreed linear/measurement, confounder-adjustment, IV, and sequential-treatment designs. Start with one or two cases per design, not a large simulation grid. The proximal example is explicitly deferred under F43. [4], F40, F43 |
| F41 | One executable example per supported model type | Low / no separate task | The team regarded this as jargon and duplication of tests already discussed. Cover the agreed examples under F15/F25/F40/F45; do not create another workstream. [4], F41 |
| F42 | Reduced RHC test | Low / later demonstration | RHC has no known ground truth here. Keep it as a possible future demonstration, not a v0.1 correctness benchmark. [4], F42 |
| F43 | Proximal/proxy example | Medium / v0.2 | Explicitly defer to the next version after the package is stable and organized. Earlier mention among possible simulations does not bring it back into v0.1. [4], F43 |
| F44 | External-validation benchmarks | High / v0.1 OLS/lavaan; Medium / v0.2 RDD; Low / ICA | Include the usual linear and measurement-model comparisons now. RDD is for the near future, not this update; ICA remains an experiment. The transcript's final conclusion, not its intermediate "we can do this now", sets RDD's timing. [4], F44 |
| F45 | Notebook consolidation | High / v0.1 | Keep three explanatory notebooks under one documentation demo folder: linear SEM versus lavaan, simulated confounder adjustment, and discrete-IV partial identification. Move regression checks into independent Python tests. Inventory and review stale material before deleting it. [4], F45 |
| F46 | User and function/class documentation | High / v0.1 | Ask the agent for two documentation approaches and one class/module example first. Humans choose the format and hosting/build approach; then expand the docstrings/reference and tutorials. No platform was finally selected. [4], F46 |
| F47 | Reproducible datasets and examples | High / v0.1, proportionate | Accept reproducibility work for the retained examples, but avoid excessive checksums and infrastructure. Record enough to rerun and interpret those examples. [4], F47 |
| F48 | Runtime, test, and example dependencies | High / v0.1 | Separate dependencies; `pytest` should not be required just to use the package. Reconcile `pyproject.toml` and `requirements.txt` and remove the outdated `.gitmodules` after checking its use. [4], F45 cleanup discussion and F48 |
| F49 | Open-source metadata and release | Low / later release task | Public release is intended, but no date or license was chosen. Humans can investigate licensing now without making release preparation a v0.1 blocker. Do not select a license on the team's behalf. [4], F49 |
| F50 | Continuous integration (CI) | Low / skipped, not approved | The team skipped this as probably overlapping the tests. Running the v0.1 tests is required; installing a new GitHub automation pipeline or notebook/GPU CI is not an agreed requirement. [4], F50 |
| F51 | Benchmark and regression budgets | Low / later | Defer formal performance budgets while the package is changing. Individual correctness tests still need sensible, reviewed tolerances. [4], F51 |

### 1.5 Longer-term topics

| ID | Original item, shortened | Priority / milestone | Meeting decision and limit |
|---|---|---|---|
| F52 | PFN posterior study | Low / research backlog | Keep the idea documented; do not work on it now. [4], final four items |
| F53 | Graph-as-input amortization | Low / research backlog | Defer along with PFNs; no v0.2 delivery commitment. [4], final four items |
| F54 | Keep graph discovery outside the product | Low / excluded | Continue with a scientifically supplied graph. Graph discovery is not a v0.1 or v0.2 feature in this plan. [4], final four items |
| F55 | More real policy/domain datasets | Low / later | Retain as a future direction, not an expansion of the immediate examples. [4], final four items |

## Sources

[1] https://github.com/causal-design/sem2/blob/f3668ce6654d2b4e4dca60aff9954d004288aea2/docs/product-and-implementation-plan.md — Initial product and implementation plan

[2] https://github.com/causal-design/sem2/blob/f3668ce6654d2b4e4dca60aff9954d004288aea2/docs/meetings/2026-09-07/transcript-f01-f09.txt — Foundations recording (F01 onwards)

[3] https://github.com/causal-design/sem2/blob/f3668ce6654d2b4e4dca60aff9954d004288aea2/docs/meetings/2026-09-03/transcript-f12-f27.txt — Fitting recording (F11–F31)

[4] https://github.com/causal-design/sem2/blob/f3668ce6654d2b4e4dca60aff9954d004288aea2/docs/meetings/2026-09-03/transcript-f32-f55.txt — Queries and planning recording (F32–F55)
