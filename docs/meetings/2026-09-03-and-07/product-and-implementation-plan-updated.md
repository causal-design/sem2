# SEM2 product and implementation plan: meeting update

**Status:** meeting-based draft for human review; no package implementation is included in this change.

The aim of **v0.1** is a coherent, usable research package with a small set of convincing tests. Preserve the formula-to-model core, consolidate the fitting code, support named causal queries and basic partial identification, and make the examples and documentation easier to use. The meetings repeatedly preferred this to a large redesign or a trainer with many options.[2][3][4]

Here, **high priority / v0.1** means the next package milestone; **medium priority / v0.2** means the following development round; **low priority** means no scheduled implementation. These version labels follow the request for this update, rather than the original draft's P0–P3 scale. They do not announce a public release or change the package version. A v0.2 research task can deliver a decision or experiment rather than a supported feature.

The tables distinguish an accepted item, a reduced scope, a deferral, and an unresolved question. Where a meeting said only "later", the proposed version assignment is marked explicitly. A skipped item is not silently approved. The original plan's detailed architecture and implementation stages were not adopted wholesale.[4]

## Reading the records

- [Initial product and implementation plan](product-and-implementation-plan.md): the original 55-item proposal.[1]
- [Foundations transcript](transcript-f01-f09.txt) and [supplied minutes](summary-f01-f09.md): F01 onwards, including the short graph-construction discussion at the end.[2]
- [Fitting transcript](transcript-f12-f27.txt) and [supplied minutes](summary-f12-f27.md): starts by saying F11 was skipped, then covers F12 through F31, despite the filename.[3]
- [Queries and planning transcript](transcript-f32-f55.txt) and [supplied minutes](summary-f32-f55.md): F32–F55, the revised user workflow, and next steps.[4]

The filenames and supplied minutes are preserved. This folder groups the records for 3 and 7 September without inferring a recording's date from its upload order. The transcripts, rather than the AI-generated minutes, determine the decisions below. Citations identify the recording; F-numbers, timestamps where available, and quoted phrases locate the relevant passage. The Sources section also links to the inspected, pre-move versions.

### Points to confirm in this review

1. **F02/F03 timing.** The detailed discussion calls named data mapping and separation of observation status high priority; the closing architecture recap calls the dataset view medium priority. The working proposal below keeps a minimal mapping/observed-node boundary in v0.1 and puts the richer schema in v0.2. Please confirm this split rather than treating it as an unambiguous meeting decision.[2][4]
2. **F12 support semantics.** The team called this a high-priority design question but deferred the technical discussion. v0.1 needs that decision, not an agent-invented universal support property.[3]
3. **F10 and F28 interpretation.** The brief graph-editor discussion does not settle the original programmatic-constructor proposal. The discrete-as-continuous discussion also contains a reversal. Both remain outside the proposed ordinary v0.1 interface, with the exact disposition to confirm.[2][3]
4. **Scope still needing design approval.** Review the minimal query/result representation, the graph-plot preview, and a sample documentation page before agents expand them. Confirm the v0.2 assignments labelled "proposed" below.[3][4]

## 1. Decisions on the 55 items

### 1.1 Model, data, and query foundations

| ID | Original item, shortened | Priority / milestone | Meeting decision and limit |
|---|---|---|---|
| F01 | Capability matrix and stability labels | Low / dropped for now | Drop the formal classification exercise at this early stage. Ask agents for focused changes instead; do not add a registry merely to label everything experimental. [2], 19:09–22:19 |
| F02 | Named dataset view/schema | High / v0.1 minimal; Medium / v0.2 broader schema — **confirm** | The detailed discussion accepts named column-to-node mapping but considers the proposed schema too large. The closing recap calls dataset views medium priority. Keep the minimum needed for the examples provisionally in v0.1; confirm the split. [2], 22:19–27:27; [4], "Put data concerns at the boundary" |
| F03 | Model separate from observation status | High / v0.1 — **confirm with F02** | Agree that observed/latent status belongs to a dataset relative to the graph, not permanently to the structural model. Decide the small implementation alongside F02; do not import the original compatibility requirement. [2], 24:23–27:27; [4], closing architecture recap |
| F04 | Query/estimand separate from the graph | High / v0.1 | Accept. Users should choose arbitrary treatment/outcome names and query a fitted model without editing global training configuration. Query-specific fitting must still be possible for partial identification. No large query-object hierarchy was agreed. [2], 27:27–30:28; [4], product-workflow discussion |
| F05 | Observational/interventional data regimes | Low / later | Defer fitting from combined observational and experimental data. This does not defer computing interventions from a supplied model. [2], after 30:28 |
| F06 | Multiple datasets and observed margins | Low / later | Defer joint fitting across datasets or margins. One dataset/observational margin is enough for the next milestone. [2], F06 |
| F07 | Empirical joint covariate distribution | Medium / v0.2 | Useful, but not required to test the core now. Preserve covariate dependence; leave open whether empirical sampling belongs in model fitting, a sampler, or the data loader. [2], 32:26–44:06 |
| F08 | Multidimensional node values | Medium / v0.2 | Prioritize vector-valued nodes ahead of the empirical-distribution extension, but not as a v0.1 prerequisite. Do not assume a scalar node per covariate means those covariates are independent. [2], 43:11–F08 discussion |
| F09 | Preprocessing and categorical encoding | Medium / v0.2 | Defer to a categorical-variable discussion. A maintained default encoding should help users without preventing deliberate alternatives; no full preprocessing framework was approved. [2], 46:52–50:46 |
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
| F28 | Keep discrete-as-continuous out of the ordinary interface | Low / excluded shortcut — **confirm wording** | The discussion first suggests retaining both routes, then says to remove treating discrete variables as continuous. Do not advertise that shortcut as proper discrete support. Clarify the intended disposition before removing existing research code. [3], 32:46–33:07 and following correction |
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

## 2. Planned shape of v0.1

### 2.1 What a user should be able to do

The intended users are researchers and method developers with a supplied graph who want to prototype or compare structural models. Linear-Gaussian analysis is included in its own right; the team explicitly removed the phrase limiting the audience to work "beyond" that setting.[4]

Two workflows must coexist:

1. Specify a model, provide data, choose a supported objective, and fit it with the minimal trainer. Inspect the loss/gradient history, then sample, intervene, or evaluate different named queries without fitting again solely to change the query.[2][3][4]
2. For partial identification, specify the query before the query-specific optimization. An initial ordinary fit is optional. Fit/retrain the models needed to investigate lower and upper values; the ordinary fitted model must not be a mandatory starting point.[4]

The team left warm starts unresolved: reusing fitted weights may help, but could also trap optimization. v0.1 should preserve a working partial-identification route without claiming to have settled warm starts or making every query launch a numerical identification experiment.[4]

For F33, the observational distribution comes from one observed margin; the *query* can still involve interventions. Project the supplied DAG onto that margin as an acyclic directed mixed graph (ADMG), then run ID within its stated scope. An unsupported query must not be reported as proven non-identifiable. ID's conclusion concerns the checked graphical problem, not every restricted model or particular contrast. Numerical lower/upper agreement is not a proof of identification. These are interpretation requirements for the proposed implementation, not results established by this documentation change.

### 2.2 Classes and methods to preserve, repair, or add

The names below are checked against the current source on PR #50. They identify where the agreed work can start, **not a claim that all these paths already work**. New concepts are described by responsibility because the meetings did not choose class names or signatures. In particular, the illustrative `Model`, `FitPlan`, `FitResult`, and capability-registry architecture in the initial plan was not approved.[1][4]

| Component | Existing class/method or new responsibility | v0.1 scope |
|---|---|---|
| Formula/model core | `FormulaParser`, `_DAG`, `SEM2`, `SEM2Linear`, `SEM2Discrete` | Preserve the existing formula-based construction and useful internals. Make only the changes required by approved data/query decisions. [2][4] |
| Linear fitting | `SEM2Linear.coef_matrix()`, `mean_vector()`, `cov_matrix()`; `LogLikelihood.evaluate()` | Retain exact linear calculations and test a usable fit through the minimal trainer, including linear/measurement reference comparisons. [3][4] |
| Continuous fitting | `EnergyLoss.evaluate()` and its continuous energy-score calculation | Repair named selection and sample/feature-axis behavior under the reviewed objective. Exercise a small confounder-adjustment simulation. [3] |
| Finite-discrete fitting | `SEM2Discrete.sample()`, finite-state probabilities/metadata; the discrete energy-loss path | Make small binary and non-binary examples usable without IV-specific assumptions in generic code. Guard state growth. Enumeration of finite states must not conceal any remaining Monte Carlo integration over exogenous noise. [3] |
| Trainer | `SEM2Trainer.train()` | One minimal implementation for the agreed examples, with loss/gradient history that can be inspected and plotted. Reconcile duplicates rather than adding a second trainer. [3] |
| Data boundary | New lightweight mapping/observed-node responsibility; **exact class/method undecided** | Map table columns to model nodes for one dataset and identify its observed margin. This is the provisional F02/F03 split, requiring confirmation. [2][4] |
| Random-variable support | `RandomVariable` and node/model support information; **design undecided** | Resolve F12 before extending these interfaces. Do not assume arbitrary endogenous support can be computed locally. [3] |
| Named interventions and queries | `SEM2.sample(..., set_nodes=...)`, `PotentialOutcome.evaluate()`, `MeanPotentialOutcome.evaluate()`, `ATE.evaluate()` | Remove global treatment/outcome coupling. Support multiple named treatments in the query layer; the meeting suggested a list of strings. Preserve intervention sampling as a model operation. [2][4] |
| Identification and query explanation | New DAG-to-ADMG projection and ID routines; **return format undecided** | Check the supported single-margin problem and explain the identification scope. Agree a small result representation rather than adopting every proposed metadata field. [4] |
| Partial identification | Existing query operators and `SEM2Loss` composition, plus consolidation of the existing research workflow | Let the target enter the fitting objective and produce a numerical lower/upper investigation for the discrete-IV example. Validate against the independent reference; do not claim a generic sharp-bound solver. [4] |
| Graph inspection | New plotting helper or model method; **name and placement undecided** | Show a small static preview with named nodes, edges, and exogenous/endogenous distinctions. Include it after human review; do not build an editor. [4] |

Source entry points: [construction](../../../sem2/model/construction/), [models](../../../sem2/model/models/), [nodes](../../../sem2/model/components/nodes.py), [linear operators](../../../sem2/operators/linear.py), [scalar operators](../../../sem2/operators/scalar_operators.py), [vector operators](../../../sem2/operators/vector_operators.py), [discrete energy loss](../../../sem2/operators/discrete_energy_loss.py), and [trainer](../../../sem2/sem2trainer.py).

### 2.3 Agreed examples and tests

These are test topics, not approved filenames or a requirement for a new testing framework. The agent should propose small cases and humans should approve their scientific targets and tolerances. Existing useful tests remain; the list does not authorize their replacement.[2][3][4]

| Test/example | What to check | Meeting basis |
|---|---|---|
| Linear-Gaussian SEM, including measurement error | Known coefficients/moments or fit targets; comparison with OLS and lavaan in appropriate examples | F15, F40, F44 [3][4] |
| Simple confounder-adjustment simulation | Fit and estimate the agreed causal contrast against known simulation truth; exercise the continuous energy-loss route | F25 discussion, F40, F45 [3][4] |
| Small finite-discrete model | Fitting and finite-state probabilities for binary and at least one non-binary case; infeasible state-space rejection | F25–F26 [3] |
| Discrete-IV partial identification | Compare the numerical lower/upper results with independently computed linear-programming bounds under the same assumptions | F34–F35 [4] |
| Sequential randomized treatments | Two treatment variables with intermediate information potentially confounded with the outcome; named multi-treatment queries and a known target | F32 [4] |
| Direct intervention checks | Equation replacement, renamed treatment/outcome nodes, shapes, unaffected mechanisms, and common-random-number comparisons where appropriate | F32 [4] |
| ID checks on projected graphs | Reference cases inside the supported observational-margin setting and clear handling of unsupported inputs | F33; the exact case list still needs review [4] |

Use a small CPU-run collection rather than many seeds, sample sizes, and tuning combinations. The agent can add focused checks of the repaired losses, mapping, and trainer inside these tasks; that is not approval for an expanding simulation grid.[3][4]

Keep exactly the three agreed explanatory notebooks: **linear SEM versus lavaan; simulated confounder adjustment; discrete-IV partial identification**. Put them together under `docs/` (for example, `docs/demos/`, with the final name to agree). Keep independent Python regression tests under `tests/`; they may share example-generating code, but must not depend on executing documentation notebooks.[4]

Proximal/proxy and RDD examples are for v0.2. RHC and ICA are not v0.1 acceptance tests. Neither uncertainty quantification nor GPU/HPC testing is required to demonstrate the agreed v0.1 core.[3][4]

## 3. Actions to deliver v0.1

The following is a proposed order for carrying out the decisions, not a claim that the meeting allocated owners or approved exact implementations. Humans choose scope, scientific meaning, and acceptance criteria; an agent can inspect, propose, implement approved changes, and return executed evidence in a focused PR. Review and merge remain human decisions.[2][3][4]

| Order / task | Human decision or review required | Work an AI coding agent can do | Result to review |
|---|---|---|---|
| 1. Confirm this scope | Resolve F02/F03, F12, F10/F28, and provisional version assignments; name a reviewer for each task | Prepare small concrete examples and explain ambiguous items in plain English | Reviewed v0.1 list with unresolved points answered |
| 2. Settle the data/support and query interfaces | Choose the minimal data mapping, observed-node handling, support semantics, and how queries enter ordinary versus partial-ID fitting | Inspect current classes, show minimal calling examples, and propose local changes before coding | Short interface proposal, then a focused implementation PR |
| 3. Establish fitting/reference cases | Approve the linear/measurement, confounder, and discrete examples; choose reference values and tolerances | Inventory existing tests/research examples and demonstrate present behavior, including failures | Small test list and reproducible baseline evidence |
| 4. Consolidate fitting | Review the energy-score definition and the one-trainer design; agree the basic diagnostic output | Repair hardcoded selection/axes, reconcile trainer copies, expose loss/gradient history, run approved CPU cases | Linear and continuous fits through one minimal trainer |
| 5. Complete finite-discrete and basic partial-ID use | Select the non-binary case and discrete-IV LP reference; confirm which existing research code is authoritative | Integrate approved discrete work, add the simple enumeration guard, make the existing partial-ID example runnable, compare results with the reference | Numerical evidence for discrete fitting and the partial-ID example |
| 6. Add named multi-treatment queries and ID | Approve the sequential-treatment graph/query; review projection and ID scope; choose the small result format | Extend query handling, implement projection/ID, add reviewed reference and intervention checks | Query/ID PR with assumptions, limitations, and executed tests |
| 7. Choose and build documentation | Compare two approaches and approve one class/module page before broad generation; decide hosting separately | Show a small preview, then write consistent docstrings, explanations, and build configuration for the selected approach | Usable documentation for the agreed paths, built and inspected before any deployment |
| 8. Clean up examples and packaging | Review what to retain/delete; confirm dependency organization and any new dependency | Inventory notebooks/experiments, consolidate the three demos, extract tests, reconcile dependency files, check/remove obsolete `.gitmodules`, rerun retained examples | Clean layout and reproducible installation/demo instructions |
| 9. Review the graph preview | Approve a simple readable plot and its interface/dependency cost | Produce a small preview; implement it only after approval | Optional small plotting PR within the agreed v0.1 scope |
| 10. Review v0.1 together | Decide whether results and usability are sufficient; accept or defer any remaining conditional item | Run the approved CPU tests and demos, build the docs, and summarize actual results and unresolved failures | Human-reviewed milestone, not an automatic public release |

Steps are dependent only where the behavior requires it. Documentation previews and the notebook inventory can start while interface decisions are being discussed; agents should wait for those decisions before rewriting affected code. No work here requires a new CI service, an HPC connection, a comprehensive trainer configuration system, or an automatic uncertainty interface.[3][4]

The completion check is practical: can a team member install the package, run the small agreed examples through the maintained interfaces, inspect fitting diagnostics, and understand the results and limitations from the documentation? Numerical acceptance thresholds still need human approval. The planning PR itself does not establish that these checks pass.

After the team approves the plan, reconcile the maintained [PROJECT.md](../../../PROJECT.md) and [AGENTS.md](../../../AGENTS.md) with it in a separately authorized change. In particular, their existing compatibility guidance predates F14. This draft does not rewrite repository memory or change runtime behavior.

## 4. Slack follow-on topics

Start separate threads for decisions that need human input, with a short proposal and usually a focused PR as the outcome. Do not require the whole team in every thread or start implementation of every topic at once. The recorded next meeting is 17 September at 2 p.m.; its timezone is not stated. The goal before then is to resolve blockers and begin reviewable work, not promise all of v0.1 by that date.[4]

| Suggested thread | First question / agent preparation | Human input | Intended output before the next meeting |
|---|---|---|---|
| **Minimal model/data boundary and support** (F02/F03/F12/F13) | Show one small model fitted to a named table. Compare minimal mapping with the broader dataset schema; explain support and tensor-axis questions | Confirm the v0.1/v0.2 split and support semantics | Short approved interface decision; flag anything still needing a technical meeting |
| **One trainer and the first fitting tests** (F15–F18/F40/F44) | Show what differs between current trainer copies and what is wrong in the energy loss; propose the linear and confounder cases | Select objective conventions, diagnostic output, references, and tolerances | Minimal implementation plan and the first focused repair/test PR |
| **Discrete fitting and partial identification** (F25/F26/F28/F34/F35) | Show the existing binary-IV workflow, one non-binary case, and the LP reference; separate current functionality from proposed robustness features | Contributors familiar with the discrete work review scope, retained code, and numerical interpretation | Approved consolidation plan and test case; warm starts remain an explicit open question |
| **Named queries, multiple treatments, and ID** (F04/F32/F33) | Show an ordinary query and a partial-ID query; sketch the sequential trial and DAG-to-ADMG-to-ID steps | Agree interfaces and scientific examples; invite an ID specialist if useful (Sina was mentioned) | Query/ID design and reference-test list, followed by focused PRs |
| **Documentation: choose an approach from a preview** (F46) | Present two build/hosting options with one real class/module page, rather than generating everything | Choose format, navigation, and hosting plan; review whether a new user can understand the example | Reviewed documentation approach and first small docs PR |
| **Three demos and a simpler repository** (F45/F47/F48) | Inventory notebooks/experiments and dependencies; propose keep/move/test/delete decisions | Review proposed deletions and installation choices | Approved cleanup map and packaging/demo PR |
| **Simple graph plot** (F39; small optional thread) | Show the graph for an existing example with no editor or interactive constructor | Decide whether the preview is useful enough to include now | Accept a small plotting PR or explicitly defer it |

Reviewers for these threads are suggestions to be agreed, not assignments inferred from speaker labels. The discrete work will benefit from Max's input; documentation should include someone reading as a new user. Licensing/release experience and the v0.2 research queue can have later threads, but need not block the first implementation tasks.[3][4]

Each thread should end with a brief recorded decision, a clearly identified reviewer, and either a PR or an explicit deferral. Creating these threads, scheduling meetings, selecting a license, merging PRs, and releasing the package are not actions performed by this planning update.

## Sources

[1] https://github.com/causal-design/sem2/blob/f3668ce6654d2b4e4dca60aff9954d004288aea2/docs/product-and-implementation-plan.md — Initial product and implementation plan

[2] https://github.com/causal-design/sem2/blob/f3668ce6654d2b4e4dca60aff9954d004288aea2/docs/meetings/2026-09-07/transcript-f01-f09.txt — Foundations recording (F01 onwards)

[3] https://github.com/causal-design/sem2/blob/f3668ce6654d2b4e4dca60aff9954d004288aea2/docs/meetings/2026-09-03/transcript-f12-f27.txt — Fitting recording (F11–F31)

[4] https://github.com/causal-design/sem2/blob/f3668ce6654d2b4e4dca60aff9954d004288aea2/docs/meetings/2026-09-03/transcript-f32-f55.txt — Queries and planning recording (F32–F55)
