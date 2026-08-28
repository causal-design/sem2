# SEM2 project status

**Last updated:** 28 August 2026<br>
**Evidence baseline:** `main` at `cafcae6e7002e8b9809ef0c55bc061a4bc1d3117`

This file records the current capabilities, limitations, decisions, and order of work. [`README.md`](README.md) is the stable user entry point, while [`AGENTS.md`](AGENTS.md) contains contributor guidance. Fresh source and test evidence takes precedence over this record.

The supporting evidence is in the [updated technical review](docs/meetings/2026-08-26-ai-review-updated.md) and [meeting summary](docs/meetings/2026-08-26-meeting.md).

## Current status

### Working core

The strongest parts of the package are:

- formula parsing and DAG construction;
- continuous and linear structural sampling;
- exact mean and covariance calculations for scalar linear SEMs;
- Gaussian likelihood fitting for linear Gaussian models; and
- finite-state probability calculations for small discrete models.

Interventions can be evaluated by replacing selected node values during sampling. This computes the intervention implied by the supplied structural model; it does not validate the graph, prove exogeneity, or establish identification from observational data.

### Experimental or incomplete

| Area | Current position |
|---|---|
| Generic continuous fitting | The energy-loss and package-trainer path needs repair and direct end-to-end tests. |
| Discrete fitting | Finite-state probability calculation works, but the fitting workflow is specialized and incomplete. |
| Mixed models | Mixed discrete/continuous graphs can be classified, but fitting is not supported end to end. |
| Partial identification | The operator abstraction can optimize a causal target alongside fit, but numerical extrema are not automatically sharp identified bounds. |
| Uncertainty quantification | No package-level confidence interval or posterior workflow is implemented. |
| MMD | Not implemented. |
| PFN inference | Not implemented. |
| Diagnostics | Optimizer and model-fit diagnostics are not yet systematic. |

### Work outside the shared package

At the meeting, contributors reported two relevant implementations outside the reviewed tree:

- a local Sinkhorn-divergence implementation with encouraging preliminary heavy-tail results; and
- additional discrete-model code developed for the associated paper.

These are candidates for bounded review and integration, not current SEM2 capabilities.

## Agreed design direction

### Separate the model, data view, and query

The structural model should describe the graph, exogenous distributions, and endogenous mechanisms. Whether a variable is observed is a property of a dataset relative to that model, not an intrinsic property of the model.

A future data layer should therefore provide:

- a named mapping from table columns to model nodes;
- the nodes observed in each dataset;
- the observational or interventional regime;
- interventions and fixed values, if any; and
- explicit dataset weights when several losses are combined.

Treatment, outcome, estimand, and intervention roles belong to a query or operator. Generic operators must not silently assume binary-IV node names or positional table columns.

The existing `unobs(...)` syntax remains part of the current API. Any migration should preserve existing formulas or provide an explicit compatibility path.

### Keep the graph supplied at first

The immediate goal is reliable inference for a graph supplied by the user. PFN work may eventually amortize across graphs provided as inputs. Learning the graph as an output is a separate and substantially less identified problem, and is not the current priority.

### Keep causal claims conditional

SEM2 can answer questions under a supplied structural model. Reports and examples must separate:

1. fit to observed distributions;
2. nonparametric or model-based identification;
3. regularity of the target under the chosen discrepancy;
4. finite-sample statistical uncertainty; and
5. numerical optimization error.

A good fit on observed variables is not evidence that all five issues have been resolved.

## Work plan

### Priority 1: make the general fitting path reliable

- define the graph-generic dataset/schema contract;
- repair the continuous energy-loss sample-axis logic;
- remove hard-coded `Z`, `X`, and `Y` assumptions from generic components;
- make device and dtype handling explicit; and
- settle the relationship between the package trainer and experiment trainers.

### Priority 2: add direct behavioral tests

Test losses, trainers, interventions, and partial-identification operators directly. Cover analytical values, gradients, batch and feature axes, devices, seeds, convergence failures, and named variable mappings. Retained examples should run in CI.

### Priority 3: review external implementations

Bring the Sinkhorn and remaining discrete code into review branches with their exact objectives, dependencies, supported shapes and devices, fixed-seed examples, reference tests, and benchmark acceptance criteria. Also compare projected or sliced energy methods where dimension makes the ordinary energy score weak.

### Priority 4: add diagnostics and uncertainty

Expose optimizer convergence, loss decomposition, parameter changes, and model-versus-data checks. Develop uncertainty theory and implementations without conflating identification, differentiability, finite-sample variation, and numerical error.

### Priority 5: run a bounded PFN study

Start with a supplied graph and learn a posterior for a specific causal query, or for a modest parameter vector in a linear SCM. Compare against an explicit baseline such as direct optimization, a Laplace approximation, MCMC, or simulation-based inference. Evaluate calibration and prior-shift sensitivity before broadening the graph or mechanism family.

## Meeting follow-up

The following ownership was discussed and should be confirmed against contributors' availability before scheduling work:

| Owner | Follow-up |
|---|---|
| Qingyuan Zhao | Develop the uncertainty argument and coordinate the next project phase. |
| Amruth | Prepare the Sinkhorn work and related RDD study for review. |
| Max | Identify and package the remaining discrete research code. |
| Team | Review projected/sliced energy methods and choose near-term software priorities. |

## Open questions

- What is the smallest useful dataset/schema API for several observed margins and regimes?
- How should losses from datasets of different sizes and scales be weighted?
- When should a missing observed node be an error rather than an intentional marginalization?
- Which diagnostics are required before reporting a numerical partial-identification bound?
- Which priors over nonlinear mechanisms are meaningful enough for Bayesian or PFN work?
- At what level does amortization provide enough reuse to justify PFN training?

## Evidence and uncertainty

Directly observed in the reviewed repository:

- the parser, DAG builder, samplers, linear moment code, linear likelihood, finite-state probability engine, and current tests;
- defects and design-specific assumptions documented in the technical review; and
- the absence of package implementations for MMD and PFN inference.

Contributor-reported, but not observed in the reviewed tree:

- the local Sinkhorn implementation and preliminary heavy-tail behavior; and
- additional discrete paper code awaiting integration.

Unverified:

- robustness of the general trainer across seeds and model families;
- sharpness of numerical partial-identification bounds;
- comparative benefits of Sinkhorn, projected energy, and MMD for target recovery; and
- calibration and prior-shift robustness of any future PFN approach.

The full source-backed assessment, equations, examples, and references remain in the [dated technical review](docs/meetings/2026-08-26-ai-review-updated.md).