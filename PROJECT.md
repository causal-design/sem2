# SEM2 project status

**Last updated:** 29 August 2026<br>
**Repository baseline:** `main` at `315df9cfe5d6f1298f9d73ec574e08c87fa59426`<br>
**Detailed source-review baseline:** `cafcae6e7002e8b9809ef0c55bc061a4bc1d3117`

This file records the current capabilities, limitations, decisions, and order of work. [`README.md`](README.md) is the stable user entry point, while [`AGENTS.md`](AGENTS.md) contains contributor guidance. Fresh source and test evidence takes precedence over this record.

**Release-preparation note (16 October 2026):** the capability assessment below is the August baseline, not a fresh review of the October code. Source and tests have advanced; the status refresh is tracked in PR #62. This file remains public. Supporting technical reviews and meeting records now live in the private [SEM2 documentation repository](https://github.com/causal-design/sem2-docs); public contributions must include enough requirements and evidence to be reviewed without private access.

## Current status

### Working core

The strongest parts of the package are:

- formula parsing and DAG construction;
- continuous and linear structural sampling;
- exact mean and covariance calculations for scalar linear SEMs;
- Gaussian likelihood fitting for linear Gaussian models; and
- finite-state probability calculations for small discrete models.

Interventions can be evaluated by replacing selected node values during sampling. This computes the intervention implied by the supplied structural model; it does not validate the graph, prove exogeneity, or establish identification from observational data.

### Experimental, incomplete, or absent

| Area | Current position |
|---|---|
| Generic continuous fitting | The energy-loss and package-trainer path needs repair and direct end-to-end tests. |
| Real-table workflow | Named column mapping, categorical encoding, missing-value policy, and reusable preprocessing are not provided as a coherent package workflow. |
| Multidimensional variables | The present graph and node interfaces are oriented toward scalar variables; a multidimensional covariate node with a supplied empirical distribution is not supported. |
| Discrete fitting | Finite-state probability calculation works, but the fitting workflow is specialized and incomplete. |
| Mixed models | Mixed discrete/continuous graphs can be classified, but fitting is not supported end to end. |
| Partial identification | The operator abstraction can optimize a causal target alongside fit, but numerical extrema are not automatically sharp identified bounds. |
| Uncertainty quantification | No package-level confidence interval or posterior workflow is implemented. |
| MMD | Not implemented. |
| PFN inference | Not implemented. |
| Diagnostics | Optimizer and model-fit diagnostics are not yet systematic. |

### Work outside the shared package

Contributors have reported two relevant implementations outside the reviewed tree:

- a local Sinkhorn-divergence implementation with encouraging preliminary heavy-tail results; and
- additional discrete-model code developed for the associated paper.

These are candidates for bounded review and integration, not current SEM2 capabilities.

## Agreed design direction

### Separate the model, data view, and query

The structural model should describe the graph, exogenous distributions, and endogenous mechanisms. Whether a variable is observed is a property of a dataset relative to that model, not an intrinsic property of the model.

A future data layer should therefore provide:

- a named mapping from table columns to model nodes;
- the nodes observed in each dataset;
- variable types, shapes, support, and missing-value policy;
- the observational or interventional regime;
- interventions and fixed values, if any; and
- explicit dataset weights when several losses are combined.

Treatment, outcome, estimand, and intervention roles belong to a query or operator. Generic operators must not silently assume binary-IV node names or positional table columns. The existing `unobs(...)` syntax remains part of the current API; any migration should preserve existing formulas or provide an explicit compatibility path.

### Preserve joint covariate distributions without unnecessary modelling

The RHC example showed that representing many dependent covariates as unrelated scalar root nodes changes their joint distribution when the fitted model resamples them. That can change the population over which a causal contrast is averaged. The package should not silently turn this convenience representation into an independence assumption.

For an ordinary confounder-adjustment problem, the software should be able to preserve the observed joint covariate distribution without requiring the user to estimate an unnecessary high-dimensional generator. A multidimensional exogenous object with a supplied empirical distribution is one candidate. Whether this belongs in the model, dataset layer, or sampling interface remains an architectural question.

### Keep the graph supplied at first

The immediate goal is reliable inference for a graph supplied by the user. PFN work may eventually amortize across graphs provided as inputs. Learning the graph as an output is a separate and substantially less identified problem, and is not the current priority.

### Validate point estimation before general uncertainty machinery

The near-term target is a polished point-estimation workflow for simple, well-specified designs. Controlled simulations with known answers should test whether the fitting method recovers the intended target. A reduced one- or two-covariate RHC example may test that the pipeline runs, but must not be presented as a substantive analysis of the full high-dimensional problem. A low-dimensional proximal-proxy graph is another candidate test case.

Confidence intervals remain important, but a general interface should follow evidence that point estimation works across the intended model classes. Bootstrap-like, jackknife-like, and influence-function routes require separate statistical assumptions and validation.

### Keep causal claims conditional

SEM2 can answer questions under a supplied structural model. Reports and examples must separate:

1. fit to observed distributions;
2. nonparametric or model-based identification;
3. regularity of the target under the chosen discrepancy;
4. finite-sample statistical uncertainty; and
5. numerical optimization error.

A good fit on observed variables is not evidence that all five issues have been resolved.

### Use coding agents within a human-controlled process

The team intends to explore more extensive use of coding agents, subject to contributor agreement and explicit safeguards. Agents may accelerate implementation, but humans retain control of the product definition, architecture, accepted behavior, review, and merge decisions.

The working process is:

1. consolidate desired features and known issues from meetings, GitHub issues, reviews, and repository evidence;
2. have the team remove, revise, and prioritize that inventory;
3. agree the intended product behavior and architectural boundaries before implementation;
4. require an implementation plan and acceptance criteria for each bounded change;
5. use behavioral tests and realistic examples to verify alignment; and
6. review changes through small branches and pull requests, questioning unexpected abstractions, files, dependencies, or behavior.

`README.md`, `PROJECT.md`, and `AGENTS.md` support this process by separating stable human context, maintained project state, and durable agent guidance. Documentation and tests are part of the control system for agent-written code.

## Work plan

### Priority 1: define the intended product and review the feature inventory

Compile the desired capabilities and known problems from the last two meetings, existing GitHub issues, dated reviews, reports, and source evidence. Organize them by product area, propose priorities, identify conflicts or dependencies, and ask the team to confirm what belongs in the near-term package before committing to an implementation architecture.

### Priority 2: make the general point-estimation path reliable

- define the graph-generic dataset and schema contract;
- decide how supplied multidimensional and empirical distributions enter the model-data boundary;
- repair the continuous energy-loss sample-axis logic;
- remove hard-coded `Z`, `X`, and `Y` assumptions from generic components;
- make device and dtype handling explicit; and
- settle the relationship between the package trainer and experiment trainers.

### Priority 3: add direct behavioral tests and evaluation examples

Test losses, trainers, interventions, and partial-identification operators directly. Cover analytical values, gradients, batch and feature axes, devices, seeds, convergence failures, and named variable mappings. Retained examples should run in CI.

Use simulations with known targets as the primary statistical evaluation. Use the reduced RHC example only as an end-to-end smoke test and consider a low-dimensional proximal-proxy design as a second method example.

### Priority 4: review external implementations

Bring the Sinkhorn and remaining discrete code into review branches with their exact objectives, dependencies, supported shapes and devices, fixed-seed examples, reference tests, and benchmark acceptance criteria. Also compare projected or sliced energy methods where dimension makes the ordinary energy score weak.

### Priority 5: add diagnostics and uncertainty

Expose optimizer convergence, loss decomposition, parameter changes, and model-versus-data checks. Once point-estimation behavior is credible, develop uncertainty theory and implementations without conflating identification, differentiability, finite-sample variation, and numerical error.

### Priority 6: run a bounded PFN study

Start with a supplied graph and learn a posterior for a specific causal query, or for a modest parameter vector in a linear SCM. Compare against an explicit baseline such as direct optimization, a Laplace approximation, MCMC, or simulation-based inference. Evaluate calibration and prior-shift sensitivity before broadening the graph or mechanism family.

## Meeting follow-up

The following ownership was discussed and should be confirmed against contributors' availability before scheduling work:

| Owner | Follow-up |
|---|---|
| Qingyuan Zhao | Develop the uncertainty argument, ask Max and Amruth for feedback on the AI-assisted workflow, and coordinate the feature-inventory review. |
| Chiara Catalano | Reduce the RHC example to one or two covariates, develop simulations with known targets, and consider the proximal-proxy example with team input. |
| Amruth | Prepare the Sinkhorn work and related RDD study for review. |
| Max | Identify and package the remaining discrete research code. |
| Team | Review the feature inventory and priorities, decide how to represent supplied empirical covariate distributions, and agree architectural boundaries and acceptance criteria. |

## Open questions

- What is the smallest useful dataset and schema API for several observed margins and regimes?
- Which preprocessing responsibilities belong inside the package, and which should remain with the caller?
- Should a supplied empirical covariate distribution be represented as an exogenous node, a dataset-level sampling mechanism, or another reusable component?
- How should losses from datasets of different sizes and scales be weighted?
- When should a missing observed node be an error rather than an intentional marginalization?
- Which simulations and real-data smoke tests are required before the point-estimation workflow is considered reliable?
- Which diagnostics are required before reporting a numerical partial-identification bound?
- Which uncertainty route is appropriate for each supported model class after point estimation has been validated?
- Which priors over nonlinear mechanisms are meaningful enough for Bayesian or PFN work?
- At what level does amortization provide enough reuse to justify PFN training?
- What review and testing rules are necessary to keep agent-written changes understandable and aligned with the agreed architecture?

## Evidence and uncertainty

Directly observed in the reviewed repository:

- the parser, DAG builder, samplers, linear moment code, linear likelihood, finite-state probability engine, and current tests;
- defects and design-specific assumptions documented in the technical review;
- the scalar-oriented node interface and absence of a coherent real-table preprocessing workflow; and
- the absence of package implementations for MMD, a supplied empirical covariate distribution, and PFN inference.

Contributor-reported, but not observed in the reviewed tree:

- the local Sinkhorn implementation and preliminary heavy-tail behavior; and
- additional discrete paper code awaiting integration.

Agreed or proposed in the 28 August meeting, but not yet implemented:

- preserving high-dimensional covariates through a supplied empirical joint distribution;
- the reduced RHC smoke test, controlled simulations, and proximal-proxy example;
- prioritizing reliable point estimation before a general confidence-interval interface; and
- the human-controlled, AI-assisted planning and implementation process.

Unverified:

- robustness of the general trainer across seeds and model families;
- recovery of known causal targets in the proposed simulations;
- sharpness of numerical partial-identification bounds;
- comparative benefits of Sinkhorn, projected energy, and MMD for target recovery;
- the best architectural location for multidimensional empirical covariate sampling; and
- calibration and prior-shift robustness of any future PFN approach.

The full dated assessment and meeting records are preserved in the private documentation repository. They record the discussion behind the directions above; current public code and executed tests remain the evidence for implemented behavior.
