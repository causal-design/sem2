# SEM2 technical review: updated after the 2026-08-26 meeting

**Review date:** 2026-08-26<br>
**Updated:** 2026-08-27<br>
**Author:** Peirce (Nous Research Hermes Agent)<br>
**Repository state checked:** `main` at `cafcae6e7002e8b9809ef0c55bc061a4bc1d3117`<br>
**Scope:** current `sem2/` source, tests and experiments; repository research reports; the earlier AI review; and the supplied meeting transcript and brief summary

## Executive assessment

SEM2 has a useful central idea: let a user write the structural model they believe, compile that model into an executable DAG, and fit its observed implications without requiring every problem to match a standard causal design. The parser, DAG builder, node modules, continuous sampler, exact linear algebra, and finite-state discrete engine already support that idea. The general path from an arbitrary table to a reliable fitted model is less mature.

The meeting sharpened the design in one important way. Exogenous versus endogenous status belongs to the structural model; observed versus latent status belongs to a dataset's view of that model. The current `unobs(...)` syntax places both ideas in the formula. A better design would keep the structural graph stable while dataset schemas tell each loss which nodes were observed, how columns map to them, and whether the rows came from an observational or interventional regime.

The meeting also corrected the status of two research implementations. Sinkhorn divergence exists in a contributor's local work and reportedly performed well in preliminary heavy-tail experiments, but it was not in the shared package at the meeting and is not in the reviewed tree. Some discrete paper code also remained outside the package. These are active implementations awaiting review, not merely paper proposals, but they are not shipped SEM2 capabilities yet.

The causal distinction remains essential. SEM2 can replace equations and simulate model interventions. That operation computes an interventional distribution under the supplied structural model. Observational fit does not validate the graph, prove exogeneity, or show that a causal target is point identified. Partial-identification searches are therefore a natural use of the software, provided numerical extrema are not mistaken for sharp identified bounds without optimization and identification checks.

The revised priorities are:

1. define a graph-generic data contract and repair the general fitting path;
2. test the loss, trainer, intervention, and partial-identification layers directly;
3. review and integrate the external Sinkhorn and discrete implementations;
4. add optimizer and model-fit diagnostics;
5. develop uncertainty theory that distinguishes identification, regularity, and numerical error; and
6. explore a bounded PFN extension, preferably for a supplied graph or a direct causal query rather than graph discovery.

The full meeting record is summarized in [`2026-08-26-meeting.md`](./2026-08-26-meeting.md).

---

## 1. Statistical model

### 1.1 Structural definition

Let $`G=(N,E)`$ be a DAG. Partition its nodes by structural role into exogenous variables $`U=(U_1,\ldots,U_q)`$ and endogenous variables $`V=(V_1,\ldots,V_p)`$. The exogenous nodes have declared distributions,

```math
U_k \sim F_k(\eta_k), \qquad k=1,\ldots,q,
```

and the endogenous nodes are deterministic assignments evaluated in topological order:

```math
V_j=f_j\left(V_{\mathrm{pa}_V(j)},
              U_{\mathrm{pa}_U(j)};\theta_j\right),
\qquad j=1,\ldots,p.
```

The explicit exogenous nodes are the model's stochastic innovations. An endogenous node does not need a separate unnamed error term if one or more exogenous parents already play that role.

Together, the exogenous law and structural assignments define a joint model $`P_\theta(U,V)`$. For a dataset $`d`$, let $`O_d\subseteq N`$ be the nodes observed in that dataset. Fitting dataset $`d`$ uses the marginal distribution

```math
P_{\theta,d}^{O_d}(A)
=\int \mathrm{I}\left\{V_{O_d}(u;\theta)\in A\right\}\,dP_U(u).
```

This notation makes a distinction that the meeting regarded as architectural rather than cosmetic:

- exogenous/endogenous status is a property of the model;
- observed/latent status is a property of a dataset relative to the model.

The current formula language combines these by using `unobs(...)`. That works for one table but becomes awkward when two datasets observe different margins of the same system.

### 1.2 Formula example

The current package accepts formulas such as:

```text
ex, ey ~ Normal(0,1);
x <- 1*ex;
y <- x + 1*ey;
unobs(ex,ey)
```

Here `ex` and `ey` are standard-normal exogenous nodes. `x` and `y` are endogenous. The coefficients on `ex -> x` and `ey -> y` are fixed at one; the bare `x` term in the equation for `y` is learned.

Under the current API, `unobs(ex,ey)` also makes `x` and `y` the observed table columns. Under the meeting's preferred design, that last line would move out of the model formula and into a dataset schema.

### 1.3 In what sense is SEM2 semiparametric?

A linear formula with a fixed number of coefficients is parametric. SEM2 also permits nonlinear assignments such as `n(x,u)`, implemented by configurable multilayer perceptrons. Mixing explicit path coefficients and distributional assumptions with flexible neural mechanisms gives the project its semiparametric motivation.

A fixed neural architecture is still finite-dimensional. The careful description is therefore a neural-sieve approximation to a semiparametric model unless the project specifies a growing architecture and proves the corresponding approximation, consistency, and inference results. This qualification matters for rates, confidence intervals, and efficiency claims. It does not reduce the practical value of a flexible structural simulator.

### 1.4 Observational fitting

For one dataset with independent observed rows $`z_1,\ldots,z_n`$, the package aims to minimize a criterion of the form

```math
L(\theta)
=D\left(\widehat P_n^{O},P_\theta^{O}\right)
+\lambda R(\theta).
```

The first term may be a negative likelihood, moment discrepancy, energy score, or another distributional criterion. The second may be ordinary regularization or a scientific functional used to trace a partially identified set.

For several datasets, the natural extension is

```math
L(\theta)=\sum_{d=1}^{m} w_d
D_d\left(\widehat P_{n_d}^{O_d},P_{\theta,d}^{O_d}\right)
+\lambda R(\theta),
```

where each dataset has its own observed-node mapping and regime. The weights $`w_d`$ must be explicit; simply adding losses can give a large dataset or an arbitrarily scaled discrepancy unintended dominance.

The meeting also considered mixing observational and experimental data in this way. The model can express such a combination, but the data and fitting APIs do not yet implement it.

---

## 2. Current software architecture

### 2.1 Executable path

The reviewed architecture is:

```text
model formula
    |
    v
formula_parser.py  --> typed statements and semantic checks
    |
    v
dag.py             --> node specifications, predecessors,
    |                    topological order, inferred SEM type,
    |                    current observed/latent partition
    v
nodes.py           --> exogenous distribution / linear / neural /
    |                    finite discrete mechanisms
    v
model class
    +-- SEM2         : general continuous structural sampler
    +-- SEM2Linear   : exact path and moment algebra
    +-- SEM2Discrete : finite-support probability calculation
    |
    v
operator
    +-- Gaussian likelihood or covariance discrepancy
    +-- continuous energy loss
    +-- finite-state energy loss
    +-- potential outcome or ATE
    |
    v
optimizer / trainer --> fitted parameters and diagnostics
```

The parser-to-model part is more coherent than the final table-to-fit part.

### 2.2 Formula parser and DAG

[`formula_parser.py`](../../sem2/model/construction/formula_parser.py) recognizes:

- exogenous distribution statements such as `u ~ Normal(0,1)`;
- linear assignments such as `y <- x + 0.5*z`;
- nonlinear assignments such as `y <- n(x,u)`;
- finite discrete assignments such as `x <- d(z,u)`;
- fixed coefficients written explicitly;
- `unobs(...)` declarations; and
- comma expansion on the left-hand side.

Typed syntax objects live in [`sem_ast.py`](../../sem2/model/construction/sem_ast.py), with semantic types in [`sem_types.py`](../../sem2/model/construction/sem_types.py). [`dag.py`](../../sem2/model/construction/dag.py) builds the node specifications, checks graph validity, derives a topological order, and classifies the model as linear Gaussian, linear non-Gaussian, nonlinear, discrete, or mixed.

The formula is an effective low-dimensional interface. A high-dimensional system will need a programmatic constructor, schema tooling, or higher-level graph builder rather than one long string.

### 2.3 Runtime nodes

[`nodes.py`](../../sem2/model/components/nodes.py) contains the executable mechanisms:

- `NodeDistribution` wraps a PyTorch distribution and uses reparameterized sampling where available;
- `NodeLinear` combines learned and fixed parent coefficients, with optional bias;
- `NodeNN` wraps a multilayer perceptron;
- `NodeMLPDiscrete` maps parent contexts to categorical probabilities.

A continuous `sample(...)` call evaluates nodes in topological order and stores the values for that call. `set_nodes` can replace a selected node before its ordinary assignment is evaluated.

### 2.4 Model classes

[`base.py`](../../sem2/model/models/base.py) defines the common `SEM2` constructor and continuous sampler. [`linear.py`](../../sem2/model/models/linear.py) adds coefficient, intercept, exogenous-moment, implied-mean, and implied-covariance calculations. [`discrete.py`](../../sem2/model/models/discrete.py) builds a Cartesian table of finite-node states, evaluates conditional probabilities, masks states ruled out by interventions, and returns probabilities after marginalizing latent dimensions.

The supported import surface is currently:

```python
from sem2.model import SEM2, SEM2Discrete, SEM2Linear, SEMType
```

The package root [`sem2/__init__.py`](../../sem2/__init__.py) is empty. Notebooks that import these classes directly from `sem2` do not match the present API.

### 2.5 Operators and trainers

Operators separate structural sampling from scalar or vector queries. The strongest current fitting operators are in [`operators/linear.py`](../../sem2/operators/linear.py):

- `LogLikelihood` for exact linear-Gaussian likelihood, with a documented Gaussian pseudo-likelihood interpretation for other linear models;
- `WeightedFrobeniusDistance` for covariance matching; and
- a still-unimplemented `PathAnalysis` placeholder.

[`operators/scalar_operators.py`](../../sem2/operators/scalar_operators.py) contains continuous/discrete energy wrappers, mean potential outcomes, ATE, and a compositional `SEM2Loss`. [`sem2trainer.py`](../../sem2/sem2trainer.py) and [`experiments/trainer.py`](../../experiments/trainer.py) provide overlapping training paths.

The operator abstraction is a good place to express the meeting's model/data distinction. Some operators depend only on a model; others require a dataset view, query roles, and a regime.

### 2.6 Proposed data-aware architecture

The meeting points toward this revised flow:

```text
structural formula --------> model DAG and mechanisms
                                  |
dataset + schema ----------> observed-node view and regime
                                  |
causal/statistical query ---> named treatment, outcome, target
                                  |
                                  v
                         graph-generic operator
                                  |
                                  v
                      optimizer and diagnostics
```

A dataset schema should record at least:

- column-to-node mapping and order;
- observed nodes for that dataset;
- support, dtype, shape, and missing-value policy;
- observational or interventional regime;
- fixed intervention values when applicable;
- sample weight or loss weight; and
- named treatment/outcome roles for query-specific operators.

That design would permit one model to be fitted to several observed margins and remove binary-IV names from generic operators.

---

## 3. Fitting routes and their current status

### 3.1 Linear Gaussian

For a linear system

```math
V=BV+c+\varepsilon,
```

with exogenous covariance contribution $`\Omega`$, the implied moments are

```math
\mu=(I-B)^{-1}c,
\qquad
\Sigma=(I-B)^{-1}\Omega(I-B)^{-T}.
```

`SEM2Linear` computes these quantities. `LogLikelihood` evaluates the observed Gaussian log-likelihood using a Cholesky factorization and triangular solves. This is an exact, deterministic objective for a linear Gaussian model.

The meeting questioned the rough Adam loop in the earlier review example. For the one-coefficient example below, L-BFGS with a closure converged in three closure evaluations to the same finite-sample estimate. This does not establish that L-BFGS is always preferable, but examples should use a suitable optimizer and report convergence information rather than an arbitrary iteration count.

### 3.2 Linear non-Gaussian

The model and moment algebra can represent linear non-Gaussian exogenous distributions. The current `LogLikelihood` explicitly treats its criterion as a Gaussian pseudo-log-likelihood outside the Gaussian case. A general observed-data likelihood for arbitrary exogenous distributions is not supplied.

Research reports discuss a change-of-variables/ICA route, but that route is not a core operator in the reviewed package. It should be integrated and tested or kept clearly labeled as research code.

### 3.3 Nonlinear continuous and energy score

When a tractable density is unavailable, the package can compare generated and observed rows. One population energy distance is

```math
E_\beta(P,Q)
=2\,\mathrm{E}\|X-Y\|^\beta
-\mathrm{E}\|X-X'\|^\beta
-\mathrm{E}\|Y-Y'\|^\beta,
```

for $`0<\beta<2`$. The observed-observed term does not depend on $`\theta`$ and may be omitted during optimization.

The current [`EnergyLoss`](../../sem2/operators/scalar_operators.py) remains experimental for two concrete reasons:

1. `_evaluate_continuous` concatenates nodes named `Z`, `X`, and `Y` instead of deriving the observed variables from a schema.
2. `_energy_score` sets `m = preds.shape[1]`, which is the feature count, then uses $`m/(m-1)`$ as a correction for a generated-sample distance matrix. That correction should depend on the number of generated draws. In one feature the current expression divides by zero.

The implementation needs an explicit U- or V-statistic definition, a tensor-axis audit, analytical tests, and gradient checks before it supports general fitting claims.

### 3.4 MMD, Sinkhorn, and projected scores

MMD has population form

```math
\mathrm{MMD}_k^2(P,Q)
=\mathrm{E}\,k(X,X')
+\mathrm{E}\,k(Y,Y')
-2\mathrm{E}\,k(X,Y).
```

The current configuration mentions MMD, but `SEM2Loss` raises `NotImplementedError` when `use_mmd=True`.

The earlier review described Sinkhorn as report-only because no operator was present. The meeting adds useful context: a contributor has implemented it in a local fork and used it in a regression-discontinuity study. Preliminary experiments suggested better behavior than MMD under heavy tails. This remains unverified in the current tree and should not be stated as a general performance result.

Before integration, the Sinkhorn submission should specify:

- the exact divergence and entropy/debiasing convention;
- the backend and numerical tolerances;
- tensor shapes, batching, and device behavior;
- gradients and failure handling;
- controlled comparisons with energy score and MMD; and
- the difference between empirical training behavior and statistical robustness.

Projected or sliced energy scores were proposed for high-dimensional data. The approach averages lower-dimensional, often one-dimensional, projected scores. Sliced Wasserstein was also mentioned. These are research candidates, not current package features. A useful benchmark would vary dimension, sample size, tail behavior, projection count, compute budget, and model misspecification.

### 3.5 Finite discrete

`SEM2Discrete` enumerates the Cartesian product of discrete endogenous states and calculates exact probability masses after averaging over sampled continuous exogenous inputs. This gives an exact finite-state conditional calculation for a fixed set of exogenous draws; it is not an ordinary sampled category vector.

The discrete energy wrapper is specialized to a binary instrumental-variable table. It assumes the first column is `Z`, sets a node named `"Z"`, uses the remaining columns as `(X,Y)`, and loops over `range(2)`. Those assumptions should move into a named IV operator rather than remain in a generic loss.

The meeting confirmed that a general discrete likelihood was not implemented and that some paper code remained outside the repository. The reported approach uses direct probabilities rather than Gumbel-Softmax. The package should state clearly when it uses:

- exact state enumeration;
- Monte Carlo integration over continuous exogenous variables;
- continuous relaxations; or
- a reparameterization such as Gumbel-Softmax.

Exact enumeration scales exponentially in the number and cardinality of discrete nodes. State-count and memory checks are needed before construction, followed later by factorization, variable elimination, sparse support, or controlled approximations if larger models are a goal.

### 3.6 Mixed models

The parser can classify a graph as mixed, but the fitting stack does not support mixed discrete/continuous models end to end. Construction or fitting should fail early with a precise message until a complete objective and gradient strategy exist.

---

## 4. Causal computation, identification, and uncertainty

### 4.1 Equation replacement

For a causal SCM, intervening on $`X`$ at $`x`$ replaces the structural assignment for $`X`$ with $`X:=x`$ and leaves the other assignments unchanged. SEM2's `set_nodes` mechanism performs this calculation. In general,

```math
P_\theta(Y\mid \mathrm{do}(X=x))
\ne P_\theta(Y\mid X=x).
```

The operation is computationally present for continuous and finite discrete models. Its causal interpretation depends on the supplied graph and exogenous assumptions.

### 4.2 What fit does not establish

A small observational discrepancy supports only the claim that the model reproduces selected features of the observed margin. It does not establish:

- edge directions or absence of omitted common causes;
- correctness of exogenous distributions or independence assumptions;
- consistency or positivity for the intended intervention;
- transportability between observational and experimental regimes;
- point identification of the query; or
- smoothness of the query under the fitting discrepancy.

The package should record the estimand and assumptions attached to a causal query rather than returning an unqualified scalar.

### 4.3 Partial identification

For the discrete IV application, many structural distributions may fit the observed law while implying different treatment effects. The research approach uses

```math
L_\lambda(\theta)
=D\left(\widehat P_n^O,P_\theta^O\right)
+\lambda a(\theta),
```

where $`a(\theta)`$ is an ATE or another causal functional. Small positive and negative values of $`\lambda`$ steer optimization toward different compatible effects.

This is a promising computational strategy, but one run in each direction is not an identified-set procedure. A reusable implementation needs:

- a stated observational-fit tolerance;
- multi-start optimization and continuation over $`\lambda`$;
- checks that extrema remain within the fit tolerance;
- convergence and boundary diagnostics;
- separation of Monte Carlo and optimizer variation; and
- a theorem or independent calculation when claiming sharp bounds.

### 4.4 Identification versus regularity

The meeting separated structural identification from regularity under the chosen loss. Even if the assumptions identify $`a(P)`$, stable inference may require a bound such as

```math
|a(P)-a(Q)|\leq C\,D(P,Q)^\gamma
```

for suitable $`C`$ and $`\gamma`$, or a differentiability condition leading to an influence function. The relevant condition depends on the target, model class, observation map, and discrepancy.

This issue is especially important for nonlinear latent-variable models. A flexible generator can fit the observed margin while its causal query remains highly sensitive in directions that the loss barely penalizes.

### 4.5 Confidence intervals and other uncertainty statements

SEM2 currently produces fitted parameters or functionals from one fitted model. A complete uncertainty account should separate:

1. sampling uncertainty in the observed data;
2. Monte Carlo error in simulation-based objectives and queries;
3. numerical optimization and restart variability;
4. model and graph sensitivity; and
5. structural non-identification.

Linear Gaussian models are the simplest place to add observed-information, Hessian, sandwich, or bootstrap calculations. Nonlinear models may require differentiability and influence-function theory, a carefully designed bootstrap, or a Bayesian analysis. Partially identified models need confidence regions for an identified set rather than an ordinary interval around a selected extremum.

The meeting accepted a pragmatic research goal alongside formal theory: report the range of causal conclusions among well-fitting generative models. That can be useful, but the output must say whether the range is a numerical sensitivity set, an approximate identified set, a posterior credible region, or a confidence region.

---

## 5. Revised reproducible examples

These examples were executed from the reviewed tree with fixed seeds. They demonstrate API behavior only. They are not a benchmark, an estimator-validation study, or evidence that the graph is causally correct.

### 5.1 Fit a path coefficient, then query the fitted model

This replaces the earlier sampling-only example. It generates a two-column table with true coefficient $`1.5`$, fits the free `x -> y` coefficient by exact Gaussian likelihood, and then computes the fitted model's intervention contrast.

```python
import torch
from sem2.default_configs import LinearNodeConfig
from sem2.model import SEM2Linear
from sem2.operators.linear import LogLikelihood

# Synthetic table. Its columns match model.observed_vars below.
torch.manual_seed(123)
n = 1_000
x = torch.randn(n, 1)
y = 1.5 * x + torch.randn(n, 1)
table = torch.cat([x, y], dim=1)

model = SEM2Linear(
    "ex, ey ~ Normal(0,1); "
    "x <- 1*ex; "
    "y <- x + 1*ey; "
    "unobs(ex,ey)",
    node_cfgs={
        "x": LinearNodeConfig(use_bias=False),
        "y": LinearNodeConfig(use_bias=False),
    },
)
criterion = LogLikelihood(model, centered=True)
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=40,
    tolerance_grad=1e-10,
    tolerance_change=1e-12,
    line_search_fn="strong_wolfe",
)

closure_calls = 0

def closure():
    global closure_calls
    closure_calls += 1
    optimizer.zero_grad()
    mean_log_likelihood, _ = criterion.evaluate(table)
    loss = -mean_log_likelihood
    loss.backward()
    return loss

optimizer.step(closure)
mean_log_likelihood, _ = criterion.evaluate(table)
learned = model.node_dict["y"].effective_weight()[0, 0].item()

# Use common random numbers so the contrast isolates equation replacement.
m = 10_000
torch.manual_seed(999)
y0, _ = model.sample(
    batch_size=m,
    set_nodes={"x": torch.zeros(m, 1)},
)
torch.manual_seed(999)
y1, _ = model.sample(
    batch_size=m,
    set_nodes={"x": torch.ones(m, 1)},
)
contrast = (y1["y"] - y0["y"]).mean().item()

print(model.observed_vars)
print(round(learned, 4))
print(round(mean_log_likelihood.item(), 4))
print(closure_calls)
print(round(contrast, 4))
```

Verified output:

```text
['x', 'y']
1.5229
-2.8476
3
1.5229
```

The fitted value differs from $`1.5`$ because the table is finite. The model intervention contrast equals the fitted path coefficient in this simple linear SCM. That equality follows from the specified equations; it does not prove that an observational application identifies the same causal effect.

The example also exposes a data-interface requirement: the tensor columns must be aligned with `model.observed_vars`. A production API should verify that mapping by name.

### 5.2 Finite-state intervention calculation

The next example makes the finite-state output and causal query explicit. The model is randomly initialized and not fitted, so the reported contrast has no empirical meaning.

```python
import torch
from sem2.default_configs import MLPConfig
from sem2.model import SEM2Discrete

torch.manual_seed(19)
model = SEM2Discrete(
    "u ~ Uniform(0,1); "
    "z ~ Categorical(probs=[1,1]); "
    "x <- d(z,u); "
    "y <- d(x,u); "
    "unobs(u)",
    node_cfgs={
        "x": MLPConfig(hidden_dims=[8], output_dim=2),
        "y": MLPConfig(hidden_dims=[8], output_dim=2),
    },
)

m = 2_000
torch.manual_seed(456)
p0, extra = model.sample(
    batch_size=m,
    set_nodes={"x": torch.zeros(m, 1, dtype=torch.long)},
)
torch.manual_seed(456)
p1, _ = model.sample(
    batch_size=m,
    set_nodes={"x": torch.ones(m, 1, dtype=torch.long)},
)

y_position = model.state_node_names.index("y")
y_values = extra["states"][:, y_position].to(p0.dtype)
mean_y0 = (p0.mean(dim=0) * y_values).sum()
mean_y1 = (p1.mean(dim=0) * y_values).sum()

print(model.state_node_names)
print(extra["states"].tolist())
print(round(mean_y0.item(), 4))
print(round(mean_y1.item(), 4))
print(round((mean_y1 - mean_y0).item(), 4))
```

Verified output:

```text
['x', 'y']
[[0, 0], [0, 1], [1, 0], [1, 1]]
0.5105
0.4967
-0.0138
```

Each row of `p0` or `p1` is a probability vector over the listed `(x,y)` states after setting `x`. The calculation averages over sampled `u` values and then marginalizes the finite support. The number $`-0.0138`$ is only the contrast implied by this unfitted random model.

---

## 6. Findings and requested updates

### 6.1 Fix before relying on the generic fitter

#### A. Correct and test the energy score

**Observation.** The diagonal correction uses feature count instead of generated-sample count, and the wrapper selects hard-coded node names.

**Impact.** The numerical objective is wrong in general and undefined for one feature. The public class cannot fit an arbitrary graph.

**Requested update.** Specify the exact finite-sample estimator, correct its axes, derive simple hand-checkable cases, add gradient tests, and build the generated table from an explicit observed-node schema.

#### B. Define a data contract

**Observation.** Current operators and trainers leave column order, naming, dtype, support, device transfer, missing values, and intervention regime partly implicit. `SEM2Trainer._fit_batch` unpacks `(Xs, targets)` and ignores `Xs`; the supplied loader is a synthetic binary-IV adapter.

**Impact.** A numerically valid tensor can be attached to the wrong nodes without an early error. Multiple observed margins cannot be represented cleanly.

**Requested update.** Add a typed dataset view or adapter that maps columns to model nodes and records regime and query roles. Move `unobs(...)` out of the long-term structural formula design after a backward-compatible transition.

#### C. Separate generic and design-specific operators

**Observation.** The discrete energy path contains binary-IV conventions, while ATE depends on treatment and outcome names in configuration.

**Impact.** Generic class names overstate their scope and make new designs harder to add safely.

**Requested update.** Keep graph-generic operators free of benchmark names. Put grouped binary-IV fitting and bound tracing in a clearly named specialized workflow built on the generic layer.

#### D. Consolidate training paths and diagnostics

**Observation.** Package and experiment trainers overlap, while examples can bypass both with manual loops. Optimizer choice, stopping, restart policy, and objective traces are not standardized.

**Impact.** Results are difficult to compare and a plausible estimate can hide non-convergence.

**Requested update.** Define one supported trainer interface with optimizer-specific closures, stopping criteria, traces, seed control, restart summaries, and post-fit checks. Use closed-form solutions where available and document when L-BFGS, Adam, SGD, or another method is appropriate.

#### E. Test the estimator and causal layers directly

**Observation.** The 190 passing tests cover parsing, construction, sampling, and linear behavior well, but they do not directly exercise `SEM2Trainer`, `EnergyLoss`, `DiscreteEnergyLoss`, `SEM2Loss`, ATE, or the potential-outcome operators as complete workflows.

**Impact.** A green test suite does not validate the generic fitting or causal-query paths.

**Requested update.** Add analytical small-model tests, schema-error tests, gradient checks, intervention tests, and one convergence test per supported regime. For partial identification, include a problem with known bounds and numerical acceptance tolerances.

### 6.2 Review external research implementations

The meeting resolved an ambiguity in the earlier report. Sinkhorn and the remaining discrete work are not absent ideas; they are implementations outside or partly outside the reviewed tree.

Each submission should arrive as a bounded branch or pull request with:

- exact method and assumptions;
- dependencies and configuration;
- deterministic examples;
- tests against analytical or independent references;
- benchmark scripts with seeds and acceptance criteria;
- explicit supported tensor shapes and model types; and
- a note separating research findings from package guarantees.

The package should maintain a capability matrix with four statuses:

| Capability | Current status in reviewed tree | Meeting update |
|---|---|---|
| Formula parsing and DAG construction | Implemented and tested | Retain; eventually separate observation schema. |
| Continuous and linear sampling | Implemented and tested | Retain. |
| Exact linear-Gaussian likelihood | Implemented and tested | Use as the first stable fitting route. |
| Continuous energy fitting | Implemented but defective/specialized | Repair estimator and schema assumptions. |
| MMD | Explicitly unimplemented | Candidate only. |
| Sinkhorn divergence | Not in reviewed tree | Local implementation awaits review. |
| Exact finite-state probabilities | Implemented and tested at the model layer | Scaling remains limited. |
| General discrete fitting | Specialized/incomplete | Remaining paper code awaits review. |
| Mixed-model fitting | Not implemented end to end | Fail early until supported. |
| Intervention calculation | Implemented at model/operator level | Add direct tests and assumption metadata. |
| Partial-identification workflow | Components present, workflow incomplete | Add sweeps, tolerances, restarts, and bound validation. |
| Confidence intervals | Not implemented | Research priority. |
| PFN posterior approximation | Not implemented in SEM2 | Future research direction. |

### 6.3 Improve discrete scale and semantics

Before increasing scale, the package should report state counts and estimated memory. The return type should make clear which dimensions index exogenous Monte Carlo draws, finite states, and observed variables. Later work can consider factorized inference or controlled approximations.

If continuous relaxations are added, they should not be described as exact discrete likelihood. If Gumbel-Softmax is added, temperature, bias, and evaluation semantics need explicit tests.

### 6.4 Repair documentation and package surface

[`README.md`](../../README.md) still contains only a heading. [`DESIGN.md`](../../DESIGN.md) describes an older API, including imports and calls that no longer match the package. Several notebooks use stale module paths or undeclared dependencies.

A useful first documentation release would include:

- an installable minimal example using the actual import surface;
- the structural formula grammar;
- the table/schema contract;
- one supported fit per stable regime;
- diagnostics and causal-interpretation warnings;
- a current architecture diagram;
- a capability matrix; and
- CI that executes retained examples.

The top-level exports, runtime versus development dependencies, license, contribution guidance, and placeholder APIs also need deliberate decisions.

---

## 7. PFNs as a possible extension

### 7.1 Statistical target

A PFN can be framed as a posterior approximator. Simulate

```math
\theta\sim\pi,
\qquad N\sim p_N,
\qquad Z_1,\ldots,Z_N\mid\theta\sim P_\theta,
```

then train a function $`F_\phi`$ that maps the variable-size dataset to a distribution over $`\theta`$. With a proper scoring rule $`S`$, the population objective is

```math
\mathrm{E}\left[S\left(F_\phi(Z_1,\ldots,Z_N),\theta\right)\right].
```

If the simulator and prior define the desired joint law, the predictive family can represent the posterior, and population optimization succeeds, the target is the corresponding posterior distribution of $`\theta`$ given the dataset.

A transformer can process variable-size sets or sequences. A simple output head could give a normal mean and covariance; more flexible heads could represent skewness, heavy tails, or multimodality.

### 7.2 Value and computational condition

Current SEM2 optimization is repeated for each dataset. A trained PFN replaces that per-dataset optimization with a forward evaluation. The up-front simulation and transformer training are worthwhile only if the approximation is reused enough times or if it is substantially more reliable than the available per-dataset alternatives.

A PFN is unnecessary if a point estimator and trustworthy asymptotic variance already provide the desired normal approximation cheaply. Its potential value is finite-sample posterior structure or rapid repeated inference.

### 7.3 Prior and simulator risk

The prior is part of the estimator. Linear coefficient priors can often be stated directly. A prior on neural-network weights is harder to interpret because its induced distribution over structural functions may not reflect scientific beliefs.

The proper-score argument does not protect against simulator misspecification, prior shift, finite model capacity, finite training data, or failed optimization. Sergio cited related ridge-regression work in which mild prior shift caused poor transformer generalization. A SEM2 PFN should therefore be evaluated under:

- in-prior calibration;
- controlled coefficient and function-prior shifts;
- graph-size and sparsity shifts;
- exogenous-distribution shifts;
- sample-size shifts;
- structural misspecification; and
- comparisons with MCMC, variational, Laplace, and direct-optimization baselines where feasible.

### 7.4 Choose the amortization level deliberately

The meeting distinguished:

1. a PFN for repeated datasets under one fixed graph;
2. a PFN that takes a user-supplied graph as input;
3. a PFN that outputs graph structure; and
4. a PFN that predicts the posterior of a causal query directly.

Fixed-graph inference is easiest to define and validate. Graph-as-input better matches SEM2's mission of fitting scientifically supplied structures, but it enlarges the training distribution. Graph-as-output introduces causal-discovery and equivalence problems and should not be the first SEM2 extension.

A direct causal-query PFN may be narrower than the full posterior over $`\theta`$. It could simulate a context dataset plus an intervention query and predict the posterior distribution of the resulting effect or counterfactual response. The lower-dimensional output may be easier, but its validity remains tied to the prior, graph, simulator, and query definition.

Related workshop work reported in the meeting trained on random SCMs and causal outputs at scales up to roughly 35 nodes and 1,000 observations. That is useful prior experience, not evidence that the approach will be calibrated for SEM2's intended models.

### 7.5 Recommended first PFN study

A bounded first study would use a fixed, low-dimensional graph and a transparent linear or mildly nonlinear prior. It should compare:

- exact or high-quality posterior inference;
- a Laplace or asymptotic normal approximation;
- the PFN posterior;
- direct SEM2 point fitting; and
- a PFN trained directly on one causal query.

The study should evaluate calibration, proper-score risk, causal-functional coverage, prior shift, sample-size variation, compute cost, and failure modes. Only after that should the graph be added as an input.

---

## 8. Revised order of work

### Phase 1: make one table-to-fit path trustworthy

1. Define the dataset schema and named column mapping.
2. Fix and test continuous energy distance.
3. Split graph-generic operators from binary-IV adapters.
4. Consolidate trainers and add convergence diagnostics.
5. Add direct tests for fitting, potential outcomes, ATE, and finite-state losses.
6. Fail early for unsupported mixed models and objectives.

A release checkpoint is one documented, tested fit for each claimed regime: linear Gaussian, nonlinear continuous, and finite discrete.

### Phase 2: reconcile repository and research code

1. Review the Sinkhorn/RDD submission.
2. Review and merge the remaining discrete implementation.
3. Decide whether the ICA/change-of-variables route belongs in the package.
4. Create a maintained capability matrix.
5. Turn retained applications into deterministic scripts with provenance, seeds, dependency records, and numerical thresholds.

### Phase 3: causal and statistical reliability

1. Implement a reusable partial-identification sweep with restarts and fit tolerances.
2. Record assumptions and estimands with every causal query.
3. Add held-out fit checks, implied-moment comparisons, optimization traces, and sensitivity runs.
4. Develop uncertainty for the linear route first.
5. State and study regularity conditions for nonlinear causal functionals.
6. Separate confidence regions, posterior regions, identified sets, and numerical sensitivity ranges in the API and documentation.

### Phase 4: scale and PFN research

1. Benchmark projected/sliced scores and Sinkhorn in controlled settings.
2. Add discrete state-count guards and investigate factorized inference.
3. Define mixed-model semantics if that remains a goal.
4. Run the bounded fixed-graph PFN study.
5. Consider graph-as-input amortization only after prior-shift behavior is understood.
6. Keep graph discovery separate until its target, prior, and identification limits are explicit.

The product target remains concrete: a user can state a fixed structural graph, map named data to it, fit a supported objective, inspect numerical and statistical diagnostics, run a declared query, and understand which conclusions depend on observational fit, causal assumptions, and identification.

---

## 9. Verification and evidence boundaries

### 9.1 Code and execution

For this update I:

- inspected the current merged review and relevant source modules;
- checked the supplied transcript against the brief summary;
- executed the revised linear and discrete examples;
- reran the full test suite against the reviewed tree:

```text
190 passed, 51 warnings in 38.01s
```

The warnings arise mainly because test helpers instantiate linear formulas through the generic `SEM2` class, which recommends `SEM2Linear`.

The full suite supports claims about the tested parser, model, sampler, and linear layers. It does not by itself validate the untested trainer and causal-operator workflows identified above.

### 9.2 Evidence hierarchy

Claims about shipped behavior follow the inspected repository and executed examples. The meeting transcript supplies project intent, contributor reports, and the status of work outside the repository. Those reports are not treated as code verification.

The following statements are therefore kept distinct:

- **Observed in the repository:** current classes, formulas, hard-coded assumptions, tests, and explicit `NotImplementedError` paths.
- **Reported in the meeting:** a local Sinkhorn implementation, preliminary heavy-tail behavior, remaining discrete code, prior PFN experiments, and intended future work.
- **Review inference:** the proposed data architecture, integration gates, and priority order.
- **Open uncertainty:** theoretical regularity, sharpness of numerical bounds, PFN robustness, and performance of unmerged implementations.

## 10. Sources

Repository sources:

- [`sem2/`](../../sem2/) and [`tests/`](../../tests/), for current behavior;
- [`MPhilSC_BothReport_Srivathsan_Amruth_Report_2_Starts_at_Page_69.pdf`](../reports/MPhilSC_BothReport_Srivathsan_Amruth_Report_2_Starts_at_Page_69.pdf), for the research record around discrete estimation, linear fitting, and the Sinkhorn/RDD work;
- [`AAAI_SEM2.pdf`](../reports/AAAI_SEM2.pdf), for early architecture and neural structural-equation framing; and
- [`2026-08-26-ai-review.md`](./2026-08-26-ai-review.md), for the original code-backed review and earlier example outputs.

Supplied sources:

- the 2026-08-26 meeting transcript;
- the bilingual brief meeting summary; and
- the rough draft *Machine learning methods for structural equation models* used in the original review for minimum-distance notation. That draft does not address causality and is not used here to support causal claims.

The equations use [GitHub's protected inline-math syntax and fenced `math` blocks](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions).
