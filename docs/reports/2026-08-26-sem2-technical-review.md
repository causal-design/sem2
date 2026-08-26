# SEM2 technical review: model, fitting architecture, and development priorities

**Review date:** 2026-08-26  
**Author:** Peirce (Nous Research Hermes Agent)  
**Repository state reviewed:** `main` at `b30c777a715ee4c4c076b595d770ec3d8dfb20ba`  
**Scope:** the current `sem2/` package, its tests and experiments, the reports under `docs/reports/`, and the attached rough draft *Machine learning methods for structural equation models*.

## Executive summary

SEM2 is best understood as a **formula-driven, differentiable structural generative modelling framework**. A formula defines a directed acyclic graph (DAG), exogenous distributions, deterministic structural assignments, latent/observed status, and optional fixed path coefficients. The package turns that declaration into PyTorch modules and exposes sampling, node replacement for interventions, exact calculations for some model classes, and differentiable fitting operators.

For independent tabular rows \(v_1,\ldots,v_n\), the common fitting principle is to choose structural parameters \(\theta\) so that the observed distribution induced by the graph, \(P_\theta^O\), is close to the empirical distribution \(\widehat P_n\). The statistically appropriate notion of “close” depends on the declared model:

- **Linear Gaussian:** use exact implied means and covariances and the exact Gaussian observed-data log-likelihood.
- **Linear non-Gaussian:** the graph and sampler are present, but a general package-level observed-data fitting operator is not.
- **Nonlinear continuous:** use differentiable simulation and a sample discrepancy. An energy-score operator exists, but its current implementation is experimental and contains a correction-factor defect; MMD is explicitly unimplemented and the Sinkhorn fitting described in Report 2 is not present in `sem2/`.
- **Finite discrete:** enumerate the finite state space, exactly marginalise latent states, and minimise a finite-support energy distance, optionally with a causal estimand penalty. The current loss wrapper is specialised to a binary \((Z,X,Y)\) instrumental-variable layout.
- **Mixed discrete/continuous:** the parser can classify a graph as mixed, but the model/operator stack does not yet provide a complete mixed-model fitting path.

The **model-construction core is substantially stronger than the top-level user and training interface**. The formula parser, DAG construction, node modules, continuous and finite-discrete samplers, and linear exact operators are coherent and covered by a sizeable test suite. During this review, the full suite completed with **190 passing tests**. The most urgent work is to make the fitting layer mathematically correct and graph-generic, reconcile Report 2’s claims with code that is actually shipped, and replace the almost empty or obsolete documentation.

A central scientific qualification is essential: a fitted SEM2 graph is a causal model **only under externally justified causal assumptions**. Observational distribution matching does not discover the causal DAG, prove exogeneity, or identify an intervention effect by itself. `set_nodes` correctly represents equation replacement inside the assumed structural model; it does not establish that the assumed model is causally valid for the data.

---

## 1. What is the semiparametric structural equation model?

### 1.1 Structural definition

Let \(G=(N,E)\) be a DAG with exogenous nodes \(U=(U_1,\ldots,U_q)\) and endogenous nodes \(V=(V_1,\ldots,V_p)\). A structural model consists of

\[
U_k \sim F_k(\eta_k), \qquad k=1,\ldots,q,
\]

and assignments evaluated in a topological order,

\[
V_j = f_j\!\left(V_{\operatorname{pa}_V(j)},
                  U_{\operatorname{pa}_U(j)};\theta_j\right),
\qquad j=1,\ldots,p.
\]

Here \(\operatorname{pa}_V(j)\) and \(\operatorname{pa}_U(j)\) are the endogenous and exogenous parents of \(V_j\). The joint exogenous law and the deterministic assignments induce a joint distribution \(P_\theta(U,V)\). If \(O\subseteq N\) is the set of observed nodes, the data likelihood or discrepancy is evaluated on the observed marginal

\[
P_\theta^O(A)=\int \mathbf 1\{V_O(u;\theta)\in A\}\,dP_U(u).
\]

In SEM2, the declaration `unobs(...)` specifies the latent set; all remaining graph nodes are treated as observed. Shared exogenous parents can induce latent confounding or residual covariance. The graph must be acyclic.

A SEM2 formula such as

```text
ex, ey ~ Normal(0,1);
x <- 1*ex;
y <- x + 1*ey;
unobs(ex,ey)
```

therefore means: `ex` and `ey` are latent exogenous standard-normal variables; `x` and `y` are observed; the coefficient on `ex -> x` and `ey -> y` is fixed at one; and the coefficient on `x -> y` is learned.

### 1.2 In what sense is the model semiparametric?

The repository uses *semiparametric* in a broad modelling sense: the graph and some distributional components are structured, while selected mechanisms can be flexible neural functions. In particular, `n(...)` creates a configurable multilayer perceptron rather than imposing a linear equation. This combines interpretable structural restrictions with a flexible function class.

There is an important precision point. The **implemented estimator is finite-dimensional for any fixed network architecture**. It is most accurately described as a neural sieve or flexible parametric approximation to a semiparametric model, unless the project additionally specifies an asymptotic regime in which network capacity grows and proves the relevant consistency or inference results. The attached rough draft motivates universal function approximation and minimum-distance fitting, but it does not supply that complete semiparametric theory. The distinction matters most when making claims about rates, efficiency, confidence intervals, or identification.

### 1.3 Observational fitting

Suppose the observed table is

\[
D=\{v_i^O\}_{i=1}^n, \qquad v_i^O\stackrel{\text{iid}}\sim P_0^O.
\]

The common SEM2 objective can be written as

\[
\widehat\theta
\in \arg\min_{\theta\in\Theta}
D\!\left(\widehat P_n^O,P_\theta^O\right)
+\lambda\,R(\theta),
\]

where \(D\) is a likelihood-based or minimum-distance criterion, and \(R\) may be a regulariser or a scientific target such as an average treatment effect (ATE). This formulation covers the repository’s main fitting regimes.

#### Exact linear-Gaussian route

For a linear SEM,

\[
V=BV+c+\varepsilon,
\]

with exogenous covariance contribution \(\Omega\), the implied moments are

\[
\mu=(I-B)^{-1}c,
\qquad
\Sigma=(I-B)^{-1}\Omega(I-B)^{-T}.
\]

[`SEM2Linear`](../../sem2/model/models/linear.py) constructs these quantities directly. [`LogLikelihood`](../../sem2/operators/linear.py) then evaluates the observed Gaussian log-likelihood using a Cholesky factorisation and triangular solves. This avoids Monte Carlo noise during optimisation. [`WeightedFrobeniusDistance`](../../sem2/operators/linear.py) provides a moment-discrepancy alternative.

#### Sample-based continuous route

When the observed density is unavailable but the model can be sampled differentiably, one can compare real and generated rows. The attached draft discusses energy distance and maximum mean discrepancy (MMD). For \(0<\beta<2\), an energy-distance form is

\[
\mathcal E_\beta(P,Q)
=2\,\mathbb E\|X-Y\|^\beta
-\mathbb E\|X-X'\|^\beta
-\mathbb E\|Y-Y'\|^\beta,
\]

where \(X,X'\stackrel{\text{iid}}\sim P\) and \(Y,Y'\stackrel{\text{iid}}\sim Q\). The term depending only on the observed data can be dropped when optimising \(\theta\). The current [`EnergyLoss`](../../sem2/operators/scalar_operators.py) implements this family for continuous samples, but Section 4 identifies a correctness issue and several hard-coded assumptions that should be fixed before treating it as a general estimator.

MMD has the analogous population form

\[
\operatorname{MMD}_k^2(P,Q)
=\mathbb E k(X,X')+\mathbb E k(Y,Y')-2\mathbb E k(X,Y).
\]

It is part of the mathematical setup and configuration surface, but `SEM2Loss` currently raises `NotImplementedError` when `use_mmd=True`.

Report 2 describes Sinkhorn-divergence fitting for its nonlinear regression-discontinuity study. No Sinkhorn/optimal-transport operator or dependency is present in the current `sem2/` package. That experiment should therefore be understood as report-level research code unless and until its fitting implementation is moved into the package and tested.

#### Exact finite-support discrete route

[`SEM2Discrete`](../../sem2/model/models/discrete.py) enumerates the Cartesian product of discrete-node states. For each global configuration it accumulates conditional log-probabilities, masks configurations incompatible with interventions, exponentiates, and sums over unobserved coordinates. Thus the output is an exact probability mass function over the observed finite state space rather than a non-differentiable categorical draw.

[`DiscreteEnergyLoss`](../../sem2/operators/discrete_energy_loss.py) compares the predicted mass \(p_\theta\) with observed states or an empirical mass vector. This is particularly useful in partially identified models because one can fit the observational law while tracing variation in a causal functional:

\[
\min_\theta
D\!\left(\widehat P_n^O,P_\theta^O\right)
+\lambda\,a(\theta),
\]

where, for example,

\[
a(\theta)
=\mathbb E_\theta[Y\mid do(X=1)]
-\mathbb E_\theta[Y\mid do(X=0)].
\]

The finite-state machinery and scalar causal operators are present. A reusable sweep, tolerance-envelope, restart, and identified-set reporting API is not.

### 1.4 Causal interpretation and its limits

Within a structural causal model, an intervention \(do(V_k=v)\) replaces the equation for \(V_k\) with the constant assignment \(V_k:=v\), leaving the other assignments unchanged. SEM2’s `set_nodes={...}` mechanism performs this node replacement during topological evaluation. It is not ordinary conditioning:

\[
P_\theta(Y\mid do(X=x))
\ne P_\theta(Y\mid X=x)
\]

in general, especially with confounding.

However, the code does not infer or validate the causal graph. The following must come from design, domain knowledge, or a separate identification argument:

1. the edge directions and absence of omitted parents;
2. which roots can be treated as exogenous;
3. independence or dependence assumptions among exogenous variables;
4. positivity/support and consistency assumptions;
5. the identification conditions for the requested causal estimand.

Consequently, good observational fit is evidence that \(P_\theta^O\) approximates the observed law. It is **not** by itself evidence that the intervention distribution is identified or that the DAG is true. The discrete IV work is valuable precisely because it treats this honestly as partial identification rather than forcing a point estimate.

---

## 2. Software architecture

### 2.1 End-to-end flow

```text
formula string
    |
    v
formula_parser.py  --> typed statements and validation
    |
    v
dag.py             --> node specifications, predecessor map,
    |                    topological order, observed/latent sets,
    |                    inferred SEMType
    v
nodes.py           --> NodeDistribution / NodeLinear / NodeNN /
    |                    NodeDiscrete
    v
model class
    +-- SEM2         : general continuous sampler
    +-- SEM2Linear   : exact coefficient/moment algebra
    +-- SEM2Discrete : finite-support exact marginalisation
    |
    v
operator
    +-- exact likelihood or covariance discrepancy
    +-- sample energy loss
    +-- finite-state energy distance
    +-- potential-outcome / ATE functional
    |
    v
optimiser / trainer --> fitted parameters and diagnostics
```

### 2.2 Formula and graph construction

[`formula_parser.py`](../../sem2/model/construction/formula_parser.py) is the language front end. It recognises:

- exogenous distribution statements: `u ~ Normal(0,1)`;
- linear assignments: `y <- x + 0.5*z`;
- nonlinear assignments: `y <- n(x,u)`;
- discrete assignments: `x <- d(z,u)`;
- latent declarations: `unobs(u,e)`;
- comma expansion on the left-hand side;
- fixed coefficients written explicitly, while bare parent terms are learned.

The parser produces typed AST objects from [`sem_ast.py`](../../sem2/model/construction/sem_ast.py) and semantic types from [`sem_types.py`](../../sem2/model/construction/sem_types.py). [`dag.py`](../../sem2/model/construction/dag.py) converts these statements into a single `NodeSpec` mapping, rejects duplicate/cyclic/undefined constructions, computes topological order, records observed and unobserved nodes, and classifies the model as linear Gaussian, linear non-Gaussian, nonlinear, discrete, or mixed.

### 2.3 Runtime node layer

[`nodes.py`](../../sem2/model/components/nodes.py) implements the graph’s executable nodes:

- `NodeDistribution` wraps `torch.distributions` and uses reparameterised sampling when available.
- `NodeLinear` concatenates parent tensors, applies learned and fixed path coefficients, and optionally adds a bias.
- `NodeNN` wraps an MLP from [`nn_models.py`](../../sem2/model/components/nn_models.py).
- `NodeDiscrete` maps discrete-parent contexts to categorical probabilities.

The model caches no global table state: each `sample(...)` call walks the graph in topological order and builds a node-value dictionary for that call. `set_nodes` can replace any selected node value before its ordinary structural assignment is evaluated.

### 2.4 Model classes

[`base.py`](../../sem2/model/models/base.py) defines `SEM2`, the common constructor and continuous sampler. It registers nodes in an `nn.ModuleDict`, exposes `node_dict`, `observed_vars`, `unobserved_vars`, `inferred_sem_type`, and `sample(...)`, and deliberately rejects `forward(...)` so that users call `sample(...)` explicitly.

[`linear.py`](../../sem2/model/models/linear.py) adds exact linear operations:

- `coef_matrix()`;
- `intercept_vector()`;
- `exog_mean_vector()` and `exog_cov_matrix()`;
- `mean_vector()` and `cov_matrix()`;
- `observed_mean_cov()`.

[`discrete.py`](../../sem2/model/models/discrete.py) creates the global finite-state table, records state-coordinate mappings, deduplicates repeated parent contexts before evaluating each discrete node, accumulates joint log-probabilities, and marginalises latent coordinates.

The public model import is currently

```python
from sem2.model import SEM2, SEM2Discrete, SEM2Linear, SEMType
```

The top-level [`sem2/__init__.py`](../../sem2/__init__.py) is empty, so the many notebooks that use `from sem2 import SEM2` do not match the current public surface.

### 2.5 Configuration and fitting layer

[`default_configs.py`](../../sem2/default_configs.py) contains Pydantic/dataclass configurations for neural, linear, discrete, loss, and training settings. [`linear.py`](../../sem2/operators/linear.py) contains the strongest package-level fitting API: deterministic Gaussian likelihood and covariance-based criteria.

The generic fitting stack consists of:

- [`SEM2Loss`](../../sem2/operators/scalar_operators.py), which composes energy and ATE terms;
- [`SEM2Trainer`](../../sem2/sem2trainer.py), which constructs Adam or SGD and iterates a loader;
- [`dataloader.py`](../../sem2/dataloader.py), which currently supplies only a synthetic binary-IV dummy loader.

This stack should be treated as a prototype. It is not yet a graph-generic tabular-data estimator: the loss has variable-name/layout assumptions, and the trainer discards the first member of each `(Xs, targets)` batch.

---

## 3. Reproducible illustrative examples

These examples are deliberately small. They illustrate the current API; they are not simulation evidence for broad statistical or causal claims. They were executed from the reviewed repository with fixed random seeds.

### 3.1 Sampling and intervention in a linear SEM

```python
import torch
from sem2.default_configs import LinearNodeConfig
from sem2.model import SEM2Linear

torch.manual_seed(7)
formula = (
    "ex, ey ~ Normal(0,1); "
    "x <- 1*ex; "
    "y <- 2*x + 1*ey; "
    "unobs(ex,ey)"
)
model = SEM2Linear(
    formula,
    node_cfgs={
        "x": LinearNodeConfig(use_bias=False),
        "y": LinearNodeConfig(use_bias=False),
    },
)

observational, _ = model.sample(batch_size=4)

torch.manual_seed(7)
interventional, _ = model.sample(
    batch_size=4,
    set_nodes={"x": torch.ones(4, 1)},
)

print(model.observed_vars)
print(observational["y"].flatten().round(decimals=4).tolist())
print(interventional["y"].flatten().round(decimals=4).tolist())
```

Verified output:

```text
['x', 'y']
[-1.277, 2.8184, 4.4758, -0.0814]
[3.6908, 1.1052, 1.6444, 3.2324]
```

The second call replaces the equation for `x`; it does not condition an observational sample on `x == 1`.

### 3.2 Fitting one path coefficient to a synthetic table

The next example generates a two-column table with true coefficient \(1.5\), then fits the learned `x -> y` coefficient with the exact linear-Gaussian likelihood.

```python
import torch
from sem2.default_configs import LinearNodeConfig
from sem2.model import SEM2Linear
from sem2.operators.linear import LogLikelihood

torch.manual_seed(123)
n = 1_000
x = torch.randn(n, 1)
y = 1.5 * x + torch.randn(n, 1)
table = torch.cat([x, y], dim=1)  # columns follow model.observed_vars

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for _ in range(300):
    optimizer.zero_grad()
    mean_log_likelihood, _ = criterion.evaluate(table)
    (-mean_log_likelihood).backward()
    optimizer.step()

x_index = model.dag.order.index("x")
y_index = model.dag.order.index("y")
coefficient = model.coef_matrix()[y_index, x_index]
print(round(coefficient.item(), 4))
print(round(mean_log_likelihood.item(), 4))
```

Verified output:

```text
1.5229
-2.8476
```

The difference from \(1.5\) is ordinary finite-sample variation. This example also shows a current API requirement that should be made safer: the tensor columns must be in exactly `model.observed_vars` order.

### 3.3 Exact finite-state output

```python
import torch
from sem2.default_configs import MLPConfig
from sem2.model import SEM2Discrete

torch.manual_seed(5)
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

probabilities, _ = model.sample(
    batch_size=3,
    set_nodes={"z": torch.tensor([[0], [1], [1]])},
)
print(model.observed_states.tolist())
print(list(probabilities.shape))
print(probabilities.sum(dim=1).tolist())
```

Verified output:

```text
[[0, 0], [0, 1], [1, 0], [1, 1]]
[3, 4]
[1.0, 1.0, 1.0]
```

Each row is an exact probability vector over the observed `(x,y)` states conditional on the supplied `z`, not a sampled state.

---

## 4. What needs to be updated?

The table separates direct **observations** from their likely **impact**. “Not found” means not found in the reviewed branch; some research code may exist outside the repository.

| Priority | Area and evidence | Impact / recommended update |
|---|---|---|
| **P0** | In `EnergyLoss._energy_score`, `m = preds.shape[1]` is the feature dimension, but `m/(m-1)` is used as the correction for the diagonal terms of the generated-sample pairwise matrix. | The U-statistic correction must use the number of generated draws, not the number of columns. It is undefined for one-dimensional data under the current code. Correct the estimator, define the intended V- or U-statistic explicitly, and add analytic unit tests in one and multiple dimensions. |
| **P0** | Continuous `EnergyLoss` assembles only `preds["Z"]`, `preds["X"]`, and `preds["Y"]`. The discrete wrapper assumes the first data column is binary `Z`, fixes the node named `"Z"`, and loops over `range(2)`. | The generic loss is actually a binary-IV-specific adapter. Derive observed variables from `model.observed_vars`, require an explicit table schema, and move IV-specific grouping into a named specialised operator. Validate empty groups and non-binary supports. |
| **P0** | `SEM2Trainer._fit_batch` unpacks `(Xs, targets)` but ignores `Xs`; it does not enforce column order, shape, dtype, device, or missing-data policy. `rng_seed` is configured but unused. | Define one public tabular fitting contract. A strong option is a tensor plus ordered column names (or a DataFrame adapter) mapped against `model.observed_vars`, with clear errors. Make seeding and device transfer explicit and test them. |
| **P0** | Report 2 describes package capabilities based on Sinkhorn divergence and a change-of-variables ICA likelihood. Neither operator exists under `sem2/`; the RDD/ICA implementation is not reproducible from the package. MMD is explicitly unimplemented. | Decide which report methods are product capabilities. Move the retained methods into versioned, tested operators with dependencies and runnable examples, or revise the reports/API claims to label them as external research scripts. Do not leave three different pictures of the fitting stack in code, reports, and notebooks. |
| **P0** | No tests under `tests/` refer to `SEM2Trainer`, `EnergyLoss`, `DiscreteEnergyLoss`, `SEM2Loss`, `ATE`, or the potential-outcome operators. | Add correctness tests for every fitting and causal operator before extending features: hand-computable finite-support distributions, intervention contrasts, gradient checks, loss convergence, and schema failures. The 190 passing tests currently establish model/parser/linear behaviour, not the generic training claims. |
| **P1** | `SEMType.MIXED` can be inferred, but there is no complete mixed-model class or fitting operator. | Either reject mixed graphs at construction with a precise message or implement a documented mixed inference strategy. Silent partial support is riskier than an explicit boundary. |
| **P1** | Exact discrete inference enumerates the Cartesian product of all discrete states. Context deduplication reduces repeated network evaluation but not the exponential state table. | Add state-space size checks and memory estimates; benchmark scaling; then consider variable elimination, factorised messages, sparse support, or controlled Monte Carlo/relaxation fallbacks. |
| **P1** | `MeanPotentialOutcome` and `ATE` are specialised and the repository has no general sweep/restart/tolerance-envelope API for partial identification. | Generalise estimands to named treatment/outcome nodes and values, then provide a reproducible multi-start \(\lambda\)-sweep workflow with observational-fit tolerances and diagnostics. Separate “a fitted extremum found” from “a sharp bound established.” |
| **P1** | SEM2 produces point estimates but no package-level standard errors, confidence sets, or bootstrap workflow. | For regular linear-Gaussian models, add Hessian/observed-information and delta-method support. Add sandwich and refit-bootstrap interfaces where justified. For nonregular or partially identified models, report confidence regions for identified sets rather than ordinary point-estimate intervals. |
| **P1** | Operator choice is manual even when `inferred_sem_type` contains useful structure. | Add an explicit `fit(...)` dispatcher or recommendation API: exact Gaussian likelihood when valid, declared non-Gaussian likelihood when available, and configured minimum-distance methods otherwise. Never silently fall back to a statistically different objective. |
| **P1** | There is no standard model-assessment layer. | Add held-out likelihood/discrepancy, empirical-versus-implied moment diagnostics, posterior-predictive-style plots, residual checks, convergence traces, restart sensitivity, and parameter/graph identifiability warnings. |
| **P1** | Causal intervention is an operational graph feature, but the API does not record identification assumptions or distinguish an interventional estimate from an observational fit. | Introduce causal-query objects or result metadata containing graph assumptions, estimand, treatment values, adjustment/identification argument where applicable, and sensitivity/partial-identification status. Avoid presenting `set_nodes` output as identified solely because it can be computed. |
| **P2** | [`README.md`](../../README.md) contains only `# sem2`. [`DESIGN.md`](../../DESIGN.md) describes a pre-refactor `sem2.py`, calls `model(...)` even though `forward` now raises, uses obsolete `model.layer`, imports from the empty package root, and refers to a nonexistent `DESIGN_refactor.md`. | Replace them with current installation, formula, architecture, data-schema, fitting, intervention, and limitation documentation. Keep one small tested example per supported model class. |
| **P2** | The notebooks use imports such as `from sem2 import SEM2`, `sem2.model.nn_models`, old object attributes, and undeclared packages such as `engression`. | Port only scientifically useful notebooks to the current API, clear stale output, pin their optional dependencies, and execute them in CI. Archive or delete superseded exploratory notebooks after human review. |
| **P2** | The package root exports nothing; `pyproject.toml` includes `pytest` as a runtime dependency and has no documented development extra, CI workflow, licence, changelog, or contribution guide. | Define and test the public API, move test tooling to a development extra, add minimal packaging metadata and CI, and publish compatibility/versioning expectations. |
| **P2** | `PathAnalysis` remains an explicit placeholder. Several configuration fields are unused or only partly honoured. | Either implement these surfaces with tests or remove/deprecate them until their semantics are designed. Public configuration should not imply functionality that is ignored. |

### 4.1 Documentation/research reconciliation

The reports are useful scientific records, but they currently mix three layers that should be separated:

1. **package capabilities present and tested in `sem2/`;**
2. **research implementations living in notebooks or one-off experiment scripts;**
3. **planned methods or theoretical proposals.**

For example, Report 2 accurately describes the current exact linear mean/covariance and Gaussian likelihood interfaces. In contrast, its Sinkhorn RDD and non-Gaussian ICA likelihood do not have corresponding core operators in this branch. The older notebooks appear to use an earlier module layout and external energy-loss code. A small capability matrix in the README and reports—*implemented, experimental, report-only, planned*—would prevent this drift.

### 4.2 Statistical and causal research gaps

Beyond engineering work, several research questions remain open:

- **Estimator theory:** consistency and rates for the chosen neural sieve, discrepancy, architecture growth, optimisation error, and latent-variable setting.
- **Identification:** conditions under which structural functions or causal queries are point identified, set identified, or unidentifiable from the declared observed margin.
- **Optimisation versus identification:** repeated starts and low observational discrepancy do not prove that an identified-set boundary is sharp; numerical error and structural non-identification need separate diagnostics.
- **Uncertainty:** sampling uncertainty must be propagated through model fitting and nonlinear intervention functionals. Partial identification requires uncertainty around a set, not only a sweep curve.
- **Robustness:** misspecified exogenous distributions, weak instruments, support violations, heavy tails, missingness, mixed measurement scales, and architecture sensitivity need controlled studies.
- **Scalability:** exact finite-state enumeration and pairwise sample discrepancies become expensive with dimension or batch size.
- **Causal validation:** the graph should be treated as an auditable scientific assumption. Where possible, test implications of the graph, compare alternative admissible graphs, and report sensitivity to unmeasured confounding.

---

## 5. Recommended development sequence

### Milestone 1 — trustworthy fitting core

1. Fix and unit-test continuous energy distance.
2. Define the ordered tabular schema and graph-generic observed-variable mapping.
3. Split generic loss composition from the binary-IV adapter.
4. Test trainer, intervention, finite-state loss, potential outcomes, and ATE end to end.
5. Make unsupported mixed graphs and objectives fail early with precise messages.

**Exit criterion:** the package can fit one linear-Gaussian, one nonlinear continuous, and one finite-discrete example through a documented API, and each result is covered by an independently checkable test.

### Milestone 2 — reconcile and reproduce the research

1. Rebuild the Report 2 ICA likelihood and Sinkhorn RDD methods as package operators or mark them report-only.
2. Turn each retained application into a deterministic script with data provenance, configuration, seeds, dependency lock, and numerical acceptance thresholds.
3. Add a benchmark matrix against OLS/lavaan/FastICA/RDD references without claiming general superiority.
4. Add the partial-identification sweep as a reusable experiment with multi-start and tolerance diagnostics.

**Exit criterion:** every numerical claim selected for continued support can be regenerated from a clean checkout.

### Milestone 3 — inference, scale, and public usability

1. Add standard errors/bootstrap or identified-set uncertainty appropriate to each regime.
2. Add model checking, convergence, restart, and sensitivity reports.
3. Improve discrete scaling and define mixed-model support.
4. Replace README/DESIGN documentation, update notebooks, stabilise exports, and add CI/release metadata.

**Exit criterion:** a new user can install SEM2, map a named table to a formula, fit a supported model, inspect diagnostics, and state exactly which observational and causal claims the result does and does not justify.

---

## 6. Verification performed for this review

- Inspected the default branch after a fresh remote fetch.
- Reviewed all Python modules under `sem2/`, the two test modules, experiment scripts, package metadata, current design guidance, and both PDFs under `docs/reports/`.
- Extracted and reviewed the attached rough mathematical draft.
- Executed the full test suite:

```text
190 passed, 51 warnings in 41.39s
```

The warnings arise primarily because test helpers intentionally instantiate linear formulas through the generic `SEM2` class, which recommends `SEM2Linear`.

- Executed the three illustrative examples above. The numerical outputs in this report are the actual outputs from that run.

## 7. Source notes

The review gives precedence to the inspected code for statements about current software behaviour. The reports and attached draft are treated as research context and are not assumed to describe shipped functionality unless a corresponding current implementation was found.

Repository research sources:

- [`MPhilSC_BothReport_Srivathsan_Amruth_Report_2_Starts_at_Page_69.pdf`](./MPhilSC_BothReport_Srivathsan_Amruth_Report_2_Starts_at_Page_69.pdf), especially Report 1’s model/discrete-estimation chapters and Report 2’s model-specific operator and application chapters.
- [`AAAI_SEM2.pdf`](./AAAI_SEM2.pdf), for the early architecture and neural structural-equation framing.
- Attached draft *Machine learning methods for structural equation models*, for the minimum-distance, energy-distance, and MMD setup. It was supplied with the review request and is not a repository artifact.

The report intentionally does not import causal claims from the attached draft, which the request correctly describes as not addressing causality.
