"""Integration test: the Bollen PoliticalDemocracy model fit with SEM2Linear.

This rebuilds the example from notebooks/linear_sem.ipynb on the current API. It
fits the full latent-variable SEM to the real PoliticalDemocracy data using
SEM2Linear's *closed-form* model-implied covariance (no sampling) via the
Gaussian LogLikelihood operator, and checks that the recovered structural
coefficients match the published lavaan ML estimates.

Reference (lavaan ML, structural regressions among latents):
    dem60 ~ ind60 : 1.483
    dem65 ~ ind60 : 0.572
    dem65 ~ dem60 : 0.837
"""
import os
import pytest
import torch

torch.manual_seed(0)
torch.set_num_threads(1)

from sem2.model import SEM2Linear, SEMType
from sem2.operators.linear import LogLikelihood

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "PoliticalDemocracy.csv")
LAVAAN = {"dem60~ind60": 1.483, "dem65~ind60": 0.572, "dem65~dem60": 0.837}

# Latents ind60/dem60/dem65; loadings learned; residual covariances modelled via
# shared exogenous noise nodes (e15, e24, ...). All exogenous noise is Normal, so
# the model is inferred LINEAR_GAUSSIAN.
FORMULA = (
    "ex1,ex2,ex3,ey1,ey2,ey3,ey4,ey5,ey6,ey7,ey8,edem60,edem65 ~ Normal(0,1);"
    "e15,e24,e26,e37,e48,e68 ~ Normal(0,1);"
    "ind60 ~ Normal(0,1); dem60 <- ind60 + edem60; dem65 <- ind60 + dem60 + edem65;"
    "x1 <- ind60 + ex1; x2 <- ind60 + ex2; x3 <- ind60 + ex3;"
    "y1 <- dem60 + ey1 + e15; y2 <- dem60 + ey2 + e24 + e26;"
    "y3 <- dem60 + ey3 + e37; y4 <- dem60 + ey4 + e24 + e48;"
    "y5 <- dem65 + ey5 + e15; y6 <- dem65 + ey6 + e26 + e68;"
    "y7 <- dem65 + ey7 + e37; y8 <- dem65 + ey8 + e48 + e68;"
    "unobs(ex1,ex2,ex3,ey1,ey2,ey3,ey4,ey5,ey6,ey7,ey8,edem60,edem65,"
    "e15,e24,e26,e37,e48,e68,ind60,dem60,dem65)"
)
OBS = ["x1", "x2", "x3", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8"]


def _load_centered():
    import csv
    with open(DATA) as f:
        reader = csv.reader(f)
        header = [h.strip().strip('"') for h in next(reader)]
        rows = [[float(v) for v in r] for r in reader]
    data = torch.tensor(rows, dtype=torch.float32)
    data = data - data.mean(0, keepdim=True)   # centre: fit covariance structure
    cols = {name: i for i, name in enumerate(header)}
    return data, cols


def _fit(iters=2000, lr=0.05):
    torch.manual_seed(0)
    data, cols = _load_centered()
    x = data[:, [cols[v] for v in OBS]]
    m = SEM2Linear(FORMULA)
    ll = LogLikelihood(m, centered=True)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        val, _ = ll.evaluate(x)
        (-val).backward()        # maximise log-likelihood == lavaan ML discrepancy
        opt.step()
    return m


def _structural_coefs(m):
    """Recover the three structural regressions in lavaan's scaling.

    Loadings are learned and latent variances are fixed to 1, so each latent's
    scale is set by the model rather than by a unit first loading (as in lavaan).
    Converting a regression coefficient beta to lavaan's first-loading-=-1 scale
    multiplies by lambda_to / lambda_from (the model's reference loadings).
    """
    B = m.coef_matrix().detach()
    ix = m._node_index
    lam = {
        "ind60": B[ix["x1"], ix["ind60"]],
        "dem60": B[ix["y1"], ix["dem60"]],
        "dem65": B[ix["y5"], ix["dem65"]],
    }
    return {
        "dem60~ind60": (B[ix["dem60"], ix["ind60"]] * lam["dem60"] / lam["ind60"]).item(),
        "dem65~ind60": (B[ix["dem65"], ix["ind60"]] * lam["dem65"] / lam["ind60"]).item(),
        "dem65~dem60": (B[ix["dem65"], ix["dem60"]] * lam["dem65"] / lam["dem60"]).item(),
    }


class TestPoliticalDemocracy:

    def test_model_is_linear_gaussian(self):
        m = SEM2Linear(FORMULA)
        assert m.inferred_sem_type == SEMType.LINEAR_GAUSSIAN
        assert m.observed_vars == OBS

    def test_cov_matrix_symmetric_and_pd(self):
        m = SEM2Linear(FORMULA)
        sigma = m.cov_matrix(m.observed_vars).detach()
        assert sigma.shape == (len(OBS), len(OBS))
        assert torch.allclose(sigma, sigma.T, atol=1e-5)
        # Cholesky succeeds iff positive definite.
        torch.linalg.cholesky(sigma)

    def test_recovers_lavaan_coefficients(self):
        m = _fit()
        coefs = _structural_coefs(m)
        for name, target in LAVAAN.items():
            assert coefs[name] == pytest.approx(target, abs=0.05), \
                f"{name}: got {coefs[name]:.3f}, expected ~{target}"
