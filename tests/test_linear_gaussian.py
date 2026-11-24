import pytest
import torch

from sem2.data.records import DataBatch
from sem2.model.models.linear_gaussian import SEM2LinearGaussian
from sem2.operators.linear import LogLikelihood


def test_joint_gaussian_analytical_moments_likelihood_and_intervention():
    model = SEM2LinearGaussian("E ~ Normal(2,3); X <- 0 + 1*E; Y <- 0 + 2*X + 1*F; E <-> F")
    with torch.no_grad():
        model.edge_weights.fill_(1)  # correlation 1/2
    mean = torch.tensor([2., 4.])
    cov = torch.tensor([[9., 19.5], [19.5, 43.]])
    torch.testing.assert_close(model.cov_matrix(["X","Y"]), cov)
    data = DataBatch(torch.tensor([[2.,4.], [1.,3.], [3.,6.]]), ["X","Y"])
    value = LogLikelihood(model).evaluate(data)[0].value
    torch.testing.assert_close(value, torch.distributions.MultivariateNormal(mean,cov).log_prob(data.values).mean())
    value.backward()
    for p in (model.edge_weights, model.log_scales):
        assert p.grad.isfinite().all() and p.grad.abs().sum() > 0
    torch.manual_seed(17)
    draws, _ = model.sample(50000)
    assert draws.names == ("X","Y")
    torch.testing.assert_close(draws.values.mean(0), mean, atol=.07, rtol=0)
    torch.testing.assert_close(torch.cov(draws.values.T), cov, atol=.1, rtol=.02)
    torch.manual_seed(17)
    do, _ = model.sample(50000, do_intervention=DataBatch(torch.zeros(50000,1), ["X"]))
    torch.testing.assert_close(do.select("Y"), draws.select("Y") - 2*draws.select("X"),
                               atol=2e-6, rtol=1e-5)


def test_sparse_signed_cycle_covariance_and_fixed_marginals():
    model = SEM2LinearGaussian("a,b,c,d ~ Normal(0,1); a <-> b <-> c <-> d <-> a; x <- a")
    target = torch.eye(4)
    for (i,j),r in zip(model._edges.T.tolist(), [.6,-.6,.6,.6]):
        target[i,j] = target[j,i] = r
    minimum = torch.linalg.eigvalsh(target)[0]
    i,j = model._edges
    with torch.no_grad():
        model.edge_weights.copy_(target[i,j]/minimum)
    torch.testing.assert_close(model.noise_covariance(), target)
    assert model.log_scales.numel() == 0


def test_fit_recovers_free_variance_and_covariance():
    model = SEM2LinearGaussian("a ~ Normal(0,1); X <- 0 + 1*a; Y <- 0 + 1*b; a <-> b")
    # Four centered rows with exact ML variances 1,4 and covariance 1.
    x = torch.tensor([-1.,-1.,1.,1.])
    y = x + 3**.5 * torch.tensor([-1.,1.,-1.,1.])
    data = DataBatch(torch.stack([x,y],dim=1), ["X","Y"])
    objective = LogLikelihood(model)
    opt = torch.optim.Adam(model.parameters(), lr=.08)
    for _ in range(160):
        opt.zero_grad()
        (-objective.evaluate(data)[0].value).backward()
        opt.step()
    torch.testing.assert_close(model.noise_covariance(), torch.tensor([[1.,1.],[1.,4.]]), atol=.01, rtol=0)


@pytest.mark.parametrize("formula", ["E ~ Uniform(0,1); X <- E", "X <- n(E)", "X <- d(E; levels=2)"])
def test_gaussian_family_restrictions(formula):
    with pytest.raises(ValueError, match="linear equations and Normal"):
        SEM2LinearGaussian(formula)
