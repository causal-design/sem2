"""Canonical fitting checks on a real Gaussian model, not IV validation."""
from copy import deepcopy
import logging

import numpy as np
import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader, TensorDataset

from experiments.trainer import run_example

from sem2.data.records import DataBatch
from sem2.default_configs import TrainingConfig
from sem2.model.models.linear import SEM2Linear
from sem2.operators.linear import LogLikelihood
from sem2.operators.divergence import EnergyLoss, WeightedFrobeniusDistance
from sem2.operators.sem2_loss import SEM2Loss
from sem2.trainer.trainer import SEM2Trainer, fit
from sem2.utils import set_seed


def test_fit_and_trainer_agree_in_place_with_execution_time_seed_and_progress(caplog):
    set_seed(7)
    truth = SEM2Linear('u,v ~ Normal(0,1); x <- 1*u; y <- 2*x + 1*v')
    observations, _ = truth.sample(48)
    frame = pd.DataFrame(observations.select(['x', 'y']).detach().numpy(), columns=['x', 'y'])
    original = frame.copy(deep=True)
    model = SEM2Linear('u,v ~ Normal(0,1); x <- 1*u; y <- x + 1*v')
    other = deepcopy(model)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    graph_names = tuple(model.node_specs)
    cfg = TrainingConfig(device='cpu', epochs=3, batch_size=16, lr=0.02, rng_seed=19)
    objective = SEM2Loss([(LogLikelihood(model), -1.)])
    trainer = SEM2Trainer(cfg, model, frame, objective)
    assert isinstance(trainer.loader, DataLoader)
    assert isinstance(trainer.loader.dataset, TensorDataset)
    assert isinstance(trainer.variables, list)
    batch = DataBatch(trainer.loader.dataset.tensors[0], trainer.variables)
    initial_loss = objective.evaluate(batch)[0].value.detach()
    torch.rand(11)  # Construction is not the seeding boundary.
    with caplog.at_level(logging.INFO, logger='sem2.trainer.trainer'):
        assert trainer.train() is model
    torch.rand(23)
    assert fit(other, frame, SEM2Loss([(LogLikelihood(other), -1.)]), cfg) is other
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[name])
    assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert tuple(model.node_specs) == graph_names
    assert objective.evaluate(batch)[0].value < initial_loss
    pd.testing.assert_frame_equal(frame, original)
    records = [record for record in caplog.records if record.name == 'sem2.trainer.trainer']
    assert len(records) == cfg.epochs
    assert all('last-batch loss=' in record.message for record in records)
    previous_optimizer = trainer.optim
    trainer.train()
    assert trainer.optim is not previous_optimizer


def test_objective_updates_avoid_scalar_validation_and_solver_error_checks():
    model = SEM2Linear('u,v ~ Normal(0,1); x <- u; y <- x + v')
    model.node_dict['u'].dist.loc.requires_grad_(True)
    cfg = TrainingConfig(device='cpu', sample_bs=8)
    batch = DataBatch(torch.tensor([[0., 1.], [1., 2.], [2., -1.]]), ['x', 'y'])
    objectives = [LogLikelihood(model), WeightedFrobeniusDistance(model), EnergyLoss(cfg, model)]
    # Setup validation is outside the measured evaluation/backward path.
    with profile(activities=[ProfilerActivity.CPU]) as execution:
        for objective in objectives:
            objective.evaluate(batch)[0].value.backward()
    events = {event.key for event in execution.key_averages()}
    assert 'aten::_local_scalar_dense' not in events
    assert 'aten::_linalg_check_errors' not in events
    # Normal's sampling implementation must still detach exogenous draws.
    assert not model.node_dict['u'].get_value().requires_grad


def test_default_fit_does_not_reseed_the_shared_stream():
    set_seed(23)
    model = SEM2Linear('u ~ Normal(0,1); x <- u')
    frame = pd.DataFrame({'x': [-1., 0., 1.]})
    cfg = TrainingConfig(device='cpu', epochs=1, batch_size=3)
    assert cfg.rng_seed is None
    objective = SEM2Loss([(LogLikelihood(model), -1.)])
    trainer = SEM2Trainer(cfg, model, frame, objective)
    rng = torch.random.get_rng_state().clone()
    # The analytic objective draws no noise; only DataLoader advances the stream.
    for _ in trainer.loader:
        pass
    expected_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(rng)
    assert fit(model, frame, objective, cfg) is model
    assert torch.equal(torch.random.get_rng_state(), expected_rng)


def test_gaussian_example_matches_independent_covariance_fit_and_final_diagnostics():
    coefficients, moments, diagnostics = run_example(seed=7)
    np.testing.assert_allclose(coefficients["true"], [0., 2.], rtol=0., atol=0.)
    assert not np.allclose(coefficients["initial"], coefficients["true"])
    np.testing.assert_allclose(coefficients.loc["intercept (fixed)"], 0., rtol=0., atol=0.)
    np.testing.assert_allclose(coefficients["fitted"], coefficients["reference"], rtol=0., atol=1e-4)
    np.testing.assert_allclose(moments["true"], [0., 0., 1., 2., 5.], rtol=0., atol=0.)
    slope = coefficients.loc["x -> y", "fitted"]
    np.testing.assert_allclose(moments["fitted"], [0., 0., 1., slope, slope**2 + 1.], atol=1e-5)

    # Independently expand the weighted covariance loss and find all stationary points.
    # W=[[1, 1/2], [1/2, 1]] counts the off-diagonal residual only once.
    b = np.polynomial.Polynomial([0., 1.])
    sample = moments["sample"]
    loss = ((sample["var(x)"] - 1.)**2 + (sample["cov(x,y)"] - b)**2
            + (sample["var(y)"] - 1. - b**2)**2)
    roots = loss.deriv().roots()
    reference_loss = min(loss(root.real) for root in roots if abs(root.imag) < 1e-8)
    np.testing.assert_allclose(diagnostics["independent minimum loss"], reference_loss, atol=1e-10)
    np.testing.assert_allclose(
        [diagnostics["initial weighted squared Frobenius loss"],
         diagnostics["fitted weighted squared Frobenius loss"]],
        [loss(coefficients.loc["x -> y", "initial"]), loss(slope)], rtol=1e-5, atol=1e-6,
    )
    assert diagnostics["fitted weighted squared Frobenius loss"] < diagnostics["initial weighted squared Frobenius loss"]
    np.testing.assert_allclose(diagnostics["fitted weighted squared Frobenius loss"], reference_loss, atol=1e-6)
    assert abs(loss.deriv()(slope)) < 1e-4
    assert diagnostics["maximum absolute final gradient"] < 1e-4
    assert diagnostics["absolute fitted-minus-reference slope"] < 1e-4
