"""Canonical fitting checks on a real Gaussian model, not IV validation."""
from copy import deepcopy
import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from sem2.data.records import DataBatch
from sem2.default_configs import TrainingConfig
from sem2.model.models.linear import SEM2Linear
from sem2.operators.linear import LogLikelihood
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
