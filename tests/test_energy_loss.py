"""Small reference calculations and real-model energy/IV fitting checks."""
from itertools import product
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from sem2.data.preparation import prepare_data
from sem2.data.records import DataBatch, VariableInfo
from sem2.default_configs import MLPConfig, TrainingConfig
from sem2.model.distributions import DiscreteJointDistribution
from sem2.model.models.discrete import SEM2Discrete
from sem2.model.models.linear import SEM2Linear
from sem2.operators.discrete_energy_loss import DiscreteEnergyLoss
from sem2.operators.discrete_iv import DiscreteIVEnergyLoss
from sem2.operators.divergence import EnergyLoss
from sem2.operators.sem2_loss import SEM2Loss
from sem2.trainer.trainer import SEM2Trainer, fit
from sem2.utils import set_seed


@pytest.mark.parametrize('estimator', ['Ustat', 'Vstat'])
def test_continuous_score_matches_pairwise_reference_and_gradients(estimator):
    cfg = TrainingConfig(device='cpu', energy_score_beta=1.3, energy_pair_estimator=estimator)
    model = SEM2Linear('u ~ Normal(0,1); y <- u')
    loss = EnergyLoss(cfg, model)
    observations = DataBatch(torch.tensor([[0., 1.], [2., -1.]]), ['x', 'y'])
    values = torch.tensor([[1., 3.], [-2., 1.], [4., 2.]], requires_grad=True)
    prediction = DataBatch(values[:, [1, 0]], ['y', 'x'])
    result, _ = loss._energy_score(observations, prediction)
    cross = torch.stack([(x - y).norm().pow(cfg.energy_score_beta)
                         for x in observations.values for y in values]).mean()
    pairs = torch.stack([(values[i] - values[j]).norm().pow(cfg.energy_score_beta)
                         for i, j in product(range(3), repeat=2) if estimator == "Vstat" or i != j]).mean()
    reference = cross - pairs / 2
    torch.testing.assert_close(result.value, reference)
    torch.testing.assert_close(torch.autograd.grad(result.value, values)[0],
                               torch.autograd.grad(reference, values)[0])


@pytest.mark.parametrize(('estimator', 'model_pair', 'data_pair'), [
    ('Ustat', 1., 4/3), ('Vstat', 15/16, 8/9),
])
def test_binary_energy_matches_pair_averages(estimator, model_pair, data_pair):
    # States are distance 2 apart. The two model rows give P(y=1) of 1/4
    # and 1/2; the empirical frequency is 1/3. The cross distance is 11/12.
    y = VariableInfo('y', categorical=True, numeric_mapping={0: 0., 1: 2.})
    states = DataBatch(torch.tensor([[0], [1]]), [y])
    data = DataBatch(torch.tensor([[0], [0], [1]]), [y])
    p = torch.tensor([.25, .5], requires_grad=True)
    prediction = DiscreteJointDistribution(states, torch.stack((1-p, p), dim=1))
    loss = DiscreteEnergyLoss(TrainingConfig(device='cpu'), states, pair_estimator=estimator)
    score, _ = loss.energy_score(data, prediction)
    distance, diagnostics = loss.energy_dist(data, prediction)
    assert score.value.item() == pytest.approx(11/12 - model_pair/2)
    assert distance.value.item() == pytest.approx(11/12 - model_pair/2 - data_pair/2, abs=1e-7)
    assert diagnostics['total_variation'].item() == pytest.approx(1/24, abs=1e-7)
    # Differentiate the binary pair averages w.r.t. the two probabilities.
    gradient, = torch.autograd.grad(distance.value, p)
    expected = [1/3, -1/6] if estimator == 'Ustat' else [1/12, 1/12]
    torch.testing.assert_close(gradient, torch.tensor(expected))


def test_discrete_energy_uses_scores_not_category_codes():
    cfg = TrainingConfig(device='cpu')

    def scored_loss(codes):
        y = VariableInfo('y', categorical=True, numeric_mapping=dict(zip(codes, [0., 2., 5.])))
        states = DataBatch(torch.tensor(codes)[:, None], [y])
        data = DataBatch(states.values[[0, 1, 1, 2]], [y])
        prediction = DiscreteJointDistribution(states, torch.tensor([[.1, .2, .7], [.6, .3, .1]]))
        return DiscreteEnergyLoss(cfg, states).energy_dist(data, prediction)[0].value

    torch.testing.assert_close(scored_loss([0, 1, 2]), scored_loss([2, 0, 1]))
    with pytest.raises(ValueError, match='numeric_mapping'):
        DiscreteEnergyLoss(cfg, DataBatch(torch.tensor([[0], [1]]), [VariableInfo('y', categorical=True)]))


def test_discrete_precomputation_tracks_named_margins_and_scores():
    cfg = TrainingConfig(device='cpu', sample_bs=4)
    model = SEM2Discrete('u ~ Normal(0,1); x <- d(u; levels=2); y <- d(x; levels=2)',
                         node_cfgs={'x': MLPConfig(hidden_dims=[4]), 'y': MLPConfig(hidden_dims=[4])})
    variables = [VariableInfo(name, categorical=True, numeric_mapping={0: 0., 1: 1.})
                 for name in ['x', 'y']]
    observations = DataBatch(torch.tensor([[0, 0], [0, 1], [1, 1]]), variables)
    loss = EnergyLoss(cfg, model)
    set_seed(3)
    original, _ = loss.evaluate(observations)
    set_seed(3)
    reordered, _ = loss.evaluate(observations.subset(['y', 'x']))
    torch.testing.assert_close(original.value, reordered.value)
    margin = observations.subset('y')
    loss.evaluate(margin)
    matrix = loss.discrete_loss.D_mat
    with patch('torch.cdist', wraps=torch.cdist) as distances:
        loss.evaluate(margin)
        assert distances.call_count == 0
        # An in-place metadata edit must not leave the old scores cached.
        margin.variables[0].numeric_mapping[1] = 3.
        loss.evaluate(margin)
        assert distances.call_count == 1
    torch.testing.assert_close(loss.discrete_loss.D_mat, matrix * 3)
    assert loss.discrete_loss.states.names == ('y',)


def test_fractional_energy_has_finite_gradients_at_coincident_points():
    model = SEM2Linear('u ~ Normal(0,1); y <- u')
    loss = EnergyLoss(TrainingConfig(device='cpu'), model, beta=0.5)
    observations = DataBatch(torch.tensor([[0.], [1.]]), ['y'])
    draws = torch.tensor([[0.], [0.], [1.]], requires_grad=True)
    value, _ = loss._energy_score(observations, DataBatch(draws, ['y']))
    value.value.backward()
    assert torch.isfinite(draws.grad).all()


@pytest.mark.parametrize('settings', [{'beta': 2.}, {'pair_estimator': 'unknown'}])
def test_invalid_energy_settings_rejected_at_construction(settings):
    model = SEM2Linear('u ~ Normal(0,1); y <- u')
    with pytest.raises(ValueError):
        EnergyLoss(TrainingConfig(device='cpu'), model, **settings)


@pytest.fixture
def iv_problem():
    levels = 3
    set_seed(5)
    model = SEM2Discrete(
        'u ~ Normal(0,1); instrument ~ Categorical(probs=[1,1,1]); '
        'treatment <- d(instrument,u; levels=2); outcome <- d(treatment,u; levels=2)',
        node_cfgs={'treatment': MLPConfig(hidden_dims=[4]), 'outcome': MLPConfig(hidden_dims=[4])},
    )
    rows = [(level, i % 2, (i + level) % 2) for level in range(levels) for i in range(level + 2)]
    frame = pd.DataFrame(rows, columns=['instrument', 'treatment', 'outcome'])
    variables = [VariableInfo(name, categorical=True, numeric_mapping={0: 0., 1: 1.})
                 for name in ['treatment', 'outcome']]
    cfg = TrainingConfig(device='cpu', batch_size=len(frame), sample_bs=4, epochs=2, rng_seed=19)
    return model, frame, variables, cfg


def test_iv_loss_weights_groups_by_observed_frequency(iv_problem):
    model, frame, variables, cfg = iv_problem
    loss = DiscreteIVEnergyLoss(cfg, model, 'instrument')
    batch, _ = prepare_data(model, frame, variables=variables, device='cpu')
    result, metrics = loss.evaluate(batch)
    expected = 0
    for level in range(3):
        prefix = f'instrument={level}/'
        assert metrics[prefix + 'count'] == level + 2
        energy = (metrics[prefix + 'estimated-true distance']
                  - metrics[prefix + 'estimated-estimated distance'] / 2
                  - metrics[prefix + 'true-true distance'] / 2)
        expected = expected + (level + 2) / len(frame) * energy
    torch.testing.assert_close(result.value, expected)
    torch.testing.assert_close(metrics['max_cell_error'], torch.stack([
        metrics[f'instrument={level}/max_cell_error'] for level in range(3)
    ]).max())


@pytest.mark.parametrize(('estimator', 'retained', 'eligible'), [
    ('Ustat', 0, False), ('Ustat', 1, False), ('Ustat', 2, True),
    ('Vstat', 0, False), ('Vstat', 1, True),
])
def test_iv_minimum_complete_cases_per_level(iv_problem, estimator, retained, eligible):
    model, frame, variables, cfg = iv_problem
    last_group = frame.index[frame.instrument == 2]
    frame.loc[last_group[retained:], 'outcome'] = float('nan')
    loss = DiscreteIVEnergyLoss(cfg, model, 'instrument', pair_estimator=estimator)
    objective = SEM2Loss([(loss, 1.)])
    if eligible:
        fit(model, frame, objective, cfg, variables=variables)
        assert all(torch.isfinite(p).all() for p in model.parameters())
    else:
        with pytest.raises(ValueError, match='counts='):
            SEM2Trainer(cfg, model, frame, objective, variables=variables)


def test_composed_iv_objective_requires_full_batch(iv_problem):
    model, frame, variables, cfg = iv_problem
    loss = DiscreteIVEnergyLoss(cfg, model, 'instrument')
    nested = SEM2Loss([(SEM2Loss([(loss, 1.)]), 1.)])
    with pytest.raises(ValueError, match='full-data batch'):
        SEM2Trainer(cfg.model_copy(update={'batch_size': 2}), model, frame, nested, variables=variables)
