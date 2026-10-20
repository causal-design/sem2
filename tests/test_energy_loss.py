"""Small reference calculations and real-model energy/IV fitting checks."""
from dataclasses import replace
from itertools import product
from unittest.mock import patch

import pandas as pd
import pytest
import torch

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


@pytest.mark.parametrize('unbiased', [True, False])
def test_continuous_score_matches_pairwise_reference_and_gradients(unbiased):
    cfg = TrainingConfig(device='cpu', energy_score_beta=1.3, energy_score_unbiased=unbiased)
    model = SEM2Linear('u ~ Normal(0,1); y <- u')
    loss = EnergyLoss(cfg, model)
    observations = DataBatch(torch.tensor([[0., 1.], [2., -1.]]), ['x', 'y'])
    values = torch.tensor([[1., 3.], [-2., 1.], [4., 2.]], requires_grad=True)
    prediction = DataBatch(values[:, [1, 0]], ['y', 'x'])
    result, _ = loss._energy_score(observations, prediction, beta=loss.beta)
    cross = torch.stack([(x - y).norm().pow(cfg.energy_score_beta)
                         for x in observations.values for y in values]).mean()
    pairs = torch.stack([(values[i] - values[j]).norm().pow(cfg.energy_score_beta)
                         for i, j in product(range(3), repeat=2) if not unbiased or i != j]).mean()
    reference = cross - pairs / 2
    torch.testing.assert_close(result.value, reference)
    torch.testing.assert_close(torch.autograd.grad(result.value, values)[0],
                               torch.autograd.grad(reference, values)[0])


def test_discrete_score_correction_and_probability_diagnostics():
    cfg = TrainingConfig(device='cpu')
    variable = VariableInfo('y', categorical=True, numeric_mapping={0: 0., 1: 2., 2: 5.})
    states = DataBatch(torch.tensor([[0], [1], [2]]), [variable])
    loss = DiscreteEnergyLoss(cfg, states)
    observations = DataBatch(torch.tensor([[0], [1], [1], [2]]), [variable])
    logits = torch.tensor([[0., 1., 2.], [2., 0., 1.]], requires_grad=True)
    probabilities = logits.softmax(-1)
    prediction = DiscreteJointDistribution(states, probabilities)
    result, details = loss.energy_dist(observations, prediction, beta=1.)
    scores, data = [0., 2., 5.], [0., 2., 2., 5.]
    # Explicit sums over independent exogenous rows, not just off-diagonal states.
    cross = sum(probabilities[i, s] * abs(scores[s] - x)
                for i, s, x in product(range(2), range(3), data)) / 8
    pair = sum(probabilities[i, s] * probabilities[j, t] * abs(scores[s] - scores[t])
               for i, j, s, t in product(range(2), range(2), range(3), range(3)) if i != j) / 2
    data_pair = sum(abs(x - y) for i, x in enumerate(data) for j, y in enumerate(data) if i != j) / 12
    reference = cross - pair / 2 - data_pair / 2
    torch.testing.assert_close(result.value, reference)
    torch.testing.assert_close(torch.autograd.grad(result.value, logits, retain_graph=True)[0],
                               torch.autograd.grad(reference, logits)[0])
    error = (probabilities.mean(0) - torch.tensor([0.25, 0.5, 0.25])).abs()
    torch.testing.assert_close(details['total_variation'], error.sum() / 2)
    torch.testing.assert_close(details['max_cell_error'], error.max())
    score, _ = loss.energy_score(observations, prediction, beta=1.)
    torch.testing.assert_close(score.value, result.value + data_pair / 2)

    # Recode identities and retain their scores: the objective is unchanged.
    recoded = replace(variable, numeric_mapping={2: 0., 0: 2., 1: 5.})
    other = DiscreteJointDistribution(DataBatch(torch.tensor([[2], [0], [1]]), [recoded]), probabilities)
    recoded_data = DataBatch(torch.tensor([[2], [0], [0], [1]]), [recoded])
    recoded_loss = DiscreteEnergyLoss(cfg, other.states)
    torch.testing.assert_close(recoded_loss.energy_dist(recoded_data, other)[0].value, result.value)
    with pytest.raises(ValueError, match='numeric_mapping'):
        DiscreteEnergyLoss(cfg, DataBatch(states.values, [replace(variable, numeric_mapping=None)]))


def test_discrete_distances_are_precomputed_with_temporary_beta_overrides():
    cfg = TrainingConfig(device='cpu', energy_score_beta=1.3)
    states = DataBatch(torch.tensor([[3.], [0.], [1.]]), ['y'])
    observations = DataBatch(torch.tensor([[0.], [1.], [1.], [3.]]), ['y'])
    logits = torch.tensor([[0., 1., 2.], [2., 0., 1.]], requires_grad=True)
    references = [DiscreteEnergyLoss(cfg, states, beta=beta) for beta in [1.3, 0.5]]
    with patch('torch.cdist', wraps=torch.cdist) as distances:
        loss = DiscreteEnergyLoss(cfg, states)
        matrix = loss.D_mat
        assert distances.call_count == 1  # Constructed before any evaluation.
        for beta, reference in [(None, references[0]), (0.5, references[1]), (None, references[0])]:
            prediction = DiscreteJointDistribution(states, logits.softmax(-1))
            actual, _ = loss.energy_dist(observations, prediction, beta=beta)
            expected, _ = reference.energy_dist(observations, prediction)
            torch.testing.assert_close(actual.value, expected.value)
            torch.testing.assert_close(torch.autograd.grad(actual.value, logits, retain_graph=True)[0],
                                       torch.autograd.grad(expected.value, logits)[0])
            assert loss.D_mat is matrix
        assert distances.call_count == 1


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


def test_config_overrides_coincident_gradients_and_continuous_fit():
    set_seed(7)
    model = SEM2Linear('u ~ Normal(0,1); y <- u')
    cfg = TrainingConfig(device='cpu', energy_score_beta=0.5, energy_score_unbiased=False,
                         sample_bs=8, batch_size=4, epochs=2, rng_seed=11)
    loss = EnergyLoss(cfg, model)
    assert loss.beta == 0.5 and loss.sample_bs == 8 and not loss.unbiased
    overridden = EnergyLoss(cfg, model, beta=1., sample_bs=12, unbiased=True)
    assert overridden.beta == 1. and overridden.sample_bs == 12 and overridden.unbiased
    observations = DataBatch(torch.tensor([[0.], [1.], [2.], [3.]]), ['y'])
    draws = torch.tensor([[0.], [0.], [1.]], requires_grad=True)
    value = loss._energy_score(observations, DataBatch(draws, ['y']), beta=loss.beta)[0].value
    value.backward()
    assert torch.isfinite(draws.grad).all()
    with pytest.raises(ValueError, match='beta'):
        loss.evaluate(observations, beta=2.)

    set_seed(23)
    changed = SEM2Loss([(SEM2Loss([(loss, 1.)]), 1.)]).evaluate(observations, beta=1., sample_bs=12)[0]
    set_seed(23)
    expected = EnergyLoss(cfg, model, beta=1., sample_bs=12).evaluate(observations)[0]
    torch.testing.assert_close(changed.value, expected.value)
    assert loss.beta == 0.5 and loss.sample_bs == 8
    before = [p.detach().clone() for p in model.parameters()]
    assert fit(model, pd.DataFrame({'y': [0., 1., 2., 3.]}), loss, cfg) is model
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), before))


@pytest.mark.parametrize('levels', [2, 3])
def test_iv_full_data_fit_weights_and_eligibility(levels):
    set_seed(5)
    model = SEM2Discrete(
        f'u ~ Normal(0,1); instrument ~ Categorical(probs={str([1.] * levels)}); '
        'treatment <- d(instrument,u; levels=2); outcome <- d(treatment,u; levels=2)',
        node_cfgs={'treatment': MLPConfig(hidden_dims=[4]), 'outcome': MLPConfig(hidden_dims=[4])},
    )
    rows = [(level, i % 2, (i + level) % 2) for level in range(levels) for i in range(level + 2)]
    frame = pd.DataFrame(rows, columns=['instrument', 'treatment', 'outcome'])
    variables = [VariableInfo(name, categorical=True, numeric_mapping={0: 0., 1: 1.})
                 for name in ['treatment', 'outcome']]
    cfg = TrainingConfig(device='cpu', batch_size=len(frame), sample_bs=4, epochs=2, rng_seed=19)
    loss = DiscreteIVEnergyLoss(cfg, model, 'instrument')
    objective = SEM2Loss([(SEM2Loss([(loss, 1.)]), 1.)])
    trainer = SEM2Trainer(cfg, model, frame, objective, variables=variables)
    batch = DataBatch(trainer.loader.dataset.tensors[0], trainer.variables)
    with patch('torch.cdist', wraps=torch.cdist) as distances:
        result, metrics = loss.evaluate(batch)
        assert distances.call_count == 1  # Shared by every instrument group.
    matrix = loss.discrete_loss.D_mat
    expected = 0
    for level in range(levels):
        prefix = f'instrument={level}/'
        assert metrics[prefix + 'count'] == level + 2
        energy = (metrics[prefix + 'estimated-true distance']
                  - metrics[prefix + 'estimated-estimated distance'] / 2
                  - metrics[prefix + 'true-true distance'] / 2)
        expected = expected + (level + 2) / len(frame) * energy
    torch.testing.assert_close(result.value, expected)
    torch.testing.assert_close(metrics['max_cell_error'], torch.stack([
        metrics[f'instrument={level}/max_cell_error'] for level in range(levels)
    ]).max())
    before = [p.detach().clone() for p in model.parameters()]
    with patch('torch.cdist', wraps=torch.cdist) as distances:
        assert fit(model, frame, objective, cfg, variables=variables) is model
        assert distances.call_count == 0
    assert loss.discrete_loss.D_mat is matrix
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), before))
    with pytest.raises(ValueError, match='full-data batch'):
        SEM2Trainer(cfg.model_copy(update={'batch_size': 2}), model, frame, objective, variables=variables)
    # Eligibility uses complete cases, and does not silently drop an empty level.
    missing = frame.copy()
    missing.loc[missing.instrument == levels - 1, 'outcome'] = float('nan')
    with pytest.raises(ValueError, match='counts='):
        SEM2Trainer(cfg, model, missing, objective, variables=variables)
    singleton = batch[batch.select('instrument')[:, 0] == 0][:1]
    with pytest.raises(ValueError, match='counts='):
        loss.evaluate(singleton)
