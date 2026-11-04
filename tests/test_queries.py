"""Ordinary query contracts and independent known-model references."""
from itertools import product
from unittest.mock import patch

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from sem2.data.records import DataBatch, VariableInfo
from sem2.default_configs import MLPConfig, TrainingConfig
from sem2.model.models.base import SEM2
from sem2.model.models.discrete import SEM2Discrete
from sem2.model.models.linear import SEM2Linear
from sem2.operators.divergence import EnergyLoss
from sem2.operators.potential_outcomes import ATE, MeanPotentialOutcome, PotentialOutcomeSample
from sem2.operators.sem2_loss import SEM2Loss
from sem2.utils import set_seed


def test_continuous_joint_contrast_gradients_and_composition():
    model = SEM2Linear('u ~ Normal(0,1); exposure <- 1*u + 0; '
                       'mediator <- 2*exposure + 0; response <- mediator + 0')
    weight = model.node_dict['response'].linear.weight
    with torch.no_grad():
        weight.fill_(3.)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    query = ATE(model, outcome='response', alternative={'exposure': 2}, reference={'exposure': 0})
    result, info = query.evaluate(sample_bs=7)
    torch.testing.assert_close(result.value, torch.tensor(12.))
    torch.testing.assert_close(torch.autograd.grad(result.value, weight)[0], torch.tensor([[4.]]))
    assert info['alternative'] == {'exposure': 2} and info['reference'] == {'exposure': 0}
    assert info['sample_bs'] == 7 and info['seed'] is None
    torch.testing.assert_close(info['alternative_mean'] - info['reference_mean'], result.value)
    # Joint assignment replaces the mediator, rather than using its old value.
    joint = ATE(model, outcome='response', alternative={'exposure': 2, 'mediator': 1},
                reference={'exposure': 0, 'mediator': 0})
    torch.testing.assert_close(joint.evaluate(sample_bs=3)[0].value, torch.tensor(3.))
    direct = PotentialOutcomeSample(model, 'response').evaluate(
        3, do_intervention=DataBatch(torch.full((3, 1), 2.), ['exposure']))[0]
    torch.testing.assert_close(direct.values, torch.full((3, 1), 12.))
    loss = SEM2Loss([(EnergyLoss(TrainingConfig(device='cpu'), model), 1.), (query, 0.2)])
    loss.evaluate(DataBatch(torch.tensor([[0.], [1.], [2.]]), ['response']), sample_bs=5)[0].value.backward()
    assert torch.isfinite(weight.grad).all()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_natural_regime_seed_and_temporary_overrides():
    model = SEM2Linear('noise ~ Normal(1,1); exposure <- 1*noise + 0; response <- 2*exposure + 0')
    mean = MeanPotentialOutcome(model, outcome='response', intervention={}, sample_bs=9)
    set_seed(51)
    first = mean.evaluate()[0].value
    second = mean.evaluate()[0].value
    set_seed(51)
    torch.testing.assert_close(first, 2 * model.sample(9)[0].select('noise').mean())
    torch.testing.assert_close(second, 2 * model.sample(9)[0].select('noise').mean())
    query = ATE(model, outcome='response', alternative={'exposure': 0}, reference={}, sample_bs=9, seed=7)
    result, info = query.evaluate(sample_bs=13, seed=3)
    expected = MeanPotentialOutcome(model, outcome='response', intervention={}).evaluate(sample_bs=13, seed=3)[0]
    torch.testing.assert_close(result.value, -expected.value)
    assert info['alternative'] == {'exposure': 0} and info['reference'] == {}
    assert query.reference.sample_bs == 9 and query.seed == 7
    torch.testing.assert_close(query.evaluate()[0].value, query.evaluate()[0].value)
    with pytest.raises(ValueError, match='positive'):
        mean.evaluate(sample_bs=0)


def test_discrete_nonbinary_scores_probabilities_and_gradients():
    model = SEM2Discrete('dose ~ Categorical(probs=[1,2,3]); response <- d(dose; levels=3)',
                         node_cfgs={'response': MLPConfig(hidden_dims=[])})
    layer = model.node_dict['response'].nn_module.net[0]
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.], [0.5], [-0.25]]))
        layer.bias.copy_(torch.tensor([0.2, -0.1, 0.3]))
    model.double()
    variable = VariableInfo('response', categorical=True,
                            encoding={'low': 0, 'middle': 1, 'high': 2},
                            numeric_mapping={'low': -1., 'middle': 2., 'high': 5.})
    query = ATE(model, outcome='response', alternative={'dose': 2}, reference={'dose': 0}, variable=variable)
    result, info = query.evaluate(sample_bs=4)
    assert result.value.dtype == torch.float64 and result.value.device == layer.weight.device
    scores = layer.weight.new_tensor([-1., 2., 5.])
    expected = ((layer.weight[:, 0] * 2 + layer.bias).softmax(0) - layer.bias.softmax(0)) @ scores
    torch.testing.assert_close(result.value, expected)
    for actual, reference in zip(torch.autograd.grad(result.value, tuple(layer.parameters())),
                                 torch.autograd.grad(expected, tuple(layer.parameters()))):
        torch.testing.assert_close(actual, reference)
    assert info['method'] == 'probability_weighted'
    probability = MeanPotentialOutcome(model, outcome='response', intervention={'dose': 2},
                                       variable=VariableInfo('response', numeric_mapping={0: 0., 1: 0., 2: 1.}))
    torch.testing.assert_close(probability.evaluate(sample_bs=2)[0].value,
                               (layer.weight[:, 0] * 2 + layer.bias).softmax(0)[2])
    # Exogenous outcomes are sampled, not looked up in the endogenous state table.
    exogenous = MeanPotentialOutcome(model, outcome='dose', intervention={'dose': 2},
                                     variable=VariableInfo('dose', numeric_mapping={0: 0., 1: 1., 2: 2.}))
    torch.testing.assert_close(exogenous.evaluate(sample_bs=2)[0].value, layer.weight.new_tensor(2.))


def test_query_validation_precedes_sampling_and_integer_coercion():
    model = SEM2Discrete('u ~ Normal(0,1); dose <- d(u; levels=3); response <- d(dose; levels=2)',
                         node_cfgs={name: MLPConfig(hidden_dims=[]) for name in ['dose', 'response']})
    scores = VariableInfo('response', numeric_mapping={0: 0., 1: 1.})
    with patch.object(model, 'sample', side_effect=AssertionError('must not sample')):
        for intervention in [{'dose': 0.9}, {'dose': 1 + 1e-9}, {'dose': 3},
                             {'u': float('nan')}, {'u': 1e100}, {'missing': 0}]:
            with pytest.raises(ValueError):
                MeanPotentialOutcome(model, outcome='response', intervention=intervention, variable=scores)
        with pytest.warns(UserWarning, match='explicit numerical scores'):
            with pytest.raises(ValueError, match='numeric_mapping'):
                MeanPotentialOutcome(model, outcome='response', intervention={})
        with pytest.raises(ValueError, match='each declared state'):
            MeanPotentialOutcome(model, outcome='response', intervention={},
                                 variable=VariableInfo('response', numeric_mapping={0: 0.}))
        with pytest.raises(ValueError, match='outcome'):
            MeanPotentialOutcome(model, outcome='missing', intervention={})


def test_eval_mode_restored_even_on_failure_with_gradients_and_no_scalar_checks():
    model = SEM2('u ~ Normal(0,1); response <- n(u)',
                 node_cfgs={'response': MLPConfig(hidden_dims=[4], dropout=0.5)})
    # Preserve heterogeneous flags, not just the top-level training flag.
    model.node_dict['u'].eval()
    flags = [module.training for module in model.modules()]
    query = MeanPotentialOutcome(model, outcome='response', intervention={'u': 1})
    with profile(activities=[ProfilerActivity.CPU]) as execution:
        result, info = query.evaluate(sample_bs=4)
        result.value.backward()
    assert 'aten::_local_scalar_dense' not in {event.key for event in execution.key_averages()}
    torch.testing.assert_close(result.value, query.evaluate(sample_bs=4)[0].value)
    assert info['evaluation_mode'] == 'eval'
    assert [module.training for module in model.modules()] == flags
    assert any(parameter.grad is not None for parameter in model.parameters())
    with patch.object(query.sampler, 'evaluate', side_effect=RuntimeError('sampling failed')):
        with pytest.raises(RuntimeError, match='sampling failed'):
            query.evaluate()
    assert [module.training for module in model.modules()] == flags


def _two_stage_model(reverse_parents=False):
    """Initialize real MLP mechanisms to the binary reference probability tables."""
    parents = 'U,L,A2,A1' if reverse_parents else 'A1,A2,L,U'
    model = SEM2Discrete(f'noise ~ Normal(0,1); U <- d(noise; levels=2); A1 <- d(noise; levels=2); '
                         f'L <- d(A1,U; levels=2); A2 <- d(L; levels=2); Y <- d({parents}; levels=2)',
                         node_cfgs={name: MLPConfig(hidden_dims=widths) for name, widths in
                                    [('U', []), ('A1', []), ('L', [4]), ('A2', [2]), ('Y', [16])]})
    with torch.no_grad():
        for name in ['U', 'A1']:
            layer = model.node_dict[name].nn_module.net[0]
            layer.weight.zero_()
            layer.bias.zero_()  # Both are independent fair coins.
        for name in ['L', 'A2', 'Y']:
            node = model.node_dict[name]
            states = torch.tensor(list(product([0., 1.], repeat=len(node.predecessors))))
            probabilities = []
            for state in states.tolist():
                p = dict(zip(node.predecessors, state))
                if name == 'L':
                    probability = 0.2 + 0.3*p['A1'] + 0.3*p['U']
                elif name == 'A2':
                    probability = 0.25 + 0.5*p['L']
                else:
                    probability = 0.1 + 0.1*p['A1'] + 0.2*p['A2'] + 0.2*p['L'] + 0.1*p['U'] + 0.1*p['A1']*p['A2']
                probabilities.append(probability)
            hidden, output = node.nn_module.net[0], node.nn_module.net[-1]
            # ReLU(1 - Hamming distance) is an indicator for each binary context.
            hidden.weight.copy_(2 * states - 1)
            hidden.bias.copy_(1 - states.sum(1))
            output.weight.zero_()
            output.weight[1].copy_(torch.tensor(probabilities).logit())
            output.bias.zero_()
    return model


def test_two_stage_known_means_contrasts_and_parent_order():
    variable = VariableInfo('Y', numeric_mapping={0: 0., 1: 1.})
    for reverse in [False, True]:
        model = _two_stage_model(reverse)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        for (a1, a2), expected in zip(product([0, 1], repeat=2), [0.22, 0.42, 0.38, 0.68]):
            # Independent sum over the latent U and regenerated intermediate L.
            reference = 0.
            for u, l in product([0, 1], repeat=2):
                p_l = 0.2 + 0.3*a1 + 0.3*u
                p_y = 0.1 + 0.1*a1 + 0.2*a2 + 0.2*l + 0.1*u + 0.1*a1*a2
                reference += 0.5 * (p_l if l else 1-p_l) * p_y
            assert reference == pytest.approx(expected)
            query = MeanPotentialOutcome(model, outcome='Y', intervention={'A1': a1, 'A2': a2}, variable=variable)
            torch.testing.assert_close(query.evaluate(sample_bs=3)[0].value, torch.tensor(reference))
        for reference, expected in [({'A1': 0, 'A2': 0}, 0.46), ({'A1': 1, 'A2': 0}, 0.30)]:
            query = ATE(model, outcome='Y', alternative={'A1': 1, 'A2': 1}, reference=reference, variable=variable)
            result, _ = query.evaluate(sample_bs=3)
            torch.testing.assert_close(result.value, torch.tensor(expected))
            result.value.backward()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])
        law, _ = model.sample(3, do_intervention=DataBatch(torch.ones(3, 2), ['A1', 'A2']))
        torch.testing.assert_close(law.probabilities.sum(-1), torch.ones(3))
