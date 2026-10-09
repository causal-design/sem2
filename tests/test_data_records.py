"""Focused record/preparation regressions, not Step-7 estimator validation."""
import pandas as pd
import pytest
import torch

from sem2.data.numeric import numeric_values
from sem2.data.preparation import prepare_data
from sem2.data.records import DataBatch, ScalarResult, VariableInfo
from sem2.default_configs import MLPConfig
from sem2.model.distributions import DiscreteJointDistribution
from sem2.model.models.discrete import SEM2Discrete


def test_named_record_dimensions_selection_and_scalar_contract():
    values = torch.arange(12.0).reshape(4, 3).requires_grad_()
    variables = [VariableInfo('vector', dim=2), VariableInfo('scalar')]
    batch = DataBatch(values, variables)
    assert isinstance(batch.variables, list)
    assert batch[1].values.shape == (1, 3)
    selected = batch.subset(['scalar', 'vector'])
    torch.testing.assert_close(selected.values, values[:, [2, 0, 1]])
    assert selected.variables == [variables[1], variables[0]]
    result = ScalarResult(selected.values.sum(), 'sum', selected.variables)
    assert not isinstance(result, DataBatch)
    result.value.backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))
    with pytest.raises(TypeError, match='VariableInfo'):
        DataBatch(torch.zeros(2, 1), ['x'])
    with pytest.raises(ValueError, match='two-dimensional'):
        DataBatch(torch.zeros(2), [VariableInfo('x')])
    with pytest.raises(ValueError, match='scalar tensor'):
        ScalarResult(torch.zeros(1), 'not_scalar')


def test_categorical_preparation_preserves_frame_levels_order_and_separate_scores():
    model = SEM2Discrete('u ~ Normal(0,1); y_var <- d(u; levels=3)',
                         node_cfgs={'y_var': MLPConfig(hidden_dims=[4])})
    frame = pd.DataFrame({'y var': pd.Categorical(
        ['middle', 'low', None], categories=['low', 'middle', 'high'], ordered=True)})
    original = frame.copy(deep=True)
    graph_names = tuple(model.node_specs)
    variable = VariableInfo('y_var', numeric_mapping={'low': -1., 'middle': 2., 'high': 10.})
    batch, report = prepare_data(model, frame, [variable])
    pd.testing.assert_frame_equal(frame, original)
    assert tuple(model.node_specs) == graph_names
    assert batch.variables[0].encoding == {'low': 0, 'middle': 1, 'high': 2}
    assert batch.variables[0].ordered is True
    assert variable.encoding is None
    assert report == {'column_names': {'y var': 'y_var'}, 'observed': ('y_var',),
                      'unobserved': ('u',), 'input_rows': 3, 'retained_rows': 2, 'dropped_rows': 1}
    torch.testing.assert_close(batch.values[:, 0], torch.tensor([1., 0.]))
    torch.testing.assert_close(numeric_values(batch)[:, 0], torch.tensor([2., -1.]))
    torch.testing.assert_close(batch.values[:, 0], torch.tensor([1., 0.]))
    # An explicit mapping also applies to quantitative finite values.
    quantitative = DataBatch(torch.tensor([[2.], [4.]]),
                             [VariableInfo('x', categorical=False, numeric_mapping={2.: 5., 4.: 9.})])
    torch.testing.assert_close(numeric_values(quantitative), torch.tensor([[5.], [9.]]))


def test_preparation_rejects_name_collisions_and_preserves_legacy_no_score_warning():
    model = SEM2Discrete('u ~ Normal(0,1); y_var <- d(u; levels=2)',
                         node_cfgs={'y_var': MLPConfig(hidden_dims=[4])})
    with pytest.raises(ValueError, match='collide'):
        prepare_data(model, pd.DataFrame({'y var': [0.], 'y_var': [1.]}))
    batch, _ = prepare_data(model, pd.DataFrame({'y_var': pd.Categorical(
        ['no', 'yes'], categories=['no', 'yes'])}))
    with pytest.warns(UserWarning, match='category codes'):
        numeric = numeric_values(batch)
    torch.testing.assert_close(numeric, batch.values)
    # This preserves compatibility, not approval of the deferred category metric.


def test_joint_distribution_preserves_draw_axis_marginal_and_gradients():
    states = DataBatch(torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]),
                       [VariableInfo('x'), VariableInfo('y')])
    logits = torch.tensor([[0., 1., 2., 3.], [3., 1., 0., 2.]], requires_grad=True)
    probabilities = logits.softmax(-1)
    law = DiscreteJointDistribution(states, probabilities)
    marginal = law.marginalize('y').to(dtype=torch.float64)
    assert marginal.states.names == ('y',)
    assert marginal.probabilities.shape == (2, 2)
    assert marginal.states.values.dtype == torch.long
    expected = torch.stack([probabilities[:, 0] + probabilities[:, 2],
                            probabilities[:, 1] + probabilities[:, 3]], dim=1)
    torch.testing.assert_close(marginal.probabilities, expected.double())
    marginal.probabilities[:, 1].sum().backward()
    assert torch.isfinite(logits.grad).all() and torch.count_nonzero(logits.grad) > 0
    empty = law.marginalize([])
    assert empty.states.values.shape == (1, 0)
    torch.testing.assert_close(empty.probabilities, torch.ones(2, 1))
    model = SEM2Discrete('u ~ Normal(0,1); y <- d(u; levels=2)',
                         node_cfgs={'y': MLPConfig(hidden_dims=[4])})
    prediction, _ = model.sample(3)
    assert isinstance(prediction, DiscreteJointDistribution)
    assert prediction.probabilities.shape == (3, 2)
    torch.testing.assert_close(prediction.probabilities.sum(-1), torch.ones(3))
