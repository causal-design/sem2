"""Core formula, sampling, and analytical-model regressions."""
import pytest
import torch

from sem2.data.records import DataBatch, VariableInfo
from sem2.default_configs import LinearNodeConfig, MLPConfig
from sem2.model.construction.dag import _DAG
from sem2.model.construction.sem_types import SEMType
from sem2.model.models.base import SEM2
from sem2.model.models.linear import SEM2Linear
from sem2.operators.divergence import WeightedFrobeniusDistance
from sem2.operators.linear import LogLikelihood
from sem2.utils import set_seed


def test_formula_lowering_preserves_names_coefficients_and_dependencies():
    dag = _DAG(' y <- -0.5*x + e2; x <- 1*e1; e1, e2 ~ Normal( 0 , 1 ) ')
    assert dag.input_node == {'e1', 'e2', 'x', 'y'}
    assert dag.graph == {'e1': [], 'e2': [], 'x': ['e1'], 'y': ['x', 'e2']}
    assert dag.node_specs['e1'].dist_params == ([0., 1.], {})
    assert dag.node_specs['y'].coefficients == [-0.5, None]
    order = dag.topological_order
    assert order.index('e1') < order.index('x') < order.index('y')
    assert order.index('e2') < order.index('y')
    assert dag.inferred_sem_type == SEMType.LINEAR_GAUSSIAN


@pytest.mark.parametrize('formula, message', [
    ('x <- y; y <- x', 'cycle'),
    ('y <- x', 'undefined parent'),
    ('e ~ Normal(0,', 'Could not parse'),
    ('e ~ NotARealDist(0,1)', 'Unknown distribution'),
    ('e ~ Normal(0,1); unobs(e)', 'Formula must contain'),
])
def test_invalid_formulas_fail_at_construction(formula, message):
    with pytest.raises(ValueError, match=message):
        _DAG(formula)


def test_unused_node_configuration_is_rejected():
    with pytest.raises(AssertionError):
        SEM2Linear('e ~ Normal(0,1); y <- e', node_cfgs={'typo': LinearNodeConfig()})


def test_nested_and_mixed_mechanisms_sample_and_backpropagate():
    set_seed(0)
    model = SEM2('e1,e2,e3 ~ Normal(0,1); y <- n(e1,n(e2,e3)) + e3')
    assert model.inferred_sem_type == SEMType.NONLINEAR
    assert len([name for name in model.node_specs if name.startswith('y:')]) == 2
    draws, _ = model.sample(4)
    assert set(draws.names) == {'e1', 'e2', 'e3', 'y'}
    assert draws.select('y').shape == (4, 1)
    draws.select('y').sum().backward()
    parameters = list(model.parameters())
    assert parameters and all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)


def test_mlp_uses_multidimensional_parents_and_output_configuration():
    model = SEM2('u,v ~ Normal(0,1); y <- n(u,v)',
                 node_cfgs={'u': 2, 'y': MLPConfig(hidden_dims=[], output_dim=2)})
    layer = model.node_dict['y'].mlp.net[0]
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1., 2., 3.], [-1., 0., 1.]]))
        layer.bias.zero_()
    fixed = DataBatch(torch.tensor([[1., 2., 4.]]), [VariableInfo('u', dim=2), VariableInfo('v')])
    draws, _ = model.sample(1, do_intervention=fixed)
    torch.testing.assert_close(draws.select('y'), torch.tensor([[17., 3.]]))


def test_fixed_and_learned_coefficients_control_values_and_gradients():
    model = SEM2Linear('u,v ~ Normal(0,1); y <- -0.5*u + v',
                       node_cfgs={'y': LinearNodeConfig(use_intercept=False)})
    node = model.node_dict['y']
    with torch.no_grad():
        node.linear.weight.copy_(torch.tensor([[99., 3.]]))
    fixed = DataBatch(torch.tensor([[2., 1.], [-4., 2.]]), ['u', 'v'])
    draws, _ = model.sample(2, do_intervention=fixed)
    torch.testing.assert_close(draws.select('y'), torch.tensor([[2.], [8.]]))
    draws.select('y').sum().backward()
    torch.testing.assert_close(node.linear.weight.grad, torch.tensor([[0., 3.]]))


def test_fixed_coefficients_expand_over_multidimensional_predecessors():
    model = SEM2('u ~ Normal(0,1); x <- u; y <- 0.5*x', node_cfgs={
        'x': LinearNodeConfig(output_dim=2, use_intercept=False),
        'y': LinearNodeConfig(use_intercept=False),
    })
    with torch.no_grad():
        model.node_dict['x'].linear.weight.copy_(torch.tensor([[2.], [4.]]))
        model.node_dict['y'].linear.weight.fill_(99.)
    draws, _ = model.sample(2, do_intervention=DataBatch(torch.tensor([[1.], [3.]]), ['u']))
    torch.testing.assert_close(draws.select('x'), torch.tensor([[2., 4.], [6., 12.]]))
    torch.testing.assert_close(draws.select('y'), torch.tensor([[3.], [9.]]))


def test_fixed_coefficients_follow_module_dtype_without_becoming_parameters():
    model = SEM2Linear('u,v ~ Normal(0,1); y <- -0.5*u + v + 2')
    node = model.node_dict['y'].double()
    with torch.no_grad():
        node.linear.weight.fill_(3.)
    buffers = dict(node.named_buffers())
    assert {'fix_weights', '_fixed_vals', '_ones', '_fixed_intercept'} <= buffers.keys()
    assert not set(buffers) & dict(node.named_parameters()).keys()
    assert node._fixed_vals.dtype == torch.float64
    result = node([torch.ones(1, 1, dtype=torch.float64)] * 2, batch_size=1)
    torch.testing.assert_close(result, torch.tensor([[4.5]], dtype=torch.float64))


def test_intervention_replaces_incoming_edge_but_not_upstream_or_later_draws():
    model = SEM2Linear('e ~ Normal(0,1); x <- 2*e + 0; y <- 3*x + 1*e + 0')
    set_seed(7)
    natural, _ = model.sample(4)
    set_seed(7)
    intervened, _ = model.sample(4, do_intervention=DataBatch(torch.ones(4, 1), ['x']))
    torch.testing.assert_close(intervened.select('e'), natural.select('e'))
    torch.testing.assert_close(intervened.select('x'), torch.ones(4, 1))
    torch.testing.assert_close(intervened.select('y'), 3 + natural.select('e'))
    set_seed(7)
    subsequent, _ = model.sample(4)
    torch.testing.assert_close(subsequent.values, natural.values)
    with pytest.raises(ValueError):
        model.sample(4, do_intervention=DataBatch(torch.zeros(4, 1), ['missing']))


def test_linear_moments_and_named_order_match_analytical_chain():
    model = SEM2Linear('e ~ Normal(2,1); noise ~ Normal(0,1); '
                       'x <- 1*e + 0; y <- 2*x + 1*noise + 3')
    indices = [model.node_order.index(name) for name in ['x', 'y']]
    torch.testing.assert_close(model.mean_vector()[indices], torch.tensor([2., 7.]))
    torch.testing.assert_close(model.cov_matrix(['y', 'x']), torch.tensor([[5., 2.], [2., 1.]]))
    coefficients = model.coef_matrix()
    expected = torch.zeros_like(coefficients)
    ix = model._node_index
    expected[ix['x'], ix['e']] = 1.
    expected[ix['y'], ix['x']] = 2.
    expected[ix['y'], ix['noise']] = 1.
    torch.testing.assert_close(coefficients, expected)


def test_linear_covariance_uses_non_gaussian_second_moments():
    model = SEM2Linear('e,noise ~ Uniform(-1,1); x <- 1*e + 0; y <- 2*x + 1*noise + 0')
    assert model.inferred_sem_type == SEMType.LINEAR
    torch.testing.assert_close(model.cov_matrix(['x', 'y']), torch.tensor([[1., 2.], [2., 5.]]) / 3)


@pytest.mark.parametrize('formula, cfgs, message', [
    ('e ~ Normal(0,1); y <- n(e)', None, 'linear'),
    ('e ~ Normal(0,1); y <- e', {'y': LinearNodeConfig(output_dim=2)}, 'scalar'),
    ('e ~ MultivariateNormal([0,0], [[1,0],[0,1]]); y <- e', None, 'scalar independent'),
])
def test_analytical_model_rejects_unsupported_mechanisms_and_dimensions(formula, cfgs, message):
    with pytest.raises(ValueError, match=message):
        SEM2Linear(formula, node_cfgs=cfgs)


def test_loglikelihood_matches_torch_reference_with_named_column_alignment():
    model = SEM2Linear('e,noise ~ Normal(0,1); x <- 1*e + 0; y <- 2*x + 1*noise + 0')
    data = DataBatch(torch.tensor([[1., 0.], [-1., 2.], [3., 1.]]), ['y', 'x'])
    reference = torch.distributions.MultivariateNormal(torch.zeros(2), torch.tensor([[1., 2.], [2., 5.]]))
    expected = reference.log_prob(data.select(['x', 'y'])).mean()
    explicit, _ = LogLikelihood(model, ['x', 'y'], loc=torch.zeros(2)).evaluate(data)
    implicit, _ = LogLikelihood(model).evaluate(data)
    torch.testing.assert_close(explicit.value, expected)
    torch.testing.assert_close(implicit.value, expected)


def test_centered_likelihood_overrides_model_mean_and_conflicts_with_loc():
    model = SEM2Linear('e ~ Normal(2,1); x <- 1*e + 0')
    data = DataBatch(torch.zeros(4, 1), ['x'])
    centered, _ = LogLikelihood(model, centered=True).evaluate(data)
    explicit, _ = LogLikelihood(model, loc=torch.zeros(1)).evaluate(data)
    model_mean, _ = LogLikelihood(model).evaluate(data)
    torch.testing.assert_close(centered.value, explicit.value)
    assert centered.value > model_mean.value
    with pytest.raises(ValueError, match='centered'):
        LogLikelihood(model, centered=True, loc=torch.zeros(1))


def test_non_gaussian_likelihood_warns_and_matches_moment_reference():
    model = SEM2Linear('u,v ~ Beta(0.5,0.5); x <- 1*u + 0; y <- 2*x + 1*v + 0')
    data = DataBatch(torch.tensor([[0., 1.], [1., 2.], [0.5, 0.5]]), ['x', 'y'])
    with pytest.warns(UserWarning, match='pseudo-log-likelihood'):
        objective = LogLikelihood(model)
    result, _ = objective.evaluate(data)
    reference = torch.distributions.MultivariateNormal(
        torch.tensor([0.5, 1.5]), torch.tensor([[1., 2.], [2., 5.]]) / 8,
    )
    torch.testing.assert_close(result.value, reference.log_prob(data.values).mean())


def test_likelihood_requires_finite_second_moments():
    model = SEM2Linear('e ~ Cauchy(0,1); y <- 1*e + 0')
    with pytest.warns(UserWarning, match='pseudo-log-likelihood'):
        with pytest.raises(ValueError, match='finite second moments'):
            LogLikelihood(model, ['y'], loc=torch.zeros(1))


def test_likelihood_requires_analytical_model_and_nonempty_observed_variables():
    with pytest.warns(UserWarning, match='sem_type'):
        model = SEM2('e ~ Normal(0,1); y <- e')
    with pytest.warns(UserWarning, match='special child class'):
        with pytest.raises(TypeError, match='SEM2Linear.*cov_matrix'):
            LogLikelihood(model, ['y'])
    with pytest.raises(ValueError, match='observed variable'):
        LogLikelihood(SEM2Linear('e ~ Normal(0,1); y <- e'), [])


def test_weighted_covariance_loss_matches_value_and_gradient_reference():
    model = SEM2Linear('u,v ~ Normal(0,1); x <- 1*u + 0; y <- x + 1*v + 0')
    weight = model.node_dict['y'].linear.weight
    with torch.no_grad():
        weight.fill_(2.)
    data = DataBatch(torch.tensor([[0., 1.], [1., 3.], [2., -1.]]), ['x', 'y'])
    weights = torch.tensor([[1., 0.5], [0.5, 2.]])
    result, _ = WeightedFrobeniusDistance(model, weight=weights).evaluate(data)
    slope = weight[0, 0]
    covariance = torch.cov(data.values.T)
    reference = ((covariance[0, 0] - 1)**2 + (covariance[0, 1] - slope)**2
                 + 2 * (covariance[1, 1] - slope**2 - 1)**2)
    torch.testing.assert_close(result.value, reference)
    torch.testing.assert_close(torch.autograd.grad(result.value, weight)[0],
                               torch.autograd.grad(reference, weight)[0])
