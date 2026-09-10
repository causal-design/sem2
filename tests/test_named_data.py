"""Essential named-data contracts and the dummy-loader/operator/trainer path."""
import pytest
import torch

from sem2.data import NamedBatch, VariableInfo, FiniteDistribution
from sem2.dataloader import DummyDiscreteDataloader
from sem2.default_configs import LinearNodeConfig, MLPConfig, TrainingConfig
from sem2.model import SEM2Linear, SEM2Discrete
from sem2.operators.linear import LogLikelihood, WeightedFrobeniusDistance
from sem2.operators.discrete_energy_loss import DiscreteEnergyLoss
from sem2.operators.scalar_operators import EnergyLoss, MeanPotentialOutcome, ATE
from sem2.operators.vector_operators import PotentialOutcome
from sem2.sem2trainer import SEM2Trainer


def named(values, *names):
    return NamedBatch(torch.as_tensor(values), tuple(VariableInfo(n) for n in names))


def linear_model():
    return SEM2Linear(
        "ex, ey ~ Normal(0,1); x <- 1*ex; y <- x + 1*ey",
        node_cfgs={n: LinearNodeConfig(use_bias=False) for n in ("x", "y")},
    )


def discrete_model(names=("Z", "X", "Y"), instrument_levels=2):
    z, x, y = names
    return SEM2Discrete(
        f"{z} ~ Categorical(probs={[1] * instrument_levels}); {x} <- d({z}); {y} <- d({x})",
        node_cfgs={n: MLPConfig(hidden_dims=[4], output_dim=2) for n in (x, y)},
    )


def test_batch_selection_rows_device_and_encoding():
    info = (VariableInfo("group", True, True, {"no": 0, "yes": 1}), VariableInfo("value"))
    data = torch.tensor([[0., 4.], [1., 7.]], requires_grad=True)
    batch = NamedBatch(data, info)
    assert batch.names == ("group", "value")
    torch.testing.assert_close(batch.select("value", "group"), data[:, [1, 0]])
    assert batch.rows(0).values.shape == (1, 2)
    assert batch.rows(slice(1)).variables is info
    moved = batch.to(dtype=torch.float64)
    assert moved.variables is info
    assert moved.variables[0].encoding == {"no": 0, "yes": 1}
    assert batch.subset("value").names == ("value",)
    moved.select("value").sum().backward()
    torch.testing.assert_close(data.grad, torch.tensor([[0., 1.], [0., 1.]]))
    with pytest.raises(ValueError, match="missing"):
        batch.select("absent")


@pytest.mark.parametrize("names", [("x",), ("x", "x")])
def test_batch_rejects_ambiguous_layout(names):
    with pytest.raises(ValueError):
        named([[1., 2.]], *names)


def test_dummy_loader_carries_shared_metadata():
    loader = DummyDiscreteDataloader(2, batch_size=8)
    first, second = list(loader)
    assert first.names == ("Z", "X", "Y")
    assert first.variables is second.variables
    assert all(v.categorical and v.discrete for v in first.variables)
    assert first.values.shape == (8, 3)
    assert first.values.dtype == torch.long


@pytest.mark.parametrize("operator", [LogLikelihood, WeightedFrobeniusDistance])
def test_linear_operators_align_named_batches_and_preserve_gradients(operator):
    model = linear_model()
    data = torch.tensor([[0., 1.], [1., 3.], [2., 4.], [-1., -3.]])
    batch = named(data, "x", "y")
    op = operator(model, ["x", "y"])
    expected, _ = op.evaluate(data)
    actual, _ = op.evaluate(batch.subset("y", "x"))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_continuous_energy_aligns_names_with_unequal_sample_counts():
    model = linear_model()
    batch = named([[0., 1.], [1., 2.], [2., 4.]], "x", "y")
    op = EnergyLoss(TrainingConfig(device="cpu", sample_bs=5), model)
    torch.manual_seed(17)
    actual, _ = op.evaluate(batch.subset("y", "x"))
    torch.manual_seed(17)
    samples, _ = model.sample(batch_size=5)
    preds = torch.cat([samples[n] for n in batch.names], dim=1)
    pair = torch.cdist(preds, preds)
    expected = torch.cdist(batch.values, preds).mean() - pair.sum() / (2 * 5 * 4)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_finite_distribution_marginal_keeps_states_and_probabilities_aligned():
    states = named([[0, 0], [0, 1], [1, 0], [1, 1]], "x", "y")
    probs = torch.tensor([[.1, .2, .3, .4]], requires_grad=True)
    marginal = FiniteDistribution(states, probs).marginal("y")
    assert marginal.states.names == ("y",)
    torch.testing.assert_close(marginal.states.values, torch.tensor([[0], [1]]))
    torch.testing.assert_close(marginal.probabilities, torch.tensor([[.4, .6]]))
    marginal.probabilities[:, 1].sum().backward()
    torch.testing.assert_close(probs.grad, torch.tensor([[0., 1., 0., 1.]]))


def test_discrete_kernel_aligns_named_observations_and_uses_each_sample_count():
    states = named([[0, 0], [0, 1], [1, 0], [1, 1]], "x", "y")
    observations = named([[0, 0], [0, 1], [1, 1], [0, 1]], "x", "y")
    logits = torch.randn(1, 3, 4, requires_grad=True)
    p = logits.softmax(-1)
    kernel = DiscreteEnergyLoss(1., states, "cpu")
    actual, _ = kernel.energy_dist(p, x_true=observations.subset("y", "x"))
    distance = torch.cdist(states.values.float(), states.values.float())
    idx = torch.tensor([0, 1, 3, 1])
    cross = (p[0] @ distance[:, idx]).mean()
    pp = p[0] @ distance @ p[0].T
    tt = distance[idx][:, idx]
    expected = cross - (pp.sum() - pp.diag().sum()) / (2 * 3 * 2) - tt.sum() / (2 * 4 * 3)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("invalid", [0.5, 3.])
def test_discrete_kernel_rejects_invalid_values_before_integer_coercion(invalid):
    kernel = DiscreteEnergyLoss(1., named([[0], [1]], "y"), "cpu")
    with pytest.raises(ValueError, match="state"):
        kernel.energy_dist(torch.full((1, 2, 2), .5), x_true=named([[0.], [invalid]], "y"))
    # The intervention adapter must reject the same values before the discrete
    # sampler's internal integer conversion, too.
    with pytest.raises(ValueError, match="state"):
        PotentialOutcome(discrete_model(), "Y").evaluate(
            batch_size=2, set_nodes=named([[0.], [invalid]], "X"),
        )


def test_named_discrete_loss_is_invariant_to_column_order_and_node_renaming():
    model = discrete_model(instrument_levels=3)
    renamed = discrete_model(("instrument", "treatment", "outcome"), instrument_levels=3)
    for new, old in zip(renamed.parameters(), model.parameters()):
        with torch.no_grad():
            new.copy_(old)
    cfg = TrainingConfig(device="cpu", sample_bs=4, instrument_node="Z")
    batch = named([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [2, 0, 0], [2, 1, 1]], "Z", "X", "Y")
    loss, _ = EnergyLoss(cfg, model).evaluate(batch)
    renamed_batch = NamedBatch(batch.values, tuple(VariableInfo(n) for n in ("instrument", "treatment", "outcome")))
    actual, _ = EnergyLoss(cfg.model_copy(update={"instrument_node": "instrument"}), renamed).evaluate(
        renamed_batch.subset("outcome", "instrument", "treatment")
    )
    torch.testing.assert_close(actual, loss)
    actual.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in renamed.parameters())
    # A batch need not contain every declared instrument level.
    one_level, _ = EnergyLoss(cfg, model).evaluate(batch.rows(slice(0, 2)))
    assert torch.isfinite(one_level)


def test_discrete_joint_loss_can_fit_an_observed_marginal_without_iv():
    model = discrete_model()
    batch = named([[0], [1], [1]], "Y")
    loss, _ = EnergyLoss(TrainingConfig(device="cpu", sample_bs=4), model).evaluate(batch)
    assert torch.isfinite(loss)
    loss.backward()


def test_potential_outcomes_return_named_samples_or_named_distributions():
    linear = linear_model()
    intervention = named([[1.], [1.]], "x")
    rng = torch.random.get_rng_state()
    result, extra = PotentialOutcome(linear, "y").evaluate(batch_size=2, set_nodes=intervention)
    assert isinstance(result, NamedBatch)
    assert result.names == ("y",)
    torch.testing.assert_close(extra["set_nodes"]["x"], intervention.values)
    cfg = TrainingConfig(device="cpu", sample_bs=4)
    torch.random.set_rng_state(rng)
    continuous_mean, _ = MeanPotentialOutcome("y", "x", cfg, linear).evaluate(intervention)
    torch.testing.assert_close(continuous_mean, result.select("y").mean())
    continuous_effect, _ = ATE(cfg.model_copy(update={"T_node": "x", "Y_node": "y"}), linear).evaluate()
    continuous_effect.backward()
    assert torch.isfinite(continuous_effect)
    model = discrete_model()
    batch = next(iter(DummyDiscreteDataloader(1, 4)))
    result, _ = PotentialOutcome(model, "Y", variables=batch.variables).evaluate(
        batch_size=4, set_nodes=named([[1], [1], [1], [1]], "X")
    )
    assert isinstance(result, FiniteDistribution)
    assert result.states.variables[0] is batch.variables[2]
    torch.testing.assert_close(result.probabilities.sum(-1), torch.ones(4))
    mean, _ = MeanPotentialOutcome("Y", "X", cfg, model).evaluate(torch.ones(4, 1))
    torch.testing.assert_close(mean, result.probabilities.mean(0) @ result.states.select("Y").flatten().float())
    effect, _ = ATE(cfg, model).evaluate()
    effect.backward()
    assert torch.isfinite(effect)


def test_dummy_loader_trains_with_optional_ate_disabled():
    torch.manual_seed(19)
    cfg = TrainingConfig(device="cpu", epochs=1, sample_bs=4, instrument_node="Z", use_ate=False)
    model = discrete_model()
    before = [p.detach().clone() for p in model.parameters()]
    loader = DummyDiscreteDataloader(2, batch_size=32)
    trainer = SEM2Trainer(cfg, model, loader)
    trainer.train()
    assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), before))
    loss, metrics = trainer._fit_batch(next(iter(loader)))
    assert torch.isfinite(loss)
    assert "ATE" not in metrics
    assert all(not value.requires_grad for value in metrics.values())
