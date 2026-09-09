"""Support inspection for exogenous and endogenous nodes."""

import pytest
import torch
from torch.distributions import constraints
import torch.distributions as D

from sem2.default_configs import LinearNodeConfig, MLPConfig
from sem2.model import SEM2Discrete
from sem2.model.components import (
    SupportInformation, RandomVariable, NodeExogenous,
    NodeLinear, NodeMLP, NodeMLPDiscrete, DiscreteModes,
)


def exogenous(dist):
    return NodeExogenous("u", type(dist).__name__, dist)


def discrete_node(n_classes):
    return NodeMLPDiscrete(
        "y", "d(u)", MLPConfig(hidden_dims=[4], output_dim=n_classes),
        [exogenous(D.Normal(0, 1))],
    )


class TestSupportInformation:
    def test_unknown_defaults(self):
        info = SupportInformation()
        assert info.constraint is None
        assert info.values is None
        assert info.is_conservative

    def test_custom_variable_can_report_unknown_support(self):
        class CustomVariable(RandomVariable):
            pass

        info = CustomVariable().get_support()
        assert info.constraint is None
        assert info.values is None
        assert info.is_conservative


class TestSupportRepr:
    @pytest.mark.parametrize("info, expected", [
        (SupportInformation(),
         "SupportInformation(constraint=unknown, is_conservative=True)"),
        (SupportInformation(constraint=constraints.real),
         "SupportInformation(constraint=Real(), is_conservative=True)"),
        (SupportInformation(constraint=constraints.interval(-2, 3)),
         "SupportInformation(constraint=Interval(lower_bound=-2, upper_bound=3), "
         "is_conservative=True)"),
    ])
    def test_unknown_and_constraint_repr(self, info, expected, capsys):
        assert repr(info) == expected
        print(info)
        assert capsys.readouterr().out == expected + "\n"

    @pytest.mark.parametrize("n_classes", [2, 3])
    def test_finite_values_repr(self, n_classes):
        info = discrete_node(n_classes).get_support()
        assert repr(info) == (
            f"SupportInformation(values={list(range(n_classes))}, is_conservative=False)"
        )

    def test_small_batched_values_keep_structure(self):
        info = exogenous(D.Bernoulli(torch.tensor([0.2, 0.7]))).get_support()
        assert repr(info) == (
            "SupportInformation(values=[[0.0, 0.0], [1.0, 1.0]], is_conservative=True)"
        )

    @pytest.mark.parametrize("shape", [(100,), (2, 100)])
    def test_large_values_show_metadata(self, shape):
        values = torch.zeros(shape, dtype=torch.int64)
        info = SupportInformation(values=values)
        assert repr(info) == (
            f"SupportInformation(values=Tensor(shape={shape}, dtype=torch.int64, "
            "device=cpu), is_conservative=True)"
        )
        assert info.values is values
        assert torch.count_nonzero(values) == 0

    def test_meta_values_do_not_require_data(self):
        info = discrete_node(3).to("meta").get_support()
        assert repr(info) == (
            "SupportInformation(values=Tensor(shape=(3,), dtype=torch.int64, "
            "device=meta), is_conservative=False)"
        )

    def test_long_constraint_is_truncated_to_one_line(self):
        info = SupportInformation(constraint=constraints.interval(
            torch.arange(100), torch.arange(100) + 1,
        ))
        text = repr(info)
        assert "\n" not in text
        assert "..." in text
        assert len(text) < 150
        assert text.endswith("is_conservative=True)")


class TestExogenousSupport:
    @pytest.mark.parametrize("dist, values, expected", [
        (D.Normal(0, 1), [-100.0, 0.0, 100.0], [True, True, True]),
        (D.Uniform(-2, 3), [-3.0, -2.0, 3.0, 4.0], [False, True, True, False]),
        (D.Exponential(1), [-1.0, 0.0, 1.0], [False, True, True]),
        (D.Poisson(2), [-1.0, 0.0, 1.5, 2.0], [False, True, False, True]),
    ])
    def test_uses_distribution_constraint(self, dist, values, expected):
        info = exogenous(dist).get_support()
        assert isinstance(info, SupportInformation)
        torch.testing.assert_close(
            info.constraint.check(torch.tensor(values)), torch.tensor(expected),
        )
        assert info.values is None
        assert info.is_conservative  # Backend domains are not certified exact supports.

    @pytest.mark.parametrize("dist, expected", [
        (D.Bernoulli(0.3), [0, 1]),
        (D.Categorical(probs=torch.tensor([0.2, 0.3, 0.5])), [0, 1, 2]),
        (D.Binomial(total_count=3, probs=0.5), [0, 1, 2, 3]),
    ])
    def test_finite_values(self, dist, expected):
        info = exogenous(dist).get_support()
        assert info.values.tolist() == expected
        assert info.constraint.check(info.values).all()
        assert info.values.device == dist.enumerate_support().device

    def test_zero_probability_categories_are_conservative(self):
        dist = D.Categorical(probs=torch.tensor([1.0, 0.0, 0.0]))
        info = exogenous(dist).get_support()
        assert info.values.tolist() == [0, 1, 2]
        assert info.is_conservative

    def test_keeps_native_enumeration_shape(self):
        dist = D.Bernoulli(torch.tensor([0.2, 0.7]))
        info = exogenous(dist).get_support()
        assert info.values.shape == (2, 2)
        torch.testing.assert_close(info.values, dist.enumerate_support())

    def test_unsupported_enumeration_keeps_constraint(self):
        dist = D.Binomial(total_count=torch.tensor([1.0, 2.0]), probs=0.5)
        info = exogenous(dist).get_support()
        assert info.values is None  # Finite, but PyTorch cannot enumerate this batch.
        assert info.constraint.check(torch.tensor([1.0, 2.0])).all()
        assert info.is_conservative

    def test_unavailable_distribution_support(self):
        info = exogenous(D.Distribution(validate_args=False)).get_support()
        assert info.constraint is None
        assert info.values is None
        assert info.is_conservative


class TestEndogenousSupport:
    @pytest.mark.parametrize("n_classes", [2, 3])
    def test_discrete_indices_and_membership(self, n_classes):
        info = discrete_node(n_classes).get_support()
        torch.testing.assert_close(info.values, torch.arange(n_classes))
        assert not info.is_conservative
        torch.testing.assert_close(
            info.constraint.check(torch.tensor([-1.0, 0.0, 0.5, float(n_classes)])),
            torch.tensor([False, True, False, False]),
        )

    def test_inspection_does_not_sample_or_change_node(self):
        node = discrete_node(3)
        probs = node([torch.zeros(4, 1)], mode=DiscreteModes.SOFTMAX)
        rng = torch.random.get_rng_state().clone()
        state = {name: value.clone() for name, value in node.state_dict().items()}
        info = node.get_support()
        assert info.values.shape == (3,)  # Category values, not a probability vector.
        assert node.get_value() is probs
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        assert node.state_dict().keys() == state.keys()
        for name, value in node.state_dict().items():
            torch.testing.assert_close(value, state[name])

    def test_discrete_values_follow_parameter_device(self):
        node = discrete_node(3).to("meta")
        assert node.get_support().values.device.type == "meta"

    @pytest.mark.parametrize("kind", ["linear", "mlp"])
    def test_real_domain_is_only_a_conservative_bound(self, kind):
        parent = exogenous(D.Bernoulli(0.5))
        if kind == "linear":
            node = NodeLinear(
                "y", "1*u", [1.0], LinearNodeConfig(use_bias=False), [parent],
            )
        else:
            node = NodeMLP(
                "y", "n(u)", MLPConfig(hidden_dims=[4], activation="sigmoid"),
                [parent],
            )
        info = node.get_support()
        assert info.constraint is constraints.real
        assert info.values is None
        assert info.is_conservative
        assert info.constraint.check(torch.tensor([-2.0, 2.0])).all()

    def test_model_exposes_support_before_sampling(self):
        model = SEM2Discrete(
            "u ~ Normal(0,1); y <- d(u)",
            node_cfgs={"y": MLPConfig(hidden_dims=[4], output_dim=3)},
        )
        for node in model.node_dict.values():
            assert isinstance(node.get_support(), SupportInformation)
            assert node.get_value() is None
        info = model.node_dict["y"].get_support()
        probs, extra = model.sample(batch_size=4)
        assert torch.equal(extra["states"][:, 0], info.values)
        torch.testing.assert_close(probs.sum(dim=1), torch.ones(4))
