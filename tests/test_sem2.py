"""
Test suite for SEM2.
Run with:  python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn

from sem2.nn_models import _LinearConfig, _LinearModel, _MLP
from sem2.default_configs import LinearNodeConfig, MLPConfig, TrainingConfig
from sem2.nodes import (
    RandomVariable, SEM2Node,
    NodeExogenous,
    NodeEndogenous, NodeLinear, NodeNonlinear, NodeMLP,
)
from sem2.model import DAG, SEM2, SEMType

#probably should breakdown into smaller test files but for now is fine I guess

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

BS = 4


def make_model(formula, cfgs=None, device="cpu"):
    return SEM2(formula, node_cfgs=cfgs, device=device)


def run(model, bs=BS, set_nodes=None, return_dict=True):
    res, _ = model.sample(batch_size=bs, set_nodes=set_nodes)
    if return_dict:
        return res
    return torch.cat([res[k] for k in model.topological_order], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. _LinearModel (internal)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLinearModel:

    def test_output_shape(self):
        cfg = _LinearConfig(input_dim=3, output_dim=2)
        m = _LinearModel(cfg)
        assert m(torch.randn(BS, 3)).shape == (BS, 2)

    def test_no_bias(self):
        cfg = _LinearConfig(input_dim=3, output_dim=1, use_bias=False)
        m = _LinearModel(cfg)
        assert m.bias is None
        assert torch.allclose(m(torch.zeros(BS, 3)), torch.zeros(BS, 1))

    def test_init_weight(self):
        W = torch.tensor([[1.0, 2.0, 3.0]])
        cfg = _LinearConfig(input_dim=3, output_dim=1, init_weight_val=W, use_bias=False)
        m = _LinearModel(cfg)
        assert torch.allclose(m.weight, W)

    def test_init_weight_wrong_shape_raises(self):
        W = torch.tensor([[1.0, 2.0]])
        cfg = _LinearConfig(input_dim=3, output_dim=1, init_weight_val=W, use_bias=False)
        with pytest.raises(AssertionError):
            _LinearModel(cfg)

    def test_init_bias_with_no_bias_raises(self):
        cfg = _LinearConfig(input_dim=2, output_dim=1, use_bias=False,
                            init_bias_val=torch.tensor([0.5]))
        with pytest.raises(AssertionError):
            _LinearModel(cfg)

    def test_weight_is_parameter(self):
        cfg = _LinearConfig(input_dim=2, output_dim=1)
        m = _LinearModel(cfg)
        assert isinstance(m.weight, nn.Parameter)
        assert isinstance(m.bias, nn.Parameter)

    def test_forward_correctness(self):
        W = torch.tensor([[2.0, 0.0]])
        b = torch.tensor([1.0])
        cfg = _LinearConfig(input_dim=2, output_dim=1, init_weight_val=W, init_bias_val=b)
        m = _LinearModel(cfg)
        x = torch.tensor([[1.0, 5.0]])
        assert torch.allclose(m(x), torch.tensor([[3.0]]))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MLP
# ═══════════════════════════════════════════════════════════════════════════════

class TestMLP:

    def test_output_shape(self):
        cfg = MLPConfig(input_dim=4, hidden_dims=[8, 8], output_dim=2)
        assert _MLP(cfg)(torch.randn(BS, 4)).shape == (BS, 2)

    def test_prob_output_range(self):
        cfg = MLPConfig(input_dim=3, hidden_dims=[8], output_dim=1, prob_output=True)
        out = _MLP(cfg)(torch.randn(100, 3))
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_no_dropout_layers_when_zero(self):
        cfg = MLPConfig(input_dim=2, hidden_dims=[4], output_dim=1, dropout=0.0)
        assert not any(isinstance(l, nn.Dropout) for l in _MLP(cfg).net)

    def test_dropout_layers_when_nonzero(self):
        cfg = MLPConfig(input_dim=2, hidden_dims=[4], output_dim=1, dropout=0.3)
        assert any(isinstance(l, nn.Dropout) for l in _MLP(cfg).net)

    def test_all_activations(self):
        for act in ["linear", "relu", "tanh", "sigmoid", "leaky_relu", "elu", "gelu", "selu"]:
            cfg = MLPConfig(input_dim=2, hidden_dims=[4], output_dim=1, activation=act)
            assert _MLP(cfg)(torch.randn(2, 2)).shape == (2, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Nodes — unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNodeExogenous:
    """Tests for NodeExogenous — the concrete exogenous node.
    We build instances via the SEM2 model so the distribution is parsed and
    instantiated through the normal path (DAG -> SEM2._build_and_initialize_nodes).
    """

    def test_normal_shape(self):
        m = make_model("e ~ Normal(0,1)")
        assert m.layer["e"]([], batch_size=BS).shape == (BS, 1)

    def test_multidim_shape(self):
        m = make_model("e ~ Normal(0,1)", cfgs={"e": 3})
        assert m.layer["e"]([], batch_size=BS).shape == (BS, 3)

    def test_beta_range(self):
        m = make_model("b ~ Beta(0.5,0.5)")
        out = m.layer["b"]([], batch_size=100)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_no_predecessors(self):
        m = make_model("e ~ Normal(0,1)")
        assert m.layer["e"].predecessors == []

    def test_is_node_exogenous(self):
        """Concrete exogenous node must be NodeExogenous."""
        m = make_model("e ~ Normal(0,1)")
        assert isinstance(m.layer["e"], NodeExogenous)



    def test_dist_attribute_set(self):
        """NodeExogenous must have a dist attribute after construction."""
        m = make_model("e ~ Normal(0,1)")
        assert m.layer["e"].dist is not None


class TestNodeLinear:

    def _make(self, coefficients, cfg=None, out_dims=None):
        """Build a NodeLinear with fake predecessor nodes."""
        n_preds = len(coefficients)
        out_dims = out_dims or [1] * n_preds
        predecessors = [f"p{i}" for i in range(n_preds)]

        class FakeNode:
            def __init__(self, d): self.out_dim = d

        pred_nodes = [FakeNode(d) for d in out_dims]
        node = NodeLinear("y", "+".join(predecessors), predecessors, coefficients, cfg, pred_nodes)
        return node

    def test_basic_shape(self):
        node = self._make([None, None])
        assert node([torch.randn(BS, 1), torch.randn(BS, 1)], BS).shape == (BS, 1)

    def test_multi_output_shape(self):
        cfg = LinearNodeConfig(output_dim=3)
        node = self._make([None, None], cfg=cfg)
        assert node([torch.randn(BS, 1), torch.randn(BS, 1)], BS).shape == (BS, 3)

    def test_fixed_coeff_one(self):
        """Fixed coefficient 1.0 — output should be 1*input regardless of learned weight."""
        cfg = LinearNodeConfig(use_bias=False)
        node = self._make([1.0], cfg=cfg)
        with torch.no_grad():
            node.linear.weight.fill_(99.0)
        assert torch.allclose(node([torch.ones(1, 1)], batch_size=1), torch.tensor([[1.0]]))

    def test_fixed_coeff_negative(self):
        """Fixed coefficient -1.0 — output should be -1*input."""
        cfg = LinearNodeConfig(use_bias=False)
        node = self._make([-1.0], cfg=cfg)
        with torch.no_grad():
            node.linear.weight.fill_(99.0)
        assert torch.allclose(node([torch.ones(1, 1)], batch_size=1), torch.tensor([[-1.0]]))

    def test_fixed_coeff_fractional(self):
        """Fixed coefficient 0.5 — output should be 0.5*input."""
        cfg = LinearNodeConfig(use_bias=False)
        node = self._make([0.5], cfg=cfg)
        with torch.no_grad():
            node.linear.weight.fill_(99.0)
        assert torch.allclose(node([torch.ones(1, 1)], batch_size=1), torch.tensor([[0.5]]))

    def test_learned_weight_used(self):
        """None coefficient — learned weight should be used."""
        cfg = LinearNodeConfig(use_bias=False)
        node = self._make([None], cfg=cfg)
        with torch.no_grad():
            node.linear.weight.fill_(5.0)
        assert torch.allclose(node([torch.ones(1, 1)], batch_size=1), torch.tensor([[5.0]]))

    def test_mixed_fixed_and_learned(self):
        """Mix of fixed and learned in same node."""
        cfg = LinearNodeConfig(use_bias=False)
        node = self._make([1.0, None], cfg=cfg)
        # fix_weights: [True, False]
        assert node.fix_weights[0, 0].item() is True
        assert node.fix_weights[0, 1].item() is False

    def test_fix_weights_mask_shape_multi_output(self):
        """Mask must broadcast correctly when out_dim > 1."""
        cfg = LinearNodeConfig(output_dim=2, use_bias=False)
        node = self._make([1.0, None], cfg=cfg)
        assert node.fix_weights.shape == (1, 2)

    def test_fix_weights_expanded_for_multidim_predecessor(self):
        """Fixed flag must expand across all dims of a multi-output predecessor."""
        cfg = LinearNodeConfig(use_bias=False)
        node = self._make([1.0, None], cfg=cfg, out_dims=[2, 1])
        # p0 has out_dim=2 (fixed), p1 has out_dim=1 (learned)
        assert node.fix_weights.shape == (1, 3)
        assert node.fix_weights[0, 0].item() is True
        assert node.fix_weights[0, 1].item() is True
        assert node.fix_weights[0, 2].item() is False

    def test_no_bias(self):
        cfg = LinearNodeConfig(use_bias=False)
        node = self._make([None], cfg=cfg)
        assert node.linear.bias is None


class TestNodeMLP:

    def _make(self, n_preds=2, out_dims=None, cfg=None):
        out_dims = out_dims or [1] * n_preds
        predecessors = [f"p{i}" for i in range(n_preds)]
        rhs = "n(" + ",".join(predecessors) + ")"

        class FakeNode:
            def __init__(self, d): self.out_dim = d

        pred_nodes = [FakeNode(d) for d in out_dims]
        node = NodeMLP("y", rhs, predecessors, cfg, pred_nodes)
        return node

    def test_output_shape(self):
        assert self._make(2)([torch.randn(BS, 1)] * 2, BS).shape == (BS, 1)

    def test_multidim_predecessor(self):
        node = self._make(2, out_dims=[3, 2])
        assert node([torch.randn(BS, 3), torch.randn(BS, 2)], BS).shape == (BS, 1)

    def test_custom_cfg(self):
        cfg = MLPConfig(hidden_dims=[16], output_dim=2)
        assert self._make(1, cfg=cfg)([torch.randn(BS, 1)], BS).shape == (BS, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DAG — pure graph
# ═══════════════════════════════════════════════════════════════════════════════

class TestDAG:

    def test_basic_graph(self):
        dag = DAG("e ~ Normal(0,1); x <- e")
        assert dag.graph["e"] == []
        assert dag.graph["x"] == ["e"]

    def test_multi_lhs_expansion(self):
        dag = DAG("e1, e2, e3 ~ Normal(0,1)")
        assert {"e1", "e2", "e3"} == set(dag.graph.keys())

    def test_topological_order_valid(self):
        dag = DAG("e1, e2 ~ Normal(0,1); x <- e1 + e2; y <- x + e2")
        o = dag.topological_order
        assert o.index("e1") < o.index("x")
        assert o.index("e2") < o.index("x")
        assert o.index("x") < o.index("y")

    def test_static_order_returns_iterator(self):
        dag = DAG("e ~ Normal(0,1); x <- e")
        assert list(dag.static_order()) == dag.topological_order

    def test_no_pytorch_objects(self):
        dag = DAG("e ~ Normal(0,1); x <- e")
        for val in dag.graph.values():
            assert not isinstance(val, nn.Module)

    def test_mlp_predecessors_in_graph(self):
        dag = DAG("e1, e2 ~ Normal(0,1); y <- n(e1,e2)")
        assert dag.graph["y"] == ["e1", "e2"]

    def test_fixed_coeff_stripped_in_graph(self):
        dag = DAG("e ~ Normal(0,1); y <- 1*e")
        assert dag.graph["y"] == ["e"]

    def test_negative_coeff_stripped_in_graph(self):
        dag = DAG("e ~ Normal(0,1); y <- -1*e")
        assert dag.graph["y"] == ["e"]

    def test_fractional_coeff_stripped_in_graph(self):
        dag = DAG("e ~ Normal(0,1); y <- 0.5*e")
        assert dag.graph["y"] == ["e"]

    def test_mixed_coeffs_in_graph(self):
        dag = DAG("e1, e2 ~ Normal(0,1); y <- 1*e1 + e2")
        assert dag.graph["y"] == ["e1", "e2"]

    def test_mixed_linear_mlp_creates_intermediate(self):
        dag = DAG("e1, e2, e3 ~ Normal(0,1); y <- n(e1,e2) + e3")
        assert len([k for k in dag.graph if k.startswith("y:")]) == 1

    def test_nested_mlp_creates_intermediates(self):
        dag = DAG("e1, e2, e3 ~ Normal(0,1); y <- n(e1, n(e2,e3))")
        assert len([k for k in dag.graph if k.startswith("y:")]) >= 1

    def test_deeply_nested_mlp(self):
        dag = DAG("e1,e2,e3,e4 ~ Normal(0,1); y <- n(e1, n(e2, n(e3,e4)))")
        assert "y" in dag.graph

    def test_whitespace_ignored(self):
        dag1 = DAG("e ~ Normal(0,1); y <- e")
        dag2 = DAG("e ~ Normal( 0 , 1 ) ; y <- e")
        assert set(dag1.graph.keys()) == set(dag2.graph.keys())

    def test_cycle_raises_value_error(self):
        with pytest.raises(ValueError, match="cycle"):
            DAG("x <- y; y <- x")

    def test_invalid_formula_raises(self):
        with pytest.raises((ValueError, AttributeError, KeyError)):
            DAG("y ? e")

    def test_dag_standalone(self):
        dag = DAG("e1, e2 ~ Normal(0,1); x <- e1; y <- x + e2")
        assert len(dag.topological_order) == 4

    # ── NodeSpec ──────────────────────────────────────────────────────────────

    def test_node_spec_is_dag_nested_class(self):
        """NodeSpec is a DAG implementation detail — must be accessed via DAG.NodeSpec."""
        dag = DAG("e ~ Normal(0,1)")
        assert isinstance(dag.node_specs["e"], DAG.NodeSpec)

    def test_node_specs_keys_match_graph(self):
        dag = DAG("e1, e2 ~ Normal(0,1); y <- e1 + e2")
        assert set(dag.node_specs.keys()) == set(dag.graph.keys())

    def test_node_spec_exogenous(self):
        dag = DAG("e ~ Normal(0,1)")
        ns = dag.node_specs["e"]
        assert ns.node_type == "exogenous"
        assert ns.predecessors == []
        assert ns.coefficients == []

    def test_node_spec_linear(self):
        dag = DAG("e1, e2 ~ Normal(0,1); y <- 1*e1 + e2")
        ns = dag.node_specs["y"]
        assert ns.node_type == "linear"
        assert ns.predecessors == ["e1", "e2"]
        assert ns.coefficients == [1.0, None]

    def test_node_spec_mlp(self):
        dag = DAG("e1, e2 ~ Normal(0,1); y <- n(e1,e2)")
        ns = dag.node_specs["y"]
        assert ns.node_type == "mlp"
        assert ns.predecessors == ["e1", "e2"]
        assert ns.coefficients == []

    def test_node_spec_linear_negative_coeff(self):
        dag = DAG("e1, e2 ~ Normal(0,1); y <- -1*e1 + 0.5*e2")
        ns = dag.node_specs["y"]
        assert ns.coefficients == [-1.0, 0.5]

    def test_node_spec_intermediate_mlp(self):
        """Mixed formula creates an intermediate NodeSpec for the MLP part."""
        dag = DAG("e1, e2, e3 ~ Normal(0,1); y <- n(e1,e2) + e3")
        intermediates = [k for k in dag.node_specs if k.startswith("y:")]
        assert len(intermediates) == 1
        assert dag.node_specs[intermediates[0]].node_type == "mlp"

    def test_graph_derived_from_node_specs(self):
        """graph must be consistent with node_specs predecessors."""
        dag = DAG("e1, e2 ~ Normal(0,1); x <- e1; y <- x + e2")
        for name, ns in dag.node_specs.items():
            assert dag.graph[name] == ns.predecessors

    # ── _parse_linear_term ────────────────────────────────────────────────────

    def test_parse_no_prefix(self):
        dag = DAG("e ~ Normal(0,1)")
        assert dag._parse_linear_term("x") == (None, "x")

    def test_parse_coeff_one(self):
        dag = DAG("e ~ Normal(0,1)")
        assert dag._parse_linear_term("1*x") == (1.0, "x")

    def test_parse_coeff_negative(self):
        dag = DAG("e ~ Normal(0,1)")
        assert dag._parse_linear_term("-1*x") == (-1.0, "x")

    def test_parse_coeff_fractional(self):
        dag = DAG("e ~ Normal(0,1)")
        assert dag._parse_linear_term("0.5*x") == (0.5, "x")

    def test_parse_coeff_negative_fractional(self):
        dag = DAG("e ~ Normal(0,1)")
        assert dag._parse_linear_term("-0.5*x") == (-0.5, "x")

    def test_parse_coeff_multidigit(self):
        dag = DAG("e ~ Normal(0,1)")
        coeff, name = dag._parse_linear_term("10*x")
        assert coeff == 10.0 and name == "x"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SEM2 inheritance
# ═══════════════════════════════════════════════════════════════════════════════

class TestSEM2Inheritance:

    def test_is_dag(self):
        assert isinstance(make_model("e ~ Normal(0,1); y <- e"), DAG)

    def test_is_nn_module(self):
        assert isinstance(make_model("e ~ Normal(0,1); y <- e"), nn.Module)

    def test_dag_graph_matches_layers(self):
        m = make_model("e1, e2 ~ Normal(0,1); y <- e1 + e2")
        assert set(m.graph.keys()) == set(m.layer.keys())

    def test_dag_topological_order_matches_static_order(self):
        m = make_model("e1, e2 ~ Normal(0,1); x <- e1; y <- x + e2")
        assert m.topological_order == list(m.static_order())

    def test_dag_predecessors_match_node_predecessors(self):
        m = make_model("e1, e2 ~ Normal(0,1); y <- e1 + e2")
        for name, node in m.layer.items():
            assert m.graph[name] == node.predecessors

    def test_parameters_registered(self):
        m = make_model("e ~ Normal(0,1); y <- e")
        assert len(list(m.parameters())) > 0

    def test_unused_cfg_raises(self):
        with pytest.raises(AssertionError):
            make_model("e ~ Normal(0,1); y <- e",
                       cfgs={"nonexistent": LinearNodeConfig()})

    def test_node_exogenous_inherits_sem2node(self):
        """NodeExogenous must inherit from SEM2Node (via NodeExogenous).
        If you add a new node type, add a corresponding test here.
        """
        m = make_model("e ~ Normal(0,1)")
        assert isinstance(m.layer["e"], SEM2Node), (
            f"NodeExogenous does not inherit from SEM2Node"
        )

    def test_node_linear_inherits_sem2node(self):
        """NodeLinear must inherit from SEM2Node.
        If you add a new node type, add a corresponding test here.
        """
        m = make_model("e ~ Normal(0,1); y <- e")
        node = m.layer["y"]
        assert isinstance(node, SEM2Node), (
            f"NodeLinear does not inherit from SEM2Node"
        )

    def test_node_mlp_inherits_sem2node(self):
        """NodeMLP must inherit from SEM2Node.
        If you add a new node type, add a corresponding test here.
        """
        m = make_model("e ~ Normal(0,1); y <- n(e)")
        node = m.layer["y"]
        assert isinstance(node, SEM2Node), (
            f"NodeMLP does not inherit from SEM2Node"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Formula parser (via SEM2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormulaParser:

    def test_simple_linear(self):
        m = make_model("e ~ Normal(0,1); x <- e")
        assert "e" in m.layer and "x" in m.layer

    def test_multi_lhs_expansion(self):
        m = make_model("e1, e2, e3 ~ Normal(0,1)")
        assert {"e1", "e2", "e3"} == set(m.layer.keys())

    def test_linear_node_type(self):
        m = make_model("e1, e2 ~ Normal(0,1); y <- e1 + e2")
        assert isinstance(m.layer["y"], NodeLinear)

    def test_fixed_coeff_one_parsed(self):
        m = make_model("e ~ Normal(0,1); y <- 1*e")
        assert m.layer["y"].fix_weights[0, 0].item() is True

    def test_fixed_coeff_negative_parsed(self):
        m = make_model("e ~ Normal(0,1); y <- -1*e")
        assert m.layer["y"].fix_weights[0, 0].item() is True
        assert m.layer["y"]._fixed_vals[0, 0].item() == -1.0

    def test_fixed_coeff_fractional_parsed(self):
        m = make_model("e ~ Normal(0,1); y <- 0.5*e")
        assert m.layer["y"]._fixed_vals[0, 0].item() == pytest.approx(0.5)

    def test_no_coeff_is_learned(self):
        m = make_model("e ~ Normal(0,1); y <- e")
        assert m.layer["y"].fix_weights[0, 0].item() is False

    def test_mlp_node_type(self):
        m = make_model("e1, e2 ~ Normal(0,1); y <- n(e1,e2)")
        assert isinstance(m.layer["y"], NodeMLP)

    def test_mixed_linear_mlp(self):
        m = make_model("e1, e2, e3 ~ Normal(0,1); y <- n(e1,e2) + e3")
        assert isinstance(m.layer["y"], NodeLinear)
        mlp_nodes = [k for k in m.layer if k.startswith("y:")]
        assert len(mlp_nodes) == 1
        assert isinstance(m.layer[mlp_nodes[0]], NodeMLP)

    def test_nested_mlp(self):
        m = make_model("e1, e2, e3 ~ Normal(0,1); y <- n(e1, n(e2,e3))")
        assert len([k for k in m.layer if k.startswith("y:")]) >= 1

    def test_deeply_nested_mlp(self):
        m = make_model("e1,e2,e3,e4 ~ Normal(0,1); y <- n(e1, n(e2, n(e3,e4)))")
        assert "y" in m.layer

    def test_whitespace_ignored(self):
        m1 = make_model("e ~ Normal(0,1); y <- e")
        m2 = make_model("e ~ Normal( 0 , 1 ) ; y <- e")
        assert set(m1.layer.keys()) == set(m2.layer.keys())

    def test_unknown_cfg_raises(self):
        with pytest.raises(AssertionError):
            make_model("e ~ Normal(0,1); y <- e", cfgs={"bad": LinearNodeConfig()})

    def test_topological_order_respected(self):
        m = make_model("e1, e2 ~ Normal(0,1); x <- e1 + e2; y <- x + e2")
        o = m.topological_order
        assert o.index("e1") < o.index("x")
        assert o.index("e2") < o.index("x")
        assert o.index("x") < o.index("y")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SEM2 forward pass
# ═══════════════════════════════════════════════════════════════════════════════

class TestSEM2Forward:

    def test_return_dict_keys(self):
        m = make_model("e ~ Normal(0,1); y <- e")
        assert set(run(m).keys()) == {"e", "y"}

    def test_return_tensor_shape(self):
        m = make_model("e ~ Normal(0,1); y <- e")
        assert run(m, bs=BS, return_dict=False).shape == (BS, 2)

    def test_batch_size_respected(self):
        m = make_model("e ~ Normal(0,1); y <- e")
        for bs in [1, 4, 32]:
            assert run(m, bs=bs)["y"].shape == (bs, 1)

    def test_set_nodes_overrides_value(self):
        m = make_model("e ~ Normal(0,1); y <- e")
        fixed = torch.zeros(BS, 1)
        assert torch.allclose(run(m, set_nodes={"e": fixed})["e"], fixed)

    def test_set_nodes_invalid_key_raises(self):
        m = make_model("e ~ Normal(0,1); y <- e")
        with pytest.raises(AssertionError):
            run(m, set_nodes={"bad": torch.zeros(BS, 1)})

    def test_fixed_coeff_one_passthrough(self):
        """y <- 1*e with no bias — output should equal input."""
        m = make_model("e ~ Normal(0,1); y <- 1*e",
                       cfgs={"y": LinearNodeConfig(use_bias=False)})
        e_val = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        out = run(m, set_nodes={"e": e_val})
        assert torch.allclose(out["y"], e_val)

    def test_fixed_coeff_negative(self):
        """y <- -1*e with no bias — output should be negated input."""
        m = make_model("e ~ Normal(0,1); y <- -1*e",
                       cfgs={"y": LinearNodeConfig(use_bias=False)})
        e_val = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        out = run(m, set_nodes={"e": e_val})
        assert torch.allclose(out["y"], -e_val)

    def test_fixed_coeff_fractional(self):
        """y <- 0.5*e with no bias — output should be halved input."""
        m = make_model("e ~ Normal(0,1); y <- 0.5*e",
                       cfgs={"y": LinearNodeConfig(use_bias=False)})
        e_val = torch.ones(BS, 1)
        out = run(m, set_nodes={"e": e_val})
        assert torch.allclose(out["y"], e_val * 0.5)

    def test_fixed_sum(self):
        """y <- 1*e1 + -1*e2 — should be e1 - e2."""
        m = make_model("e1, e2 ~ Normal(0,1); y <- 1*e1 + -1*e2",
                       cfgs={"y": LinearNodeConfig(use_bias=False)})
        e1 = torch.ones(BS, 1) * 3
        e2 = torch.ones(BS, 1) * 1
        out = run(m, set_nodes={"e1": e1, "e2": e2})
        assert torch.allclose(out["y"], torch.ones(BS, 1) * 2)

    def test_multi_output_node(self):
        m = make_model("e ~ Normal(0,1); y <- e",
                       cfgs={"y": LinearNodeConfig(output_dim=3)})
        assert run(m)["y"].shape == (BS, 3)

    def test_multi_output_predecessor_feeds_downstream(self):
        m = make_model("e ~ Normal(0,1); x <- e; y <- x",
                       cfgs={"x": LinearNodeConfig(output_dim=2),
                              "y": LinearNodeConfig()})
        out = run(m)
        assert out["x"].shape == (BS, 2)
        assert out["y"].shape == (BS, 1)

    def test_fix_coeff_with_multi_output_predecessor(self):
        """1*x where x has out_dim=2 — both columns must use fixed value 1."""
        m = make_model("e ~ Normal(0,1); x <- e; y <- 1*x",
                       cfgs={"x": LinearNodeConfig(output_dim=2),
                              "y": LinearNodeConfig(use_bias=False)})
        node_y = m.layer["y"]
        with torch.no_grad():
            node_y.linear.weight.fill_(99.0)
        out = node_y([torch.ones(1, 2)], batch_size=1)
        assert torch.allclose(out, torch.tensor([[2.0]]))  # 1*1 + 1*1

    def test_mlp_forward(self):
        m = make_model("e1, e2 ~ Normal(0,1); y <- n(e1,e2)")
        assert run(m)["y"].shape == (BS, 1)

    def test_chain_graph(self):
        m = make_model("e ~ Normal(0,1); x1 <- e; x2 <- x1; x3 <- x2")
        assert all(v.shape == (BS, 1) for v in run(m).values())

    def test_diamond_graph(self):
        m = make_model("e ~ Normal(0,1); x1 <- e; x2 <- e; y <- x1 + x2")
        assert run(m)["y"].shape == (BS, 1)

    def test_gradients_flow(self):
        cfg = MLPConfig(hidden_dims=[8], output_dim=1)
        m = make_model("e ~ Normal(0,1); y <- n(e)", cfgs={"y": cfg})
        out = run(m, set_nodes={"e": torch.randn(BS, 1)})
        out["y"].sum().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_intervention_does_not_corrupt_upstream(self):
        m = make_model("e ~ Normal(0,1); x <- e; y <- x")
        out = run(m, set_nodes={"y": torch.zeros(BS, 1)})
        assert out["x"].shape == (BS, 1)
        assert torch.allclose(out["y"], torch.zeros(BS, 1))


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_single_exogenous_node(self):
        assert run(make_model("e ~ Normal(0,1)"))["e"].shape == (BS, 1)

    def test_batch_size_one(self):
        assert run(make_model("e ~ Normal(0,1); y <- e"), bs=1)["y"].shape == (1, 1)

    def test_large_batch(self):
        assert run(make_model("e ~ Normal(0,1); y <- e"), bs=1024)["y"].shape == (1024, 1)

    def test_exogenous_out_dim(self):
        m = make_model("e ~ Normal(0,1); y <- e", cfgs={"e": 3})
        assert run(m)["e"].shape == (BS, 3)

    def test_get_set_reset_value(self):
        m = make_model("e ~ Normal(0,1)")
        n = m.layer["e"]
        t = torch.tensor([[1.0]])
        n.set_value(t)
        assert torch.allclose(n.get_value(), t)
        n.reset_value()
        assert n.get_value() is None

    def test_repr(self):
        m = make_model("e ~ Normal(0,1)")
        assert "e" in repr(m.layer["e"])

    def test_mixed_fixed_learned_coefficients(self):
        m = make_model("e1, e2 ~ Normal(0,1); y <- 1*e1 + e2")
        node = m.layer["y"]
        assert node.fix_weights[0, 0].item() is True
        assert node.fix_weights[0, 1].item() is False

# ═══════════════════════════════════════════════════════════════════════════════
# 9. Class hierarchy
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassHierarchy:
    """Verify the RandomVariable -> SEM2Node -> Node* inheritance chain."""

    def test_random_variable_is_abstract(self):
        """RandomVariable cannot be instantiated directly."""
        import inspect
        assert inspect.isabstract(RandomVariable) or hasattr(RandomVariable, '__abstractmethods__')

    def test_sem2node_inherits_random_variable(self):
        assert issubclass(SEM2Node, RandomVariable)

    def test_node_exogenous_torch_is_concrete(self):
        """NodeExogenous is the concrete torch implementation."""
        m = make_model("e ~ Normal(0,1)")
        node = m.layer["e"]
        assert type(node) is NodeExogenous

    def test_node_exogenous_torch_inherits_chain(self):
        m = make_model("e ~ Normal(0,1)")
        node = m.layer["e"]
        assert isinstance(node, NodeExogenous)
        assert isinstance(node, NodeExogenous)
        assert isinstance(node, SEM2Node)
        assert isinstance(node, RandomVariable)
        assert isinstance(node, nn.Module)

    def test_node_linear_inherits_chain(self):
        m = make_model("e ~ Normal(0,1); y <- e")
        node = m.layer["y"]
        assert isinstance(node, NodeLinear)
        assert isinstance(node, NodeEndogenous)
        assert isinstance(node, SEM2Node)
        assert isinstance(node, RandomVariable)
        assert isinstance(node, nn.Module)

    def test_node_linear_dist_is_none(self):
        """Endogenous nodes have no distribution — dist stays None."""
        m = make_model("e ~ Normal(0,1); y <- e")
        assert m.layer["y"].dist is None

    def test_node_mlp_inherits_chain(self):
        m = make_model("e ~ Normal(0,1); y <- n(e)")
        node = m.layer["y"]
        assert isinstance(node, NodeMLP)
        assert isinstance(node, NodeNonlinear)
        assert isinstance(node, NodeEndogenous)
        assert isinstance(node, SEM2Node)
        assert isinstance(node, RandomVariable)

    def test_node_nonlinear_is_abstract(self):
        import inspect
        assert inspect.isabstract(NodeNonlinear)

    def test_node_endogenous_is_abstract(self):
        import inspect
        assert inspect.isabstract(NodeEndogenous)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Formula parsing robustness
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormulaParsingRobustness:
    """Verify that malformed exogenous formulas raise clear, descriptive errors."""

    def test_unknown_distribution_raises(self):
        """An unrecognised distribution name should raise ValueError, not AttributeError."""
        with pytest.raises(ValueError, match="Unknown distribution"):
            DAG("e ~ NotARealDist(0,1)")

    def test_malformed_rhs_no_parens_raises(self):
        """Missing parentheses should raise ValueError."""
        with pytest.raises(ValueError, match="Could not parse"):
            DAG("e ~ Normal")

    def test_non_numeric_param_raises(self):
        """Non-numeric distribution parameters should raise ValueError."""
        with pytest.raises(ValueError, match="Non-numeric"):
            DAG("e ~ Normal(a,b)")

    def test_typo_suggestion_in_error(self):
        """A near-miss distribution name should suggest the correct one."""
        try:
            DAG("e ~ Normla(0,1)")
            assert False, "Should have raised"
        except ValueError as exc:
            # The error should mention the typo'd name
            assert "Normla" in str(exc)

    def test_valid_distributions_parse(self):
        """A sample of valid distributions should parse without error."""
        for formula in [
            "e ~ Normal(0,1)",
            "e ~ Beta(0.5,0.5)",
            "e ~ Bernoulli(0.3)",
            "e ~ Poisson(2.0)",
            "e ~ Exponential(1.0)",
        ]:
            dag = DAG(formula)
            ns = dag.node_specs["e"]
            assert ns.dist_name is not None
            assert ns.dist_params is not None

    def test_dist_name_stored_on_node_spec(self):
        dag = DAG("e ~ Normal(0,1)")
        assert dag.node_specs["e"].dist_name == "Normal"

    def test_dist_params_stored_on_node_spec(self):
        dag = DAG("e ~ Normal(0,1)")
        assert dag.node_specs["e"].dist_params == ["0", "1"]

    def test_beta_params_stored(self):
        dag = DAG("e ~ Beta(0.5,0.5)")
        assert dag.node_specs["e"].dist_params == ["0.5", "0.5"]


class TestSEMTypeDetection:
    """Verify sem_type is correctly inferred from the formula."""

    def test_linear_gaussian(self):
        dag = DAG("e ~ Normal(0,1); y <- e")
        assert dag.sem_type == SEMType.LINEAR_GAUSSIAN

    def test_linear_gaussian_multi_node(self):
        dag = DAG("e1, e2 ~ Normal(0,1); y <- e1 + e2")
        assert dag.sem_type == SEMType.LINEAR_GAUSSIAN

    def test_linear_non_gaussian(self):
        dag = DAG("e ~ Beta(0.5,0.5); y <- e")
        assert dag.sem_type == SEMType.LINEAR

    def test_nonlinear(self):
        dag = DAG("e1, e2 ~ Normal(0,1); y <- n(e1,e2)")
        assert dag.sem_type == SEMType.NONLINEAR

    def test_mixed_functional_form_is_nonlinear(self):
        """Linear + MLP in same model → nonlinear (MLP supersedes linear)."""
        dag = DAG("e1, e2, e3 ~ Normal(0,1); y <- n(e1,e2) + e3")
        assert dag.sem_type == SEMType.NONLINEAR

    def test_discrete(self):
        dag = DAG("e ~ Bernoulli(0.5); y <- e")
        assert dag.sem_type == SEMType.LINEAR_DISCRETE

    def test_discrete_poisson(self):
        dag = DAG("e ~ Poisson(2.0); y <- e")
        assert dag.sem_type == SEMType.LINEAR_DISCRETE

    def test_mixed_discrete(self):
        dag = DAG("e ~ Bernoulli(0.5); y <- n(e)")
        assert dag.sem_type == SEMType.NONLINEAR_DISCRETE

    def test_exogenous_only_gaussian(self):
        dag = DAG("e1, e2 ~ Normal(0,1)")
        assert dag.sem_type == SEMType.LINEAR_GAUSSIAN

    def test_sem2_inherits_sem_type(self):
        """SEM2 exposes the same sem_type as the underlying DAG."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = make_model("e ~ Normal(0,1); y <- e")
        assert m.sem_type == SEMType.LINEAR_GAUSSIAN

    def test_sem2_init_emits_warning(self):
        """SEM2.__init__ should warn the user about the detected sem_type."""
        with pytest.warns(UserWarning, match="sem_type"):
            SEM2("e ~ Normal(0,1); y <- e")


    def test_nonlinear_discrete(self):
        dag = DAG("e ~ Bernoulli(0.5); y <- n(e)")
        assert dag.sem_type == SEMType.NONLINEAR_DISCRETE

    def test_linear_mixed_exog(self):
        """Some continuous + some discrete exog nodes, linear endogenous."""
        dag = DAG("e1 ~ Normal(0,1); e2 ~ Bernoulli(0.5); y <- e1 + e2")
        assert dag.sem_type == SEMType.LINEAR_MIXED_EXOG

    def test_nonlinear_mixed_exog(self):
        """Some continuous + some discrete exog nodes, MLP endogenous."""
        dag = DAG("e1 ~ Normal(0,1); e2 ~ Bernoulli(0.5); y <- n(e1, e2)")
        assert dag.sem_type == SEMType.NONLINEAR_MIXED_EXOG

    def test_exogenous_only_non_gaussian(self):
        dag = DAG("e1, e2 ~ Beta(0.5,0.5)")
        assert dag.sem_type == SEMType.LINEAR

    def test_exogenous_only_discrete(self):
        dag = DAG("e ~ Bernoulli(0.3)")
        assert dag.sem_type == SEMType.LINEAR_DISCRETE

    def test_nonlinear_continuous(self):
        """Pure MLP with continuous noise is nonlinear."""
        dag = DAG("e ~ Normal(0,1); y <- n(e)")
        assert dag.sem_type == SEMType.NONLINEAR


# ═══════════════════════════════════════════════════════════════════════════════
# 11. register_buffer correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegisterBuffer:
    """Verify that NodeLinear's static tensors are proper buffers."""

    def _make_linear_node(self, coefficients, out_dims=None):
        n_preds = len(coefficients)
        out_dims = out_dims or [1] * n_preds
        predecessors = [f"p{i}" for i in range(n_preds)]

        class FakeNode:
            def __init__(self, d): self.out_dim = d

        pred_nodes = [FakeNode(d) for d in out_dims]
        return NodeLinear("y", "+".join(predecessors), predecessors, coefficients, None, pred_nodes)

    def test_fix_weights_is_buffer(self):
        """fix_weights must be a registered buffer, not a plain attribute."""
        node = self._make_linear_node([1.0, None])
        assert "fix_weights" in dict(node.named_buffers())

    def test_fixed_vals_is_buffer(self):
        node = self._make_linear_node([1.0, None])
        assert "_fixed_vals" in dict(node.named_buffers())

    def test_ones_is_buffer(self):
        node = self._make_linear_node([None])
        assert "_ones" in dict(node.named_buffers())

    def test_buffers_not_parameters(self):
        """Buffers must not appear in named_parameters (they are not learnable)."""
        node = self._make_linear_node([1.0, None])
        param_names = {n for n, _ in node.named_parameters()}
        assert "fix_weights" not in param_names
        assert "_fixed_vals" not in param_names
        assert "_ones" not in param_names

    def test_buffers_move_with_to_cpu(self):
        """model.to('cpu') must move buffers — smoke test on CPU (always available)."""
        node = self._make_linear_node([1.0, None])
        node.to("cpu")
        for name, buf in node.named_buffers():
            assert buf.device.type == "cpu", f"Buffer '{name}' not on cpu"
