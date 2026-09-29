"""
Test suite for SEM2.
Run with:  python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn

from sem2.model.components.nn_models import _LinearConfig, _LinearModel, _MLP
from sem2.dataloader.data import DataBatch, VariableInfo
from sem2.default_configs import LinearNodeConfig, MLPConfig
from sem2.model.components.nodes import (
    RandomVariable, SEM2Node,
    NodeExogenous,
    NodeEndogenous, NodeLinear, NodeNonlinear, NodeMLP,
)
from sem2.model.models import SEM2
from sem2.model.construction import _DAG
from sem2.model.construction import SEMType
from sem2.model.construction import FormulaParser
from sem2.model.construction import NodeSpec

#probably should breakdown into smaller test files but for now is fine I guess

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

BS = 4


def make_model(formula, cfgs=None, device="cpu"):
    return SEM2(formula, node_cfgs=cfgs, device=device)


def batch(values, *names):
    return DataBatch(values, tuple(VariableInfo(name) for name in names))


def run(model, bs=BS, set_nodes=None):
    res, _ = model.sample(batch_size=bs, set_nodes=set_nodes)
    return res


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
        assert m.node_dict["e"]([], batch_size=BS).shape == (BS, 1)

    def test_multidim_shape(self):
        m = make_model("e ~ Normal(0,1)", cfgs={"e": 3})
        assert m.node_dict["e"]([], batch_size=BS).shape == (BS, 3)

    def test_beta_range(self):
        m = make_model("b ~ Beta(0.5,0.5)")
        out = m.node_dict["b"]([], batch_size=100)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_no_predecessors(self):
        m = make_model("e ~ Normal(0,1)")
        assert m.node_dict["e"].predecessors == []

    def test_is_node_exogenous(self):
        """Concrete exogenous node must be NodeExogenous."""
        m = make_model("e ~ Normal(0,1)")
        assert isinstance(m.node_dict["e"], NodeExogenous)



    def test_dist_attribute_set(self):
        """NodeExogenous must have a dist attribute after construction."""
        m = make_model("e ~ Normal(0,1)")
        assert m.node_dict["e"].dist is not None


class TestNodeLinear:

    def _make(self, coefficients, cfg=None, out_dims=None):
        """Build a NodeLinear with fake predecessor nodes."""
        n_preds = len(coefficients)
        out_dims = out_dims or [1] * n_preds

        class FakeNode:
            def __init__(self, name, d):
                self.name = name
                self.out_dim = d

        pred_nodes = [FakeNode(f"p{i}", d) for i, d in enumerate(out_dims)]
        node = NodeLinear("y", "+".join(f"p{i}" for i in range(n_preds)), coefficients, cfg, pred_nodes)
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
            def __init__(self, name, d):
                self.name = name
                self.out_dim = d

        pred_nodes = [FakeNode(f"p{i}", d) for i, d in enumerate(out_dims)]
        node = NodeMLP("y", rhs, cfg, pred_nodes)
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
        dag = _DAG("e ~ Normal(0,1); x <- e")
        assert dag.graph["e"] == []
        assert dag.graph["x"] == ["e"]

    def test_multi_lhs_expansion(self):
        dag = _DAG("e1, e2, e3 ~ Normal(0,1)")
        assert {"e1", "e2", "e3"} == set(dag.graph.keys())

    def test_topological_order_valid(self):
        dag = _DAG("e1, e2 ~ Normal(0,1); x <- e1 + e2; y <- x + e2")
        o = dag.topological_order
        assert o.index("e1") < o.index("x")
        assert o.index("e2") < o.index("x")
        assert o.index("x") < o.index("y")

    def test_static_order_returns_iterator(self):
        dag = _DAG("e ~ Normal(0,1); x <- e")
        assert list(dag.static_order()) == dag.topological_order

    def test_no_pytorch_objects(self):
        dag = _DAG("e ~ Normal(0,1); x <- e")
        for val in dag.graph.values():
            assert not isinstance(val, nn.Module)

    def test_mlp_predecessors_in_graph(self):
        dag = _DAG("e1, e2 ~ Normal(0,1); y <- n(e1,e2)")
        assert dag.graph["y"] == ["e1", "e2"]

    def test_fixed_coeff_stripped_in_graph(self):
        dag = _DAG("e ~ Normal(0,1); y <- 1*e")
        assert dag.graph["y"] == ["e"]

    def test_negative_coeff_stripped_in_graph(self):
        dag = _DAG("e ~ Normal(0,1); y <- -1*e")
        assert dag.graph["y"] == ["e"]

    def test_fractional_coeff_stripped_in_graph(self):
        dag = _DAG("e ~ Normal(0,1); y <- 0.5*e")
        assert dag.graph["y"] == ["e"]

    def test_mixed_coeffs_in_graph(self):
        dag = _DAG("e1, e2 ~ Normal(0,1); y <- 1*e1 + e2")
        assert dag.graph["y"] == ["e1", "e2"]

    def test_mixed_linear_mlp_creates_intermediate(self):
        dag = _DAG("e1, e2, e3 ~ Normal(0,1); y <- n(e1,e2) + e3")
        assert len([k for k in dag.graph if k.startswith("y:")]) == 1

    def test_nested_mlp_creates_intermediates(self):
        dag = _DAG("e1, e2, e3 ~ Normal(0,1); y <- n(e1, n(e2,e3))")
        assert len([k for k in dag.graph if k.startswith("y:")]) >= 1

    def test_deeply_nested_mlp(self):
        dag = _DAG("e1,e2,e3,e4 ~ Normal(0,1); y <- n(e1, n(e2, n(e3,e4)))")
        assert "y" in dag.graph

    def test_whitespace_ignored(self):
        dag1 = _DAG("e ~ Normal(0,1); y <- e")
        dag2 = _DAG("e ~ Normal( 0 , 1 ) ; y <- e")
        assert set(dag1.graph.keys()) == set(dag2.graph.keys())

    def test_cycle_raises_value_error(self):
        with pytest.raises(ValueError, match="cycle"):
            _DAG("x <- y; y <- x")

    def test_invalid_formula_raises(self):
        with pytest.raises((ValueError, AttributeError, KeyError)):
            _DAG("y ? e")

    def test_dag_standalone(self):
        dag = _DAG("e1, e2 ~ Normal(0,1); x <- e1; y <- x + e2")
        assert len(dag.topological_order) == 4

    # ── NodeSpec ──────────────────────────────────────────────────────────────

    def test_node_spec_is_dag_nested_class(self):
        """NodeSpec is the shared lowered IR (defined in sem_ast); _DAG.NodeSpec
        is an alias to it for convenience."""
        dag = _DAG("e ~ Normal(0,1)")
        assert isinstance(dag.node_specs["e"], NodeSpec)
        assert _DAG.NodeSpec is NodeSpec

    def test_node_specs_keys_match_graph(self):
        dag = _DAG("e1, e2 ~ Normal(0,1); y <- e1 + e2")
        assert set(dag.node_specs.keys()) == set(dag.graph.keys())

    def test_node_spec_exogenous(self):
        dag = _DAG("e ~ Normal(0,1)")
        ns = dag.node_specs["e"]
        assert ns.node_type == "exogenous"
        assert ns.predecessors == []
        assert ns.coefficients == []

    def test_node_spec_linear(self):
        dag = _DAG("e1, e2 ~ Normal(0,1); y <- 1*e1 + e2")
        ns = dag.node_specs["y"]
        assert ns.node_type == "linear"
        assert ns.predecessors == ["e1", "e2"]
        assert ns.coefficients == [1.0, None]

    def test_node_spec_mlp(self):
        dag = _DAG("e1, e2 ~ Normal(0,1); y <- n(e1,e2)")
        ns = dag.node_specs["y"]
        assert ns.node_type == "mlp"
        assert ns.predecessors == ["e1", "e2"]
        assert ns.coefficients == []

    def test_node_spec_linear_negative_coeff(self):
        dag = _DAG("e1, e2 ~ Normal(0,1); y <- -1*e1 + 0.5*e2")
        ns = dag.node_specs["y"]
        assert ns.coefficients == [-1.0, 0.5]

    def test_node_spec_intermediate_mlp(self):
        """Mixed formula creates an intermediate NodeSpec for the MLP part."""
        dag = _DAG("e1, e2, e3 ~ Normal(0,1); y <- n(e1,e2) + e3")
        intermediates = [k for k in dag.node_specs if k.startswith("y:")]
        assert len(intermediates) == 1
        assert dag.node_specs[intermediates[0]].node_type == "mlp"

    def test_graph_derived_from_node_specs(self):
        """graph must be consistent with node_specs predecessors."""
        dag = _DAG("e1, e2 ~ Normal(0,1); x <- e1; y <- x + e2")
        for name, ns in dag.node_specs.items():
            assert dag.graph[name] == ns.predecessors

    # ── _parse_linear_term ────────────────────────────────────────────────────

    def test_parse_no_prefix(self):
        assert FormulaParser._parse_linear_term("x") == (None, "x")

    def test_parse_coeff_one(self):
        assert FormulaParser._parse_linear_term("1*x") == (1.0, "x")

    def test_parse_coeff_negative(self):
        assert FormulaParser._parse_linear_term("-1*x") == (-1.0, "x")

    def test_parse_coeff_fractional(self):
        assert FormulaParser._parse_linear_term("0.5*x") == (0.5, "x")

    def test_parse_coeff_negative_fractional(self):
        assert FormulaParser._parse_linear_term("-0.5*x") == (-0.5, "x")

    def test_parse_coeff_multidigit(self):
        coeff, name = FormulaParser._parse_linear_term("10*x")
        assert coeff == 10.0 and name == "x"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SEM2 inheritance
# ═══════════════════════════════════════════════════════════════════════════════

class TestSEM2Inheritance:

    def test_is_dag(self):
        assert isinstance(make_model("e ~ Normal(0,1); y <- e"), _DAG)

    def test_is_nn_module(self):
        assert isinstance(make_model("e ~ Normal(0,1); y <- e"), nn.Module)

    def test_dag_graph_matches_layers(self):
        m = make_model("e1, e2 ~ Normal(0,1); y <- e1 + e2")
        assert set(m.graph.keys()) == set(m.node_dict.keys())

    def test_dag_topological_order_matches_static_order(self):
        m = make_model("e1, e2 ~ Normal(0,1); x <- e1; y <- x + e2")
        assert m.topological_order == list(m.static_order())

    def test_dag_predecessors_match_node_predecessors(self):
        m = make_model("e1, e2 ~ Normal(0,1); y <- e1 + e2")
        for name, node in m.node_dict.items():
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
        assert isinstance(m.node_dict["e"], SEM2Node), (
            f"NodeExogenous does not inherit from SEM2Node"
        )

    def test_node_linear_inherits_sem2node(self):
        """NodeLinear must inherit from SEM2Node.
        If you add a new node type, add a corresponding test here.
        """
        m = make_model("e ~ Normal(0,1); y <- e")
        node = m.node_dict["y"]
        assert isinstance(node, SEM2Node), (
            f"NodeLinear does not inherit from SEM2Node"
        )

    def test_node_mlp_inherits_sem2node(self):
        """NodeMLP must inherit from SEM2Node.
        If you add a new node type, add a corresponding test here.
        """
        m = make_model("e ~ Normal(0,1); y <- n(e)")
        node = m.node_dict["y"]
        assert isinstance(node, SEM2Node), (
            f"NodeMLP does not inherit from SEM2Node"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Formula parser (via SEM2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormulaParser:

    def test_simple_linear(self):
        m = make_model("e ~ Normal(0,1); x <- e")
        assert "e" in m.node_dict and "x" in m.node_dict

    def test_multi_lhs_expansion(self):
        m = make_model("e1, e2, e3 ~ Normal(0,1)")
        assert {"e1", "e2", "e3"} == set(m.node_dict.keys())

    def test_linear_node_type(self):
        m = make_model("e1, e2 ~ Normal(0,1); y <- e1 + e2")
        assert isinstance(m.node_dict["y"], NodeLinear)

    def test_fixed_coeff_one_parsed(self):
        m = make_model("e ~ Normal(0,1); y <- 1*e")
        assert m.node_dict["y"].fix_weights[0, 0].item() is True

    def test_fixed_coeff_negative_parsed(self):
        m = make_model("e ~ Normal(0,1); y <- -1*e")
        assert m.node_dict["y"].fix_weights[0, 0].item() is True
        assert m.node_dict["y"]._fixed_vals[0, 0].item() == -1.0

    def test_fixed_coeff_fractional_parsed(self):
        m = make_model("e ~ Normal(0,1); y <- 0.5*e")
        assert m.node_dict["y"]._fixed_vals[0, 0].item() == pytest.approx(0.5)

    def test_no_coeff_is_learned(self):
        m = make_model("e ~ Normal(0,1); y <- e")
        assert m.node_dict["y"].fix_weights[0, 0].item() is False

    def test_mlp_node_type(self):
        m = make_model("e1, e2 ~ Normal(0,1); y <- n(e1,e2)")
        assert isinstance(m.node_dict["y"], NodeMLP)

    def test_mixed_linear_mlp(self):
        m = make_model("e1, e2, e3 ~ Normal(0,1); y <- n(e1,e2) + e3")
        assert isinstance(m.node_dict["y"], NodeLinear)
        mlp_nodes = [k for k in m.node_dict if k.startswith("y:")]
        assert len(mlp_nodes) == 1
        assert isinstance(m.node_dict[mlp_nodes[0]], NodeMLP)

    def test_nested_mlp(self):
        m = make_model("e1, e2, e3 ~ Normal(0,1); y <- n(e1, n(e2,e3))")
        assert len([k for k in m.node_dict if k.startswith("y:")]) >= 1

    def test_deeply_nested_mlp(self):
        m = make_model("e1,e2,e3,e4 ~ Normal(0,1); y <- n(e1, n(e2, n(e3,e4)))")
        assert "y" in m.node_dict

    def test_whitespace_ignored(self):
        m1 = make_model("e ~ Normal(0,1); y <- e")
        m2 = make_model("e ~ Normal( 0 , 1 ) ; y <- e")
        assert set(m1.node_dict.keys()) == set(m2.node_dict.keys())

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

    def test_return_batch_names(self):
        m = make_model("e ~ Normal(0,1); y <- e")
        assert set(run(m).names) == {"e", "y"}

    def test_return_tensor_shape(self):
        m = make_model("e ~ Normal(0,1); y <- e")
        assert run(m, bs=BS).values.shape == (BS, 2)

    def test_batch_size_respected(self):
        m = make_model("e ~ Normal(0,1); y <- e")
        for bs in [1, 4, 32]:
            assert run(m, bs=bs).select("y").shape == (bs, 1)

    def test_set_nodes_overrides_value(self):
        m = make_model("e ~ Normal(0,1); y <- e")
        fixed = torch.zeros(BS, 1)
        assert torch.allclose(run(m, set_nodes=batch(fixed, "e")).select("e"), fixed)

    def test_set_nodes_invalid_key_raises(self):
        m = make_model("e ~ Normal(0,1); y <- e")
        with pytest.raises(ValueError):
            run(m, set_nodes=batch(torch.zeros(BS, 1), "bad"))

    def test_fixed_coeff_one_passthrough(self):
        """y <- 1*e with no bias — output should equal input."""
        m = make_model("e ~ Normal(0,1); y <- 1*e",
                       cfgs={"y": LinearNodeConfig(use_bias=False)})
        e_val = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        out = run(m, set_nodes=batch(e_val, "e"))
        assert torch.allclose(out.select("y"), e_val)

    def test_fixed_coeff_negative(self):
        """y <- -1*e with no bias — output should be negated input."""
        m = make_model("e ~ Normal(0,1); y <- -1*e",
                       cfgs={"y": LinearNodeConfig(use_bias=False)})
        e_val = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        out = run(m, set_nodes=batch(e_val, "e"))
        assert torch.allclose(out.select("y"), -e_val)

    def test_fixed_coeff_fractional(self):
        """y <- 0.5*e with no bias — output should be halved input."""
        m = make_model("e ~ Normal(0,1); y <- 0.5*e",
                       cfgs={"y": LinearNodeConfig(use_bias=False)})
        e_val = torch.ones(BS, 1)
        out = run(m, set_nodes=batch(e_val, "e"))
        assert torch.allclose(out.select("y"), e_val * 0.5)

    def test_fixed_sum(self):
        """y <- 1*e1 + -1*e2 — should be e1 - e2."""
        m = make_model("e1, e2 ~ Normal(0,1); y <- 1*e1 + -1*e2",
                       cfgs={"y": LinearNodeConfig(use_bias=False)})
        e1 = torch.ones(BS, 1) * 3
        e2 = torch.ones(BS, 1) * 1
        out = run(m, set_nodes=batch(torch.cat([e1, e2], dim=-1), "e1", "e2"))
        assert torch.allclose(out.select("y"), torch.ones(BS, 1) * 2)

    def test_multi_output_node(self):
        m = make_model("e ~ Normal(0,1); y <- e",
                       cfgs={"y": LinearNodeConfig(output_dim=3)})
        assert run(m).select("y").shape == (BS, 3)

    def test_multi_output_predecessor_feeds_downstream(self):
        m = make_model("e ~ Normal(0,1); x <- e; y <- x",
                       cfgs={"x": LinearNodeConfig(output_dim=2),
                              "y": LinearNodeConfig()})
        out = run(m)
        assert out.select("x").shape == (BS, 2)
        assert out.select("y").shape == (BS, 1)

    def test_fix_coeff_with_multi_output_predecessor(self):
        """1*x where x has out_dim=2 — both columns must use fixed value 1."""
        m = make_model("e ~ Normal(0,1); x <- e; y <- 1*x",
                       cfgs={"x": LinearNodeConfig(output_dim=2),
                              "y": LinearNodeConfig(use_bias=False)})
        node_y = m.node_dict["y"]
        with torch.no_grad():
            node_y.linear.weight.fill_(99.0)
        out = node_y([torch.ones(1, 2)], batch_size=1)
        assert torch.allclose(out, torch.tensor([[2.0]]))  # 1*1 + 1*1

    def test_mlp_forward(self):
        m = make_model("e1, e2 ~ Normal(0,1); y <- n(e1,e2)")
        assert run(m).select("y").shape == (BS, 1)

    def test_chain_graph(self):
        m = make_model("e ~ Normal(0,1); x1 <- e; x2 <- x1; x3 <- x2")
        result = run(m)
        assert all(result.select(name).shape == (BS, 1) for name in result.names)

    def test_diamond_graph(self):
        m = make_model("e ~ Normal(0,1); x1 <- e; x2 <- e; y <- x1 + x2")
        assert run(m).select("y").shape == (BS, 1)

    def test_gradients_flow(self):
        cfg = MLPConfig(hidden_dims=[8], output_dim=1)
        m = make_model("e ~ Normal(0,1); y <- n(e)", cfgs={"y": cfg})
        out = run(m, set_nodes=batch(torch.randn(BS, 1), "e"))
        out.select("y").sum().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_intervention_does_not_corrupt_upstream(self):
        m = make_model("e ~ Normal(0,1); x <- e; y <- x")
        out = run(m, set_nodes=batch(torch.zeros(BS, 1), "y"))
        assert out.select("x").shape == (BS, 1)
        assert torch.allclose(out.select("y"), torch.zeros(BS, 1))


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_single_exogenous_node(self):
        assert run(make_model("e ~ Normal(0,1)")).select("e").shape == (BS, 1)

    def test_batch_size_one(self):
        assert run(make_model("e ~ Normal(0,1); y <- e"), bs=1).select("y").shape == (1, 1)

    def test_large_batch(self):
        assert run(make_model("e ~ Normal(0,1); y <- e"), bs=1024).select("y").shape == (1024, 1)

    def test_exogenous_out_dim(self):
        m = make_model("e ~ Normal(0,1); y <- e", cfgs={"e": 3})
        assert run(m).select("e").shape == (BS, 3)

    def test_get_set_reset_value(self):
        m = make_model("e ~ Normal(0,1)")
        n = m.node_dict["e"]
        t = torch.tensor([[1.0]])
        n.set_value(t)
        assert torch.allclose(n.get_value(), t)
        n.reset_value()
        assert n.get_value() is None

    def test_repr(self):
        m = make_model("e ~ Normal(0,1)")
        assert "e" in repr(m.node_dict["e"])

    def test_mixed_fixed_learned_coefficients(self):
        m = make_model("e1, e2 ~ Normal(0,1); y <- 1*e1 + e2")
        node = m.node_dict["y"]
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
        node = m.node_dict["e"]
        assert type(node) is NodeExogenous

    def test_node_exogenous_torch_inherits_chain(self):
        m = make_model("e ~ Normal(0,1)")
        node = m.node_dict["e"]
        assert isinstance(node, NodeExogenous)
        assert isinstance(node, NodeExogenous)
        assert isinstance(node, SEM2Node)
        assert isinstance(node, RandomVariable)
        assert isinstance(node, nn.Module)

    def test_node_linear_inherits_chain(self):
        m = make_model("e ~ Normal(0,1); y <- e")
        node = m.node_dict["y"]
        assert isinstance(node, NodeLinear)
        assert isinstance(node, NodeEndogenous)
        assert isinstance(node, SEM2Node)
        assert isinstance(node, RandomVariable)
        assert isinstance(node, nn.Module)

    def test_node_linear_dist_is_none(self):
        """Endogenous nodes have no distribution — dist stays None."""
        m = make_model("e ~ Normal(0,1); y <- e")
        assert m.node_dict["y"].dist is None

    def test_node_mlp_inherits_chain(self):
        m = make_model("e ~ Normal(0,1); y <- n(e)")
        node = m.node_dict["y"]
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
            _DAG("e ~ NotARealDist(0,1)")

    def test_malformed_rhs_no_parens_raises(self):
        """Missing parentheses should raise ValueError."""
        with pytest.raises(ValueError, match="Could not parse"):
            _DAG("e ~ Normal")

    def test_malformed_rhs_syntax_error_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not parse"):
            _DAG("e ~ Normal(0,")

    def test_undefined_parent_raises_clear_value_error(self):
        with pytest.raises(ValueError, match="undefined parent"):
            _DAG("y <- x")

    def test_non_numeric_param_raises(self):
        """Non-numeric distribution parameters should raise ValueError."""
        with pytest.raises(ValueError, match="Non-numeric"):
            _DAG("e ~ Normal(a,b)")

    def test_typo_suggestion_in_error(self):
        """A near-miss distribution name should suggest the correct one."""
        try:
            _DAG("e ~ Normla(0,1)")
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
            dag = _DAG(formula)
            ns = dag.node_specs["e"]
            assert ns.dist_name is not None
            assert ns.dist_params is not None

    def test_dist_name_stored_on_node_spec(self):
        dag = _DAG("e ~ Normal(0,1)")
        assert dag.node_specs["e"].dist_name == "Normal"

    def test_dist_params_stored_on_node_spec(self):
        dag = _DAG("e ~ Normal(0,1)")
        # dist_params is a parsed (args, kwargs) tuple, ready for instantiation.
        assert dag.node_specs["e"].dist_params == ([0.0, 1.0], {})

    def test_beta_params_stored(self):
        dag = _DAG("e ~ Beta(0.5,0.5)")
        assert dag.node_specs["e"].dist_params == ([0.5, 0.5], {})


class TestFormulaParserAST:
    """The parser builds a typed AST that lowers to NodeSpec objects."""

    def test_parse_returns_specs_and_inputs(self):
        specs, inputs = FormulaParser("e1, e2 ~ Normal(0,1); y <- e1 + e2").parse()
        assert set(specs.keys()) == {"e1", "e2", "y"}
        assert inputs == {"e1", "e2", "y"}

    def test_ast_statement_types(self):
        from sem2.model.construction import ExogenousDecl, EndogenousDecl
        parser = FormulaParser("e ~ Normal(0,1); y <- 1*e")
        parser.parse()
        kinds = [type(s).__name__ for s in parser.statements]
        assert kinds == ["ExogenousDecl", "EndogenousDecl"]

    def test_exogenous_ast_holds_distribution(self):
        from sem2.model.construction import ExogenousDecl
        parser = FormulaParser("e ~ Normal(0,1)")
        parser.parse()
        decl = parser.statements[0]
        assert isinstance(decl, ExogenousDecl)
        assert decl.dist.name == "Normal"
        assert decl.dist.args == [0.0, 1.0]

    def test_linear_ast_holds_terms(self):
        from sem2.model.construction import EndogenousDecl
        parser = FormulaParser("e1, e2 ~ Normal(0,1); y <- 1*e1 + e2")
        parser.parse()
        y_decl = parser.statements[-1]
        assert isinstance(y_decl, EndogenousDecl)
        assert y_decl.node_type == "linear"
        assert [(t.coeff, t.parent) for t in y_decl.terms] == [(1.0, "e1"), (None, "e2")]

    def test_mechanism_ast_holds_parents(self):
        parser = FormulaParser("e1, e2 ~ Normal(0,1); y <- n(e1,e2)")
        parser.parse()
        y_decl = parser.statements[-1]
        assert y_decl.node_type == "mlp"
        assert y_decl.mechanism.parents == ["e1", "e2"]

    def test_lowered_spec_matches_dag(self):
        """Specs produced standalone match those built via _DAG."""
        formula = "e1, e2 ~ Normal(0,1); y <- 0.5*e1 + -1*e2"
        specs, _ = FormulaParser(formula).parse()
        dag = _DAG(formula)
        assert specs["y"].coefficients == dag.node_specs["y"].coefficients
        assert specs["y"].predecessors == dag.node_specs["y"].predecessors

    def test_unobs_statement_sets_observed_vars(self):
        dag = _DAG("e ~ Normal(0,1); x <- e; y <- x; unobs(e)")
        assert dag.unobserved_vars == ["e"]
        assert dag.observed_vars == ["x", "y"]

    def test_unobs_statement_can_be_empty(self):
        dag = _DAG("e ~ Normal(0,1); x <- e; unobs()")
        assert dag.unobserved_vars == []
        assert dag.observed_vars == ["e", "x"]

    def test_unobs_statement_rejects_unknown_node(self):
        with pytest.raises(ValueError, match="Unobserved variables"):
            _DAG("unobs(z); e ~ Normal(0,1); x <- e")

    def test_unobs_statement_rejects_duplicates(self):
        with pytest.raises(ValueError, match="Duplicate unobserved"):
            _DAG("unobs(e,e); e ~ Normal(0,1); x <- e")


class TestDistributionRegistry:
    """Distribution-name validation lives in the parser frontend."""

    def test_known_distributions_accepted(self):
        for name in ["Normal", "Beta", "Bernoulli", "Poisson", "Uniform"]:
            FormulaParser(f"e ~ {name}(0.5)").parse()  # should not raise

    def test_unknown_distribution_rejected(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            FormulaParser("e ~ NotARealDist(0,1)").parse()

    def test_typo_suggestion(self):
        with pytest.raises(ValueError, match="Normal"):
            FormulaParser("e ~ Normla(0,1)").parse()


class TestSEMTypeDetection:
    """Verify sem_type is correctly inferred from the formula."""

    def test_linear_gaussian(self):
        dag = _DAG("e ~ Normal(0,1); y <- e")
        assert dag.inferred_sem_type == SEMType.LINEAR_GAUSSIAN

    def test_linear_gaussian_multi_node(self):
        dag = _DAG("e1, e2 ~ Normal(0,1); y <- e1 + e2")
        assert dag.inferred_sem_type == SEMType.LINEAR_GAUSSIAN

    def test_linear_non_gaussian(self):
        dag = _DAG("e ~ Beta(0.5,0.5); y <- e")
        assert dag.inferred_sem_type == SEMType.LINEAR

    def test_multivariate_normal_is_not_scalar_linear_gaussian(self):
        dag = _DAG("e ~ MultivariateNormal([0,0], [[1,0],[0,1]])")
        assert dag.inferred_sem_type == SEMType.LINEAR

    def test_nonlinear(self):
        dag = _DAG("e1, e2 ~ Normal(0,1); y <- n(e1,e2)")
        assert dag.inferred_sem_type == SEMType.NONLINEAR

    def test_mixed_functional_form_is_nonlinear(self):
        """Linear + MLP in same model → nonlinear (MLP supersedes linear)."""
        dag = _DAG("e1, e2, e3 ~ Normal(0,1); y <- n(e1,e2) + e3")
        assert dag.inferred_sem_type == SEMType.NONLINEAR

    # NOTE: "discrete" here refers to a discrete *exogenous distribution* (Bernoulli,
    # Poisson, ...). The current taxonomy does not give these a dedicated type; they
    # fold into LINEAR / NONLINEAR based on the endogenous nodes. (SEMType.DISCRETE is
    # reserved for discrete *endogenous* d(...) nodes.) A finer discrete-exog taxonomy
    # is intentionally not implemented yet.
    def test_discrete_exog_dist_is_linear(self):
        dag = _DAG("e ~ Bernoulli(0.5); y <- e")
        assert dag.inferred_sem_type == SEMType.LINEAR

    def test_poisson_exog_dist_is_linear(self):
        dag = _DAG("e ~ Poisson(2.0); y <- e")
        assert dag.inferred_sem_type == SEMType.LINEAR

    def test_discrete_exog_dist_is_nonlinear(self):
        dag = _DAG("e ~ Bernoulli(0.5); y <- n(e)")
        assert dag.inferred_sem_type == SEMType.NONLINEAR

    def test_exogenous_only_gaussian(self):
        dag = _DAG("e1, e2 ~ Normal(0,1)")
        assert dag.inferred_sem_type == SEMType.LINEAR_GAUSSIAN

    def test_sem2_inherits_sem_type(self):
        """SEM2 exposes the same sem_type as the underlying DAG."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = make_model("e ~ Normal(0,1); y <- e")
        assert m.inferred_sem_type == SEMType.LINEAR_GAUSSIAN

    def test_sem2_init_emits_warning(self):
        """SEM2.__init__ should warn the user about the detected sem_type."""
        with pytest.warns(UserWarning, match="sem_type"):
            SEM2("e ~ Normal(0,1); y <- e")


    def test_mixed_exog_dist_is_linear(self):
        """Mixed continuous + discrete exogenous dists, linear endogenous -> LINEAR."""
        dag = _DAG("e1 ~ Normal(0,1); e2 ~ Bernoulli(0.5); y <- e1 + e2")
        assert dag.inferred_sem_type == SEMType.LINEAR

    def test_mixed_exog_dist_is_nonlinear(self):
        """Mixed continuous + discrete exogenous dists, MLP endogenous -> NONLINEAR."""
        dag = _DAG("e1 ~ Normal(0,1); e2 ~ Bernoulli(0.5); y <- n(e1, e2)")
        assert dag.inferred_sem_type == SEMType.NONLINEAR

    def test_exogenous_only_non_gaussian(self):
        dag = _DAG("e1, e2 ~ Beta(0.5,0.5)")
        assert dag.inferred_sem_type == SEMType.LINEAR

    def test_exogenous_only_discrete_dist_is_linear(self):
        dag = _DAG("e ~ Bernoulli(0.3)")
        assert dag.inferred_sem_type == SEMType.LINEAR

    def test_nonlinear_continuous(self):
        """Pure MLP with continuous noise is nonlinear."""
        dag = _DAG("e ~ Normal(0,1); y <- n(e)")
        assert dag.inferred_sem_type == SEMType.NONLINEAR


# ═══════════════════════════════════════════════════════════════════════════════
# 11. register_buffer correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegisterBuffer:
    """Verify that NodeLinear's static tensors are proper buffers."""

    def _make_linear_node(self, coefficients, out_dims=None):
        n_preds = len(coefficients)
        out_dims = out_dims or [1] * n_preds

        class FakeNode:
            def __init__(self, name, d):
                self.name = name
                self.out_dim = d

        pred_nodes = [FakeNode(f"p{i}", d) for i, d in enumerate(out_dims)]
        return NodeLinear("y", "+".join(f"p{i}" for i in range(n_preds)), coefficients, None, pred_nodes)

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


# ═══════════════════════════════════════════════════════════════════════════════
# 12. SEM2Linear: coef_matrix / cov_matrix
# ═══════════════════════════════════════════════════════════════════════════════

from sem2.model import SEM2Linear
from sem2.operators.linear import (
    SEM2LinearModelOperator, LogLikelihood, WeightedFrobeniusDistance,
)


def _chain_linear():
    """x = e1; y = 2x + ey, all noise N(0,1), no bias. Cov[x,y] = [[1,2],[2,5]]."""
    return SEM2Linear(
        "e1, ey ~ Normal(0,1); x <- 1*e1; y <- 2*x + 1*ey",
        node_cfgs={"x": LinearNodeConfig(use_bias=False), "y": LinearNodeConfig(use_bias=False)},
    )


def _chain_linear_nongaussian():
    """x = e1; y = 2x + ey, all noise Beta(0.5,0.5), no bias."""
    return SEM2Linear(
        "e1, ey ~ Beta(0.5,0.5); x <- 1*e1; y <- 2*x + 1*ey",
        node_cfgs={"x": LinearNodeConfig(use_bias=False), "y": LinearNodeConfig(use_bias=False)},
    )


class TestSEM2Linear:

    def test_rejects_nonlinear_formula(self):
        with pytest.raises(ValueError, match="linear"):
            SEM2Linear("a, b ~ Normal(0,1); y <- n(a,b)",
                       node_cfgs={"y": MLPConfig(hidden_dims=[4], output_dim=1)})

    def test_accepts_linear_formula(self):
        m = _chain_linear()
        assert SEMType.is_linear(m.inferred_sem_type)

    def test_node_order_is_topological(self):
        m = _chain_linear()
        assert m.node_order == list(m.topological_order)

    def test_coef_matrix_lower_triangular(self):
        m = _chain_linear()
        B = m.coef_matrix()
        # strictly lower triangular under topological order
        assert torch.allclose(torch.triu(B), torch.zeros_like(B))

    def test_coef_matrix_values(self):
        m = _chain_linear()
        B = m.coef_matrix()
        i = m._node_index
        assert B[i["x"], i["e1"]].item() == pytest.approx(1.0)
        assert B[i["y"], i["x"]].item() == pytest.approx(2.0)
        assert B[i["y"], i["ey"]].item() == pytest.approx(1.0)

    def test_cov_matrix_full_symmetric(self):
        m = _chain_linear()
        S = m.cov_matrix()
        assert torch.allclose(S, S.T, atol=1e-6)

    def test_cov_matrix_observed_subset(self):
        m = _chain_linear()
        S = m.cov_matrix(["x", "y"])
        expected = torch.tensor([[1.0, 2.0], [2.0, 5.0]])
        assert torch.allclose(S, expected, atol=1e-5)

    def test_cov_matrix_matches_empirical(self):
        torch.manual_seed(0)
        m = _chain_linear()
        samp, _ = m.sample(batch_size=100000)
        data = samp.select(["x", "y"]).detach()
        emp = torch.cov(data.T)
        theo = m.cov_matrix(["x", "y"]).detach()
        assert torch.allclose(emp, theo, atol=0.1)

    def test_multi_output_node_raises(self):
        with pytest.raises(ValueError, match="scalar"):
            SEM2Linear("e ~ Normal(0,1); y <- e",
                       node_cfgs={"y": LinearNodeConfig(output_dim=2)})

    def test_multivariate_exogenous_distribution_raises(self):
        with pytest.raises(ValueError, match="scalar independent"):
            SEM2Linear("e ~ MultivariateNormal([0,0], [[1,0], [0,1]]); y <- e")

    def test_cov_matrix_works_for_non_gaussian_linear_second_moments(self):
        m = SEM2Linear(
            "e1, ey ~ Uniform(-1,1); x <- 1*e1; y <- 2*x + 1*ey",
            node_cfgs={"x": LinearNodeConfig(use_bias=False), "y": LinearNodeConfig(use_bias=False)},
        )
        S = m.cov_matrix(["x", "y"])
        expected = torch.tensor([[1 / 3, 2 / 3], [2 / 3, 5 / 3]])
        assert torch.allclose(S, expected, atol=1e-5)


class TestLinearOperators:

    def _data(self, m, n=5000, seed=0):
        torch.manual_seed(seed)
        samp, _ = m.sample(batch_size=n)
        return batch(samp.select(["x", "y"]).detach(), "x", "y")

    def test_base_guards_linearity(self):
        nl = SEM2("a, b ~ Normal(0,1); y <- n(a,b)",
                  node_cfgs={"y": MLPConfig(hidden_dims=[4], output_dim=1)})
        with pytest.raises(ValueError, match="linear"):
            LogLikelihood(nl, ["y"])

    def test_base_rejects_plain_sem2_linear_with_hint(self):
        with pytest.warns(UserWarning, match="sem_type"):
            m = SEM2("e ~ Normal(0,1); y <- e")
        with pytest.warns(UserWarning, match="special child class"):
            with pytest.raises(TypeError, match="SEM2Linear.*cov_matrix"):
                LogLikelihood(m, ["y"])

    def test_loglikelihood_matches_torch_mvn(self):
        m = _chain_linear()
        data = self._data(m)
        ll, _ = LogLikelihood(m, ["x", "y"], loc=torch.zeros(2)).evaluate(data)
        Sigma = m.cov_matrix(["x", "y"]).detach()
        mvn = torch.distributions.MultivariateNormal(torch.zeros(2), Sigma)
        assert ll.value.item() == pytest.approx(mvn.log_prob(data.values).mean().item(), abs=1e-4)

    def test_loglikelihood_uses_formula_observed_vars(self):
        m = SEM2Linear(
            "e1, ey ~ Normal(0,1); x <- 1*e1; y <- 2*x + 1*ey; unobs(e1,ey)",
            node_cfgs={"x": LinearNodeConfig(use_bias=False), "y": LinearNodeConfig(use_bias=False)},
        )
        data = self._data(m)
        implicit, _ = LogLikelihood(m, centered=True).evaluate(data)
        explicit, _ = LogLikelihood(m, ["x", "y"], centered=True).evaluate(data)
        assert implicit.value.item() == pytest.approx(explicit.value.item(), abs=1e-6)

    def test_loglikelihood_requires_observed_vars_without_formula_mapping(self):
        m = _chain_linear()
        with pytest.raises(ValueError, match="observed variables"):
            LogLikelihood(m)

    def test_centered_loglikelihood_matches_zero_loc(self):
        m = SEM2Linear("e ~ Normal(2,1); x <- 1*e; unobs(e)",
                       node_cfgs={"x": LinearNodeConfig(use_bias=False)})
        data = batch(torch.zeros(4, 1), "x")
        centered, _ = LogLikelihood(m, centered=True).evaluate(data)
        explicit, _ = LogLikelihood(m, loc=torch.zeros(1)).evaluate(data)
        model_mean, _ = LogLikelihood(m).evaluate(data)
        assert centered.value.item() == pytest.approx(explicit.value.item(), abs=1e-6)
        assert centered.value.item() > model_mean.value.item()

    def test_centered_and_loc_are_mutually_exclusive(self):
        m = SEM2Linear("e ~ Normal(0,1); x <- 1*e; unobs(e)",
                       node_cfgs={"x": LinearNodeConfig(use_bias=False)})
        with pytest.raises(ValueError, match="centered"):
            LogLikelihood(m, centered=True, loc=torch.zeros(1))

    def test_loglikelihood_warns_for_non_gaussian_linear(self):
        m = _chain_linear_nongaussian()
        with pytest.warns(UserWarning, match="pseudo-log-likelihood"):
            LogLikelihood(m, ["x", "y"])

    def test_loglikelihood_matches_mvn_formula_for_non_gaussian_linear(self):
        m = _chain_linear_nongaussian()
        data = self._data(m)
        with pytest.warns(UserWarning, match="finite second moments"):
            op = LogLikelihood(m, ["x", "y"])
        ll, metrics = op.evaluate(data)
        idx = torch.tensor(m._index(["x", "y"]))
        mu = m.mean_vector().detach().index_select(0, idx)
        Sigma = m.cov_matrix(["x", "y"]).detach()
        mvn = torch.distributions.MultivariateNormal(mu, Sigma)
        assert ll.value.item() == pytest.approx(mvn.log_prob(data.values).mean().item(), abs=1e-4)
        assert "log_likelihood" in metrics

    def test_loglikelihood_requires_finite_second_moments(self):
        m = SEM2Linear("e ~ Cauchy(0,1); y <- 1*e",
                       node_cfgs={"y": LinearNodeConfig(use_bias=False)})
        with pytest.warns(UserWarning, match="pseudo-log-likelihood"):
            op = LogLikelihood(m, ["y"], loc=torch.zeros(1))
        with pytest.raises(ValueError, match="finite second moments"):
            op.evaluate(batch(torch.zeros(3, 1), "y"))

    def test_weighted_frobenius_zero_on_model_data(self):
        m = _chain_linear()
        data = self._data(m, n=20000)
        loss, _ = WeightedFrobeniusDistance(m, ["x", "y"]).evaluate(data)
        assert loss.value.item() < 0.05

    def test_weighted_frobenius_weighted(self):
        m = _chain_linear()
        data = self._data(m)
        W = torch.ones(2, 2)
        unweighted, _ = WeightedFrobeniusDistance(m, ["x", "y"]).evaluate(data)
        weighted, _ = WeightedFrobeniusDistance(m, ["x", "y"], weight=W).evaluate(data)
        assert weighted.value.item() == pytest.approx(unweighted.value.item(), abs=1e-6)


    def test_gradients_flow_through_cov(self):
        m = _chain_linear()
        data = self._data(m)
        loss, _ = WeightedFrobeniusDistance(m, ["x", "y"]).evaluate(data)
        loss.value.backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert len(grads) > 0
