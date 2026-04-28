import pytest
import torch

from sem2 import SEM2, NodeDistribution, NodeLinear, NodeNN, LinearConfig
from nn_models import MLPConfig


# Does the parser handle all formula syntax correctly?
def test_formula_parsing():
    model = SEM2("x1, x2 ~ Normal(0,1); y <- x1 + x2; z <- n(x1,y)")
    order = list(model.static_order())
    assert set(order) == {"x1", "x2", "y", "z"}
    assert isinstance(model.layer["x1"], NodeDistribution)
    assert isinstance(model.layer["y"], NodeLinear)
    assert isinstance(model.layer["z"], NodeNN)
    assert model.layer["y"].predecessors == ["x1", "x2"]

    model2 = SEM2("u1, u2, u3 ~ Normal(0,1); y <- n(n(u1,u2), u3)")
    assert "y:0" in model2.layer
    assert isinstance(model2.layer["y:0"], NodeNN)

    model3 = SEM2("u1, u2, e ~ Normal(0,1); y <- n(u1,u2) + e")
    assert isinstance(model3.layer["y"], NodeLinear)
    assert "y:0" in model3.layer

    model4 = SEM2("e ~ Normal(0,1); y <- 1*e + e2; e2 ~ Normal(0,1)")
    assert model4.layer["y"].fix_weights[0] == True
    assert model4.layer["y"].fix_weights[1] == False

    with pytest.raises(ValueError):
        SEM2("x Normal(0,1)")
    with pytest.raises(AssertionError):
        SEM2("x ~ Normal(0,1)", node_cfgs={"nonexistent": 1})


# Does forward() produce correct shapes, respect set_nodes, and maintain topological order?
def test_forward_pass():
    model = SEM2("x1, x2 ~ Normal(0,1); y <- x1 + x2")
    sample = model(50)
    assert sample.shape == (50, 3)

    result = model(10, return_dict=True)
    assert set(result.keys()) == {"x1", "x2", "y"}
    assert result["x1"].shape == (10, 1)

    fixed_e = torch.ones(5, 1) * 42.0
    model2 = SEM2("e ~ Normal(0,1); y <- e")
    result2 = model2(5, set_nodes={"e": fixed_e}, return_dict=True)
    assert torch.allclose(result2["e"], fixed_e)

    with pytest.raises(AssertionError):
        model2(5, set_nodes={"nonexistent": torch.ones(5, 1)})

    torch.manual_seed(42)
    model3 = SEM2("x ~ Normal(0,1)")
    s1 = model3(10)
    torch.manual_seed(42)
    s2 = model3(10)
    assert torch.allclose(s1, s2)

    order = list(SEM2("a ~ Normal(0,1); b <- a; c <- b").static_order())
    assert order.index("a") < order.index("b") < order.index("c")


# Do node configs (out_dim, bias, init weights, MLP config, distributions) work as specified?
def test_node_configs():
    model = SEM2("x ~ Normal(0,1)", node_cfgs={"x": 2})
    assert model(10).shape == (10, 2)

    cfg_nobias = LinearConfig(use_bias=False)
    model2 = SEM2("e ~ Normal(0,1); y <- e", node_cfgs={"y": cfg_nobias})
    w, b = model2.layer["y"].weight_matrix()
    assert b is None

    init_w = torch.tensor([[2.0, 3.0]])
    cfg_init = LinearConfig(init_weight_val=init_w, use_bias=True)
    model3 = SEM2("e1, e2 ~ Normal(0,1); y <- e1 + e2", node_cfgs={"y": cfg_init})
    w3, _ = model3.layer["y"].weight_matrix()
    assert torch.allclose(w3, init_w)

    mlp_cfg = MLPConfig(input_dim=2, hidden_dims=[32], output_dim=1)
    model4 = SEM2("u1, u2 ~ Normal(0,1); y <- n(u1,u2)", node_cfgs={"y": mlp_cfg})
    assert model4(10).shape == (10, 3)

    model5 = SEM2("x ~ Exponential(1.0)")
    assert model5(10).shape[0] == 10
    model6 = SEM2("x ~ Uniform(0,1)")
    assert model6(10).shape[0] == 10


# TODO: Add convergence tests that verify the model learns correct parameters
# (e.g., given y <- 2*x + e, check that trained weight approaches 2).
# This requires longer training loops and tolerance for non-determinism.
