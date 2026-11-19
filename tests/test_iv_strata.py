import math

import pytest
import torch

from sem2.data.records import DataBatch, VariableInfo
from sem2.default_configs import MLPConfig, TrainingConfig
from sem2.model.models.discrete import SEM2Discrete
from sem2.operators.discrete_iv import DiscreteIVEnergyLoss


@pytest.mark.parametrize("assignment", ["Z ~ Bernoulli(0.5)", "Z <- d(EZ;levels=2)"])
def test_stratified_loss_matches_four_cell_reference(assignment):
    model = SEM2Discrete(
        f"C <- d(EC;levels=2); {assignment}; A <- d(C,Z;levels=2)",
        node_cfgs={"A": MLPConfig(hidden_dims=[])})
    with torch.no_grad():
        layer = model.node_dict["A"].nn_module.net[0]
        layer.weight.copy_(torch.tensor([[0., 0.], [math.log(3), -math.log(3)]]))
        layer.bias.zero_()
    # Counts 1, 2, 3, 1. The stratum/level probabilities predicted for
    # A=1 are 1/2, 1/4, 3/4, 1/2. Binary half-energy = squared discrepancy.
    batch = DataBatch(torch.tensor([[0,0,0], [0,1,1], [0,1,0], [1,0,1],
                                    [1,0,1], [1,0,1], [1,1,1]], dtype=torch.float32),
                      ["C", "Z", VariableInfo("A", categorical=True, numeric_mapping={0:0.,1:1.})])
    loss = DiscreteIVEnergyLoss(TrainingConfig(device="cpu"), model, "Z", covariates=("C",), unbiased=False)
    result, _ = loss.evaluate(batch, sample_bs=8)
    expected = (.5**2 + 2*(.5-.25)**2 + 3*(1-.75)**2 + .5**2)/7
    assert result.value.item() == pytest.approx(expected, abs=1e-6)
    result.value.backward()
    assert layer.weight.grad.isfinite().all()
    with pytest.raises(ValueError, match="counts=0"):
        loss.evaluate(batch[:-1])


@pytest.mark.parametrize(("extra","error"), [
    ("", None), ("; EZ <-> EY", "not separated"), ("; W <- d(Z;levels=2)", "Exclusion"),
])
def test_iv_graph_checks(extra, error):
    outcome = "A,EY,W" if "W" in extra else "A,EY"
    model = SEM2Discrete(f"Z <- d(EZ;levels=2); A <- d(Z,EA;levels=2); "
                         f"Y <- d({outcome};levels=2); EA <-> EY" + extra)
    loss = DiscreteIVEnergyLoss(TrainingConfig(device="cpu"), model, "Z")
    if error:
        with pytest.raises(ValueError, match=error):
            loss.check_iv_graph("A","Y")
    else:
        loss.check_iv_graph("A","Y")
