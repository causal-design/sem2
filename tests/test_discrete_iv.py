import math

import pytest
import torch

from sem2.data.records import DataBatch, VariableInfo
from sem2.default_configs import MLPConfig, TrainingConfig
from sem2.model.models.discrete import SEM2Discrete
from sem2.operators.discrete_iv import DiscreteIVEnergyLoss


@pytest.mark.parametrize("assignment", ["Z ~ Bernoulli(0.5)", "Z <- d(EZ;levels=2)"])
def test_iv_loss_scores_only_the_response_in_both_encodings(assignment):
    model = SEM2Discrete(f"{assignment}; A <- d(Z;levels=2)", node_cfgs={"A": MLPConfig(hidden_dims=[])})
    layer = model.node_dict["A"].nn_module.net[0]
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.], [math.log(3.)]]))
        layer.bias.zero_()
    batch = DataBatch(torch.tensor([[0., 0.], [0., 0.], [1., 0.], [1., 1.]]),
                      ["Z", VariableInfo("A", categorical=True, numeric_mapping={0: 0., 1: 1.})])
    loss = DiscreteIVEnergyLoss(TrainingConfig(device="cpu"), model, "Z", unbiased=False)
    result, _ = loss.evaluate(batch, sample_bs=4)
    # Binary half-energy distance is squared probability error in each group.
    assert result.value.item() == pytest.approx((.5**2 + (.75 - .5)**2) / 2)
    assert [v.name for v in result.variables] == ["A"]
    with pytest.raises(ValueError, match="must be observed"):
        loss.evaluate(batch.subset("A"))


@pytest.mark.parametrize(("extra", "error"), [
    ("", None), ("; EZ <-> EY", "not separated"), ("; W <- d(Z;levels=2)", "Exclusion"),
])
def test_iv_graph_checks(extra, error):
    parents = "A,EY,W" if "W" in extra else "A,EY"
    model = SEM2Discrete(f"Z <- d(EZ;levels=2); A <- d(Z,EA;levels=2); "
                         f"Y <- d({parents};levels=2); EA <-> EY" + extra)
    loss = DiscreteIVEnergyLoss(TrainingConfig(device="cpu"), model, "Z")
    if error:
        with pytest.raises(ValueError, match=error):
            loss.check_iv_graph("A", "Y")
    else:
        loss.check_iv_graph("A", "Y")
