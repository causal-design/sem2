"""Full-data IV fitting smoke example using the canonical trainer.

This fixture-based run is not an IV fit-recovery or bounds validation example.
"""
import logging
import pandas as pd
import torch
from sem2.data.records import VariableInfo
from sem2.data.dummy import DummyDiscreteDataloader
from sem2.default_configs import TrainingConfig, MLPConfig
from sem2.model.models.discrete import SEM2Discrete
from sem2.trainer.trainer import SEM2Trainer
from sem2.operators.potential_outcomes import MeanPotentialOutcome
from sem2.operators.sem2_loss import SEM2Loss
from sem2.operators.discrete_iv import DiscreteIVEnergyLoss
from sem2.utils import set_seed


def main():
    set_seed(0)
    logging.basicConfig(level=logging.INFO)
    cfg = TrainingConfig(device="cpu", epochs=2, batch_size=128, sample_bs=32)
    formula = "U ~ Uniform(0,1); Z ~ Categorical(probs=[1,1]); X <- d(Z,U; levels=2); Y <- d(X,U; levels=2)"
    model = SEM2Discrete(formula, node_cfgs={
        "X": MLPConfig(hidden_dims=[8]),
        "Y": MLPConfig(hidden_dims=[8]),
    })
    batch = next(iter(DummyDiscreteDataloader(1, batch_size=128)))
    data = pd.DataFrame(batch.values.numpy(), columns=batch.names)
    objective = SEM2Loss([(DiscreteIVEnergyLoss(cfg, model, instrument="Z"), 1.)])
    trainer = SEM2Trainer(cfg, model, data, objective, variables=batch.variables)
    print("Batch:", batch.names, batch.values.shape)
    set_seed(0)
    result, _ = trainer.loss_fn.evaluate(batch)
    set_seed(0)
    reordered, _ = trainer.loss_fn.evaluate(batch.subset(["Y", "Z", "X"]))
    torch.testing.assert_close(result.value, reordered.value)
    print("Named-column reorder check passed.")
    trainer.train()
    outcome = MeanPotentialOutcome(model, outcome="Y", intervention={"X": 1}, cfg=cfg,
                                   variable=VariableInfo("Y", categorical=True, numeric_mapping={0: 0., 1: 1.}))
    result, _ = outcome.evaluate(sample_bs=128)
    print(result.name, result.value.detach(), result.variables)


if __name__ == "__main__":
    main()
