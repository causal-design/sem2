"""Small named-data training example using the canonical package trainer."""
import logging
import pandas as pd
import torch
from sem2.dataloader.data import DataBatch, VariableInfo
from sem2.dataloader import DummyDiscreteDataloader
from sem2.default_configs import TrainingConfig, MLPConfig
from sem2.model import SEM2Discrete
from sem2.trainer import SEM2Trainer
from sem2.operators.scalar_operators import EnergyLoss, MeanPotentialOutcome


def main():
    torch.manual_seed(0)
    logging.basicConfig(level=logging.INFO)
    cfg = TrainingConfig(device="cpu", epochs=2, batch_size=64)
    formula = "U ~ Uniform(0,1); Z ~ Categorical(probs=[1,1]); X <- d(Z,U); Y <- d(X,U)"
    model = SEM2Discrete(formula, node_cfgs={
        "X": MLPConfig(hidden_dims=[8], output_dim=2),
        "Y": MLPConfig(hidden_dims=[8], output_dim=2),
    })
    batch = next(iter(DummyDiscreteDataloader(1, batch_size=128)))
    data = pd.DataFrame(batch.values.numpy(), columns=batch.names)
    objective = EnergyLoss(model, sample_bs=32, instrument_node="Z")
    trainer = SEM2Trainer(cfg, model, data, objective, variables=batch.variables)
    print("Batch:", batch.names, batch.values.shape)
    rng = torch.get_rng_state()
    result, _ = trainer.loss_fn.evaluate(batch)
    torch.set_rng_state(rng)
    reordered, _ = trainer.loss_fn.evaluate(batch.subset(["Y", "Z", "X"]))
    torch.testing.assert_close(result.value, reordered.value)
    print("Named-column reorder check passed.")
    trainer.train()
    intervention = DataBatch(torch.ones(32, 1), (VariableInfo("X"),))
    outcome = MeanPotentialOutcome("Y", "X", model,
                                   variable=VariableInfo("Y", categorical=True, scores={0: 0., 1: 1.}))
    result, _ = outcome.evaluate(intervention)
    print(result.name, result.value.detach(), result.variables)


if __name__ == "__main__":
    main()
