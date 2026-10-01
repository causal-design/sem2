"""Small named-data training example using the canonical package trainer."""
import logging
import torch
from sem2.data import VariableInfo
from sem2.dataloader import DummyDiscreteDataloader
from sem2.default_configs import TrainingConfig, MLPConfig
from sem2.model import SEM2Discrete
from sem2.sem2trainer import SEM2Trainer
from sem2.operators.scalar_operators import MeanPotentialOutcome, SEM2Loss
from sem2.operators.discrete_iv import DiscreteIVEnergyLoss
from sem2.utils import set_seed


def main():
    set_seed(0)
    logging.basicConfig(level=logging.INFO)
    cfg = TrainingConfig(device="cpu", epochs=2, batch_size=64, sample_bs=32)
    formula = "U ~ Uniform(0,1); Z ~ Categorical(probs=[1,1]); X <- d(Z,U; levels=2); Y <- d(X,U; levels=2)"
    model = SEM2Discrete(formula, node_cfgs={
        "X": MLPConfig(hidden_dims=[8]),
        "Y": MLPConfig(hidden_dims=[8]),
    })
    loader = DummyDiscreteDataloader(2, batch_size=cfg.batch_size)
    trainer = SEM2Trainer(cfg, model, loader)
    trainer.loss_fn = SEM2Loss([(DiscreteIVEnergyLoss(cfg, model, instrument="Z"), cfg.energy_score_weight)])
    batch = next(iter(loader))
    print("Batch:", batch.names, batch.values.shape)
    set_seed(0)
    result, _ = trainer.loss_fn.evaluate(batch)
    set_seed(0)
    reordered, _ = trainer.loss_fn.evaluate(batch.subset(["Y", "Z", "X"]))
    torch.testing.assert_close(result.value, reordered.value)
    print("Named-column reorder check passed.")
    trainer.train()
    outcome = MeanPotentialOutcome(model, outcome="Y", intervention={"X": 1}, n_samples=cfg.sample_bs,
                                   variable=VariableInfo("Y", categorical=True, numeric_mapping={0: 0., 1: 1.}))
    result, _ = outcome.evaluate()
    print(result.name, result.value.detach(), result.variables)


if __name__ == "__main__":
    main()
