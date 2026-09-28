"""Small named-data training example using the canonical package trainer."""
import logging
import torch
from sem2.data import DataBatch, VariableInfo
from sem2.dataloader import DummyDiscreteDataloader
from sem2.default_configs import TrainingConfig, MLPConfig
from sem2.model import SEM2Discrete
from sem2.sem2trainer import SEM2Trainer
from sem2.operators.scalar_operators import MeanPotentialOutcome


def main():
    torch.manual_seed(0)
    logging.basicConfig(level=logging.INFO)
    cfg = TrainingConfig(device="cpu", epochs=2, batch_size=64, sample_bs=32, instrument_node="Z")
    formula = "U ~ Uniform(0,1); Z ~ Categorical(probs=[1,1]); X <- d(Z,U); Y <- d(X,U)"
    model = SEM2Discrete(formula, node_cfgs={
        "X": MLPConfig(hidden_dims=[8], output_dim=2),
        "Y": MLPConfig(hidden_dims=[8], output_dim=2),
    })
    loader = DummyDiscreteDataloader(2, batch_size=cfg.batch_size)
    trainer = SEM2Trainer(cfg, model, loader)
    batch = next(iter(loader))
    print("Batch:", batch.names, batch.values.shape)
    rng = torch.get_rng_state()
    result, _ = trainer.loss_fn.evaluate(batch)
    torch.set_rng_state(rng)
    reordered, _ = trainer.loss_fn.evaluate(batch.subset(["Y", "Z", "X"]))
    torch.testing.assert_close(result.value, reordered.value)
    print("Named-column reorder check passed.")
    trainer.train()
    intervention = DataBatch(torch.ones(cfg.sample_bs, 1), (VariableInfo("X"),))
    outcome = MeanPotentialOutcome("Y", "X", cfg, model,
                                   variable=VariableInfo("Y", categorical=True, scores={0: 0., 1: 1.}))
    result, _ = outcome.evaluate(intervention)
    print(result.name, result.value.detach(), result.variables)


if __name__ == "__main__":
    main()
