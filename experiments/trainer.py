"""Small named-data IV example: python -m experiments.trainer

Uses the shared SEM2Trainer, not a separate optimization loop. The dummy loader
already supplies numeric binary states; this is not a category-conversion or
DataFrame-preparation implementation. A few steps demonstrate integration only,
not estimator convergence or a validated causal-effect estimate.
"""
import logging

import torch

from sem2.dataloader import DummyDiscreteDataloader
from sem2.default_configs import TrainingConfig, MLPConfig
from sem2.model import SEM2Discrete
from sem2.operators.scalar_operators import EnergyLoss, ATE
from sem2.operators.vector_operators import PotentialOutcome
from sem2.sem2trainer import SEM2Trainer


def main():
    torch.manual_seed(0)
    logging.basicConfig(level=logging.INFO)
    cfg = TrainingConfig(device="cpu", epochs=2, batch_size=32, sample_bs=16,
                         instrument_node="Z", use_ate=True, ate_weight=0.015)
    model = SEM2Discrete(
        "U ~ Uniform(0,1); Z ~ Categorical(probs=[1,1]); X <- d(Z,U); Y <- d(X,U)",
        node_cfgs={n: MLPConfig(hidden_dims=[8], output_dim=2) for n in ("X", "Y")},
        device=cfg.device,
    )
    loader = DummyDiscreteDataloader(2, batch_size=cfg.batch_size)
    batch = next(iter(loader))
    print("Named columns:", batch.names)
    print("Metadata:", batch.variables)
    print("First rows:\n", batch.rows(slice(0, 3)).values)
    print("Select treatment by name:\n", batch.rows(slice(0, 3)).select("X"))

    # Both data and model states are aligned by names, not tensor column order.
    loss_fn = EnergyLoss(cfg, model)
    torch.manual_seed(11)
    original, _ = loss_fn.evaluate(batch)
    torch.manual_seed(11)
    reordered, _ = loss_fn.evaluate(batch.subset("Y", "Z", "X"))
    torch.testing.assert_close(original, reordered)
    print("Energy loss, original/reordered:", original.item(), reordered.item())

    SEM2Trainer(cfg, model, loader).train()
    effect, _ = ATE(cfg, model).evaluate()
    print("ATE after four demonstration updates (not converged):", effect.item())

    outcome = PotentialOutcome(model, "Y", variables=batch.variables)
    result, extra = outcome.evaluate(
        batch_size=cfg.sample_bs, set_nodes={"X": torch.ones(cfg.sample_bs, 1)},
    )
    print("Intervention:", tuple(extra["set_nodes"]))
    print("Outcome metadata:", result.states.variables)
    print("Named outcome states:\n", result.states.values)
    print("Probability axes [exogenous draws, states]:", tuple(result.probabilities.shape))
    # Averaging out exogenous draws is separate from state marginalization.
    print("Marginal outcome probabilities:", result.probabilities.mean(0).detach())
    torch.testing.assert_close(result.probabilities.sum(-1), torch.ones(cfg.sample_bs))


if __name__ == "__main__":
    main()
