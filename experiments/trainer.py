import torch
import torch.nn as nn
from typing import Iterable
import time
import logging

from sem2.dataloader import DummyDiscreteDataloader
from sem2.model.model import SEM2Discrete
from sem2.operators.scalar_operators import SEM2Loss
from sem2.default_configs import TrainingConfig, MLPConfig

logger = logging.getLogger(__name__)


def get_optimiser(model: nn.Module, cfg) -> torch.optim.Optimizer:
    if cfg.optim == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optim == "Adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optim == "SGD":
        return torch.optim.SGD(model.parameters(), momentum=0.9, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Invalid optimiser: {cfg.optim}")


class SEM2Trainer:
    def __init__(self, cfg: TrainingConfig, sem: SEM2Discrete, dl: Iterable):
        """ Trainer class. Configured for IV graph Z->X->Y
        Args:
            cfg: TrainingConfig object containing training configuration.
            sem: SEM2 model to train.
            dl: Iterable dataset to train on. Each element should be a batch of data.
        """

        self.cfg = cfg
        self.sem = sem
        self.dl = dl
        self.device = cfg.device

        self.loss_fn = SEM2Loss(cfg, sem)
        self.optim = get_optimiser(sem, cfg)

    def train(self):
        """ Train the model. """
        cfg = self.cfg

        self.sem.train()

        for epoch in range(cfg.epochs):
            epoch_st_t = time.perf_counter()

            losses = []
            energy_loss, ate = 0, 0
            # Train steps
            for batch in self.dl:
                batch = batch.to(self.device)
                loss, loss_metrics = self._fit_batch(batch)
                losses.append(loss)
                energy_loss += loss_metrics["EnergyLoss"]
                ate += loss_metrics["ATE"]

            # Timing
            epoch_end_t = time.perf_counter()
            epoch_time = epoch_end_t - epoch_st_t
            logger.info(f"Epoch {epoch}: {epoch_time:.2f}s")

            # Get loss statistics
            average_loss = torch.stack(losses).mean().item()
            average_energy_loss = energy_loss / len(self.dl)
            average_ATE = ate / len(self.dl)
            logger.info(f'{average_loss=:.4g}, {average_energy_loss=:.4g}, {average_ATE=:.4g}')

    def _fit_batch(self, batch):
        """ Fit model on a single batch.
            Includes making predictions, calculating gradients, and updating model parameters.
        """
        self.optim.zero_grad()


        # Compute loss
        loss, loss_metrics = self.loss_fn.evaluate(batch)
        # Calculate gradients and update model parameters
        loss.backward()
        self.optim.step()

        # Detach to stop memory leak
        loss = loss.detach()
        loss_metrics = {k: v.detach() for k, v in loss_metrics.items()}
        return loss, loss_metrics


def main():
    torch.set_printoptions(sci_mode=False, precision=5)
    torch.manual_seed(0)
    logging.basicConfig(level=logging.INFO)
    cfg = TrainingConfig(ate_weight=-0.02)

    formula1 = "U ~ Uniform(0,1); eps_d ~ Uniform(0,1); eps_y ~ Uniform(0,1); Z ~ Categorical(probs=[1, 1]);"
    formula2 = "X <- d(Z, U, eps_d); Y <- d(X, U, eps_y)"
    mlp_cfgs = {"U": 4, "eps_d": 4, "eps_y": 4,
                "X": MLPConfig(input_dim=1 + 4 + 4, output_dim=2),
                "Y": MLPConfig(input_dim=1 + 4 + 4, output_dim=2)
                }
    model_formula = formula1 + formula2
    sem_model = SEM2Discrete(model_formula, node_cfgs=mlp_cfgs, device=cfg.device)

    trainer = SEM2Trainer(cfg, sem_model, DummyDiscreteDataloader(256, batch_size=cfg.batch_size))
    trainer.train()

    # Evaluate model
    from sem2.operators.scalar_operators import ATE
    eval_cfg = cfg.model_copy(update={"batch_size": 10000})
    ATE_evaluator = ATE(eval_cfg, sem_model)

    ate, _ = ATE_evaluator.evaluate()
    print(f'{ate = }')

    # Sample Z=0
    Z0 = torch.zeros([10000, 1], device="cuda", dtype=torch.long)
    set_nodes = {"Z": Z0}
    preds, _  = sem_model.sample(batch_size=10000, set_nodes=set_nodes)
    preds = preds.mean(dim=0) / 2
    print(f'Z=0, {preds = }')

    # Sample Z=1
    Z0 = torch.ones([10000, 1], device="cuda", dtype=torch.long)
    set_nodes = {"Z": Z0}
    preds, _  = sem_model.sample(batch_size=10000, set_nodes=set_nodes)
    preds = preds.mean(dim=0) / 2
    print(f'Z=1, {preds = }')


if __name__ == "__main__":
    main()
