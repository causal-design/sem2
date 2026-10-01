import torch
import torch.nn as nn
from typing import Iterable
import time
import logging

from .model.models import SEM2
from .operators.scalar_operators import SEM2Loss, EnergyLoss, ATE
from .default_configs import TrainingConfig
from .data import DataBatch
from .utils import set_seed

logger = logging.getLogger(__name__)


def get_optimiser(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    if cfg.optim == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optim == "Adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optim == "SGD":
        return torch.optim.SGD(model.parameters(), momentum=0.9, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Invalid optimiser: {cfg.optim}")


class SEM2Trainer:
    def __init__(self, cfg: TrainingConfig, sem: SEM2, ds: Iterable[DataBatch]) -> None:
        """ Trainer class. Handles training a SEM2 model.
        Args:
            cfg: TrainingConfig object containing training configuration.
            sem: SEM2 model to train.
            ds: Iterable dataset to train on. Each element should be a batch of data.
        """

        self.cfg: TrainingConfig = cfg
        self.sem = sem
        self.ds = ds
        self.device = cfg.device

        # Express the existing config-driven defaults as operator–weight pairs.
        components = []
        if cfg.use_energy_score:
            components.append((EnergyLoss(cfg, sem), cfg.energy_score_weight))
        if cfg.use_ate:
            components.append((ATE(cfg, sem), cfg.ate_weight))
        if cfg.use_mmd:
            raise NotImplementedError("MMD loss not implemented yet.")
        self.loss_fn = SEM2Loss(components)
        self.optim = get_optimiser(sem, cfg)

    def train(self) -> None:
        """ Train the model. """
        cfg = self.cfg

        self.sem.train()

        for epoch in range(cfg.epochs):
            epoch_st_t = time.perf_counter()

            losses = []
            # Train steps
            for batch in self.ds:
                loss, loss_metrics = self._fit_batch(batch)
                losses.append(loss)
            # Timing
            epoch_end_t = time.perf_counter()
            epoch_time = epoch_end_t - epoch_st_t
            logger.info(f"Epoch {epoch}: {epoch_time:.2f}s")
            # Get loss statistics
            average_loss = torch.stack(losses).mean().item()
            logger.info(f'{average_loss=:.4g}')

    def _fit_batch(self, batch: DataBatch) -> tuple[torch.Tensor, dict]:
        """ Fit model on a single batch.
            Includes making predictions, calculating gradients, and updating model parameters.
        """
        self.optim.zero_grad()

        if not isinstance(batch, DataBatch):
            raise TypeError("The loader must yield DataBatch objects.")
        batch = batch.to(self.device)
        # Compute loss
        result, loss_metrics = self.loss_fn.evaluate(batch)
        loss = result.value
        # Calculate gradients and update model parameters
        loss.backward()
        self.optim.step()

        # Detach to stop memory leak
        loss = loss.detach()
        loss_metrics = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in loss_metrics.items()}
        return loss, loss_metrics


def main() -> None:
    from .dataloader import DummyDiscreteDataloader
    from .default_configs import MLPConfig
    from .model import SEM2Discrete
    from .operators.discrete_iv import DiscreteIVEnergyLoss

    set_seed(0)
    logging.basicConfig(level=logging.INFO)
    cfg = TrainingConfig(device="cpu", epochs=1, sample_bs=32)
    model = SEM2Discrete("Z ~ Categorical(probs=[1,1]); X <- d(Z; levels=2); Y <- d(X; levels=2)", node_cfgs={
        "X": MLPConfig(hidden_dims=[8]),
        "Y": MLPConfig(hidden_dims=[8]),
    })
    trainer = SEM2Trainer(cfg, model, DummyDiscreteDataloader(2, batch_size=64))
    trainer.loss_fn = SEM2Loss([(DiscreteIVEnergyLoss(cfg, model, instrument="Z"), cfg.energy_score_weight)])
    trainer.train()


if __name__ == "__main__":
    main()
