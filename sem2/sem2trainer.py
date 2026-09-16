import torch
import torch.nn as nn
from typing import Iterable
import time
import logging

from .model.models import SEM2
from .operators.scalar_operators import SEM2Loss
from .default_configs import TrainingConfig
from .data import NamedBatch

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
    def __init__(self, cfg: TrainingConfig, sem: SEM2, ds: Iterable):
        """ Trainer class. Handles training a SEM2 model.
        Args:
            cfg: TrainingConfig object containing training configuration.
            sem: SEM2 model to train.
            ds: Iterable yielding NamedBatch objects (e.g. DummyDiscreteDataloader).
        """

        self.cfg = cfg
        self.sem = sem
        self.ds = ds
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

    def _fit_batch(self, batch: NamedBatch):
        """ Fit model on a single batch.
            Includes making predictions, calculating gradients, and updating model parameters.
        """
        self.optim.zero_grad()

        if not isinstance(batch, NamedBatch):
            raise TypeError("SEM2Trainer expects NamedBatch objects from its dataloader.")
        batch = batch.to(self.device)
        # Compute loss
        loss, loss_metrics = self.loss_fn.evaluate(batch)
        # Calculate gradients and update model parameters
        loss.backward()
        self.optim.step()

        # Detach to stop memory leak
        loss = loss.detach()
        loss_metrics = {k: v.detach() for k, v in loss_metrics.items()}
        return loss, loss_metrics
