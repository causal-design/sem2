import logging
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..model.models.base import SEM2
from ..operators.operators_base import OperatorInput, OperatorOutput, SEM2Operator
from ..default_configs import TrainingConfig
from ..data.records import DataBatch, VariableInfo
from ..data.preparation import prepare_data
from ..utils import set_seed

logger = logging.getLogger(__name__)


def get_optimiser(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    if cfg.optim == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optim == "Adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optim == "SGD":
        return torch.optim.SGD(model.parameters(), momentum=cfg.momentum, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Invalid optimiser: {cfg.optim}")


class SEM2Trainer:
    """Prepare once, then minimize an explicit scalar objective in place.

    Ordinary tensor minibatches use PyTorch's shuffled DataLoader. Specialized
    IV-valid batching is deferred; no stratification or group balancing is done.
    """

    def __init__(self, cfg: TrainingConfig, sem: SEM2, data: pd.DataFrame,
                 objective: SEM2Operator, *, variables: list[VariableInfo] | None = None):
        """Prepare named data once for in-place fitting with an explicit objective.

        Construct the model on cfg.device. Data-taking scalar objectives are minimized;
        use a negative weight for LogLikelihood in SEM2Loss.
        """
        if objective.input_type != OperatorInput.DATA or objective.output_type != OperatorOutput.SCALAR:
            raise TypeError("Training requires a data-taking scalar objective.")
        if torch.device(sem.device) != torch.device(cfg.device):
            raise ValueError("Construct the model on the same device as TrainingConfig.device.")
        self.cfg: TrainingConfig = cfg
        self.sem = sem
        self.loss_fn = objective
        batch, self.preparation_report = prepare_data(
            sem, data, variables=variables, device=cfg.device,
        )
        self.variables = batch.variables
        self.loader = DataLoader(TensorDataset(batch.values), batch_size=cfg.batch_size, shuffle=True)
        if self.preparation_report["dropped_rows"]:
            logger.warning("Dropped %d rows with missing values.", self.preparation_report["dropped_rows"])

    def train(self) -> SEM2:
        """Run the epoch budget, returning the same model with updated parameters.

        Each call creates a fresh optimizer and reseeds only if cfg.rng_seed is set.
        Model parameters are not reset.
        INFO logging reports the last minibatch loss once per epoch, before that
        minibatch's update; it is not an epoch mean or a final evaluation.
        Scalar extraction occurs only at that logging boundary, not each update.
        Histories, final-evaluation diagnostics and convergence checks are deferred.
        """
        cfg = self.cfg
        set_seed(cfg.rng_seed, device=cfg.device)
        self.optim = get_optimiser(self.sem, cfg)
        self.sem.train()
        for epoch in range(cfg.epochs):
            for (values,) in self.loader:
                last_loss = self._fit_batch(DataBatch(values, self.variables))
            if logger.isEnabledFor(logging.INFO):
                logger.info("Epoch %d/%d completed; last-batch loss=%g.",
                            epoch + 1, cfg.epochs, last_loss.item())
        return self.sem

    def _fit_batch(self, batch: DataBatch) -> torch.Tensor:
        """Take one optimizer step and return its detached pre-update loss."""
        self.optim.zero_grad()
        result, _ = self.loss_fn.evaluate(batch)
        loss = result.value
        loss.backward()
        self.optim.step()
        return loss.detach()


def fit(model: SEM2, data: pd.DataFrame, objective: SEM2Operator, config: TrainingConfig,
        *, variables: list[VariableInfo] | None = None) -> SEM2:
    """Fit and return the supplied model in place, using a fresh optimizer.

    Parameters
    ----------
    model : SEM2
        Model whose parameters are updated; its structural graph is unchanged.
    data : pandas.DataFrame
        Named observations, prepared once without mutating the frame.
    objective : SEM2Operator
        Data-taking scalar objective to minimize. Use a negative weight for
        LogLikelihood inside SEM2Loss. Model-only components may be composed.
    config : TrainingConfig
        Optimizer, epoch and batch settings. rng_seed=None leaves the shared RNG
        stream alone; an explicit seed resets it when training executes, not at
        trainer construction. It does not reproduce earlier model initialization.
    variables : list[VariableInfo] or None
        Optional column metadata passed to prepare_data().

    Returns
    -------
    SEM2
        The same object with updated parameters. Repeated fits create a fresh
        optimizer, not fresh weights. No convergence or successful-fit claim is
        made; per-update finiteness checks, rollback and fit histories are absent.
    """
    return SEM2Trainer(config, model, data, objective, variables=variables).train()


def main():
    """Compatibility smoke demo, not validation of the deferred IV fitting API."""
    from ..data.dummy import DummyDiscreteDataloader
    from ..default_configs import MLPConfig
    from ..model.models.discrete import SEM2Discrete
    from ..operators.discrete_iv import DiscreteIVEnergyLoss

    set_seed(0)
    logging.basicConfig(level=logging.INFO)
    cfg = TrainingConfig(device="cpu", epochs=1, batch_size=64, sample_bs=32)
    model = SEM2Discrete("Z ~ Categorical(probs=[1,1]); X <- d(Z; levels=2); Y <- d(X; levels=2)", node_cfgs={
        "X": MLPConfig(hidden_dims=[8]),
        "Y": MLPConfig(hidden_dims=[8]),
    })
    batch = next(iter(DummyDiscreteDataloader(1, batch_size=128)))
    data = pd.DataFrame(batch.values.numpy(), columns=batch.names)
    objective = DiscreteIVEnergyLoss(cfg, model, instrument="Z")
    fit(model, data, objective, cfg, variables=batch.variables)


if __name__ == "__main__":
    main()
