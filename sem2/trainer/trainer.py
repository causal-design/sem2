import logging
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..model.models.base import SEM2
from ..operators.operators_base import OperatorInput, OperatorOutput, SEM2Operator
from ..operators.discrete_iv import DiscreteIVEnergyLoss
from ..operators.sem2_loss import SEM2Loss
from ..default_configs import TrainingConfig
from ..data.records import DataBatch, VariableInfo
from ..data.preparation import prepare_data
from ..utils import set_seed

logger = logging.getLogger(__name__)


def _check_iv_batch(objective: SEM2Operator, batch: DataBatch, batch_size: int) -> None:
    """Check IV eligibility once on complete cases, including composed objectives."""
    if isinstance(objective, SEM2Loss):
        for component, _ in objective.loss_components:
            _check_iv_batch(component, batch, batch_size)
    elif isinstance(objective, DiscreteIVEnergyLoss):
        counts = dict(zip(objective.levels, map(len, objective._group_observations(batch))))
        if min(counts.values()) < 2:
            raise ValueError(f"Instrument {objective.instrument!r} requires at least two observations "
                             f"per declared level; counts={counts}.")
        if batch_size < len(batch):
            raise ValueError(f"IV fitting requires a full-data batch; set batch_size >= {len(batch)}.")


def get_optimiser(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    """Create a fresh optimizer for all model parameters.

    Parameters
    ----------
    model : torch.nn.Module
        Module whose parameters are passed to the optimizer.
    cfg : TrainingConfig
        Optimizer name, learning rate, weight decay and SGD momentum.

    Returns
    -------
    torch.optim.Optimizer
        AdamW, Adam or SGD with no reused optimizer state.
    """
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

    Ordinary tensor minibatches use PyTorch's shuffled DataLoader. IV objectives
    require one full-data batch; no stratification or group balancing is done.

    Parameters
    ----------
    cfg : TrainingConfig
        Device, optimizer, minibatch and epoch settings.
    sem : SEM2
        Model to update in place, already on ``cfg.device``.
    data : pandas.DataFrame
        Named scalar observations, prepared once without mutating the frame.
        Incomplete rows are dropped and reported in ``preparation_report``.
    objective : SEM2Operator
        Data-taking scalar objective bound to the supplied model, minimized by
        training. Negate log-likelihood when composing a loss.
    variables : list[VariableInfo], optional
        Metadata keyed by normalized data column names.
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
        _check_iv_batch(objective, batch, cfg.batch_size)
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

        Returns
        -------
        SEM2
            The same model with updated parameters, left in training mode.
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
    """Fit a model to named observations by minimizing a scalar objective.

    Parameters
    ----------
    model : SEM2
        Model to update in place, on the device specified by ``config.device``.
    data : pandas.DataFrame
        Observations of shape (n_rows, n_variables), with columns naming scalar
        model variables. Rows with missing values are dropped; the input is
        unchanged.
    objective : SEM2Operator
        Data-taking scalar objective bound to this model. To maximize
        log-likelihood, give ``LogLikelihood`` a negative weight in ``SEM2Loss``.
    config : TrainingConfig
        Optimizer, learning rate, epoch budget, batch size and device settings.
        An explicit ``rng_seed`` reseeds training; ``None`` does not reseed.
        IV objectives require ``batch_size`` at least the retained row count.
    variables : list[VariableInfo], optional
        Column metadata for data preparation, including categorical encodings.
        Names must match the normalized DataFrame column names.

    Returns
    -------
    SEM2
        The supplied model with updated parameters, left in training mode.

    Notes
    -----
    Each call uses a fresh optimizer without resetting model parameters.
    Completing the epoch budget does not establish convergence.
    """
    return SEM2Trainer(config, model, data, objective, variables=variables).train()


def main():
    """Small IV fitting smoke demo, not a fit-recovery benchmark."""
    from ..data.dummy import DummyDiscreteDataloader
    from ..default_configs import MLPConfig
    from ..model.models.discrete import SEM2Discrete

    set_seed(0)
    logging.basicConfig(level=logging.INFO)
    cfg = TrainingConfig(device="cpu", epochs=1, batch_size=128, sample_bs=32)
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
