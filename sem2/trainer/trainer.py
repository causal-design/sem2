import logging
import pandas as pd
import torch
import torch.nn as nn

from ..model.models import SEM2
from ..operators.operators_base import OperatorInput, OperatorOutput, SEM2Operator
from ..default_configs import TrainingConfig
from ..dataloader.data import DataBatch, VariableInfo
from ..dataloader import prepare_dataloader
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
    def __init__(self, cfg: TrainingConfig, sem: SEM2, data: pd.DataFrame,
                 objective: SEM2Operator, *, variables: tuple[VariableInfo, ...] = ()):
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
        self.loader, self.variables, self.preparation_report = prepare_dataloader(
            sem, data, cfg.batch_size, variables=variables, device=cfg.device,
        )
        if self.preparation_report["dropped_rows"]:
            logger.warning("Dropped %d rows with missing values.", self.preparation_report["dropped_rows"])

    def train(self) -> SEM2:
        """Run the epoch budget, returning the same model with updated parameters.

        Each call creates a fresh optimizer and reseeds only if cfg.rng_seed is set.
        Model parameters are not reset.
        No histories, final evaluation or convergence claim are produced.
        """
        cfg = self.cfg
        set_seed(cfg.rng_seed, device=cfg.device)
        self.optim = get_optimiser(self.sem, cfg)
        self.sem.train()
        for epoch in range(cfg.epochs):
            for (values,) in self.loader:
                self._fit_batch(DataBatch(values, self.variables))
            logger.info("Epoch %d/%d completed.", epoch + 1, cfg.epochs)
        return self.sem

    def _fit_batch(self, batch: DataBatch) -> None:
        """Evaluate the objective and take one optimizer step."""
        self.optim.zero_grad()
        result, _ = self.loss_fn.evaluate(batch)
        loss = result.value
        loss.backward()
        self.optim.step()


def fit(model: SEM2, data: pd.DataFrame, objective: SEM2Operator, config: TrainingConfig,
        *, variables: tuple[VariableInfo, ...] = ()) -> SEM2:
    """Fit and return the supplied model in place, using a fresh optimizer.

    variables optionally carries column meaning/encoding for prepare_data().
    No per-update finiteness checks or rollback are performed.
    """
    return SEM2Trainer(config, model, data, objective, variables=variables).train()


def main():
    from ..dataloader import DummyDiscreteDataloader
    from ..default_configs import MLPConfig
    from ..model import SEM2Discrete
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
