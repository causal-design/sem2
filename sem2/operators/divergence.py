"""Distributional and covariance discrepancies for model fitting."""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from .operators_base import OperatorInput, OperatorOutput, SEM2Operator
from .covariance import Covariance
from .discrete_energy_loss import DiscreteEnergyLoss
from ..data.records import DataBatch, ScalarResult
from ..data.numeric import numeric_values
from ..model.construction.sem_types import SEMType
from ..default_configs import TrainingConfig

if TYPE_CHECKING:
    from ..model.models.base import SEM2


class EnergyLoss(SEM2Operator):
    """Compare observations with draws or probabilities using provisional energy code.

    The calculation is retained as an executable placeholder pending Step-7
    distance/exponent/scaling and estimator review; no new estimator is approved
    by this layout/dispatch correction. IV specialization remains separate.

    Settings resolve as evaluation override > constructor override > config.
    Per-call overrides do not change the stored defaults.
    """
    input_type = OperatorInput.DATA
    output_type = OperatorOutput.SCALAR

    def __init__(self, cfg: TrainingConfig, model: SEM2, *,
                 sample_bs: int | None = None, beta: float | None = None) -> None:
        self.model = model
        self.device = cfg.device
        self.sample_bs = cfg.sample_bs if sample_bs is None else sample_bs
        self.beta = cfg.energy_score_beta if beta is None else beta
        self.discrete_loss = DiscreteEnergyLoss()
        self._evaluate_model = (self._evaluate_continuous if SEMType.is_continuous(model.sem_type)
                                else self._evaluate_discrete)

    def evaluate(self, observations: DataBatch, *, sample_bs: int | None = None,
                 beta: float | None = None) -> tuple[ScalarResult, dict]:
        sample_bs = self.sample_bs if sample_bs is None else sample_bs
        beta = self.beta if beta is None else beta
        if not isinstance(observations, DataBatch):
            raise TypeError("observations must be a DataBatch.")
        observations = observations.to(self.device)
        unknown = set(observations.names) - self.model.input_node
        if unknown:
            raise ValueError(f"Unknown or internal observed variables: {sorted(unknown)}")
        return self._evaluate_model(observations, sample_bs=sample_bs, beta=beta)

    def _evaluate_continuous(self, observations: DataBatch, *,
                             sample_bs: int, beta: float) -> tuple[ScalarResult, dict]:
        """Compare named observations with realized continuous model draws."""
        prediction, _ = self.model.sample(batch_size=sample_bs)
        return self._energy_score(observations, prediction, beta=beta)

    def _evaluate_discrete(self, observations: DataBatch, *,
                          sample_bs: int, beta: float) -> tuple[ScalarResult, dict]:
        """Align observed endogenous names with the enumerated model law."""
        names = [name for name in observations.names if name in self.model.state_node_names
                 and not self.model.node_specs[name].is_helper]
        outcomes = observations.subset(names)
        prediction, _ = self.model.sample(batch_size=sample_bs)
        return self.discrete_loss.energy_dist(outcomes, prediction, beta=beta)

    def _energy_score(self, observations: DataBatch, prediction: DataBatch,
                      *, beta: float) -> tuple[ScalarResult, dict]:
        """Compare observations [N, D] with named draws [M, D], allowing N != M."""
        targets = numeric_values(observations)
        draws = numeric_values(DataBatch(prediction.select(observations.names), observations.variables)).to(targets)
        n = draws.shape[0]
        if not len(observations) or n < 2:
            raise ValueError("Energy score requires observations and at least two model draws.")
        # Cross term: average every observation/model pair, even when N != M.
        cross = (torch.cdist(targets, draws) ** beta).mean()
        distances = torch.cdist(draws, draws) ** beta
        # Model pair term: omit self-pairs for the unbiased finite-sample estimate.
        pair = (distances.sum() - distances.diagonal().sum()) / (n * (n - 1))
        value = cross - pair / 2
        return ScalarResult(value, "energy_score", observations.variables), {
            "energy_score": value, "estimated-true distance": cross, "estimated-estimated distance": pair
        }

    def __repr__(self) -> str:
        return "EnergyLoss"


class WeightedFrobeniusDistance(SEM2Operator):
    """Weighted Frobenius distance between the sample covariance and the
    model-implied covariance.

        F = sum_{i,j} W[i,j] * (S[i,j] - Sigma[i,j])^2

    With W all-ones this is the (squared) Frobenius / unweighted-least-squares
    discrepancy used in SEM estimation (cf. lavaan's ULS objective). A weight
    matrix W can be supplied to recover weighted/GLS-style discrepancies.
    beta is accepted for scalar-objective composition but does not affect this loss.
    """
    input_type = OperatorInput.DATA
    output_type = OperatorOutput.SCALAR

    def __init__(self, model: SEM2, observed_vars: list[str] | None = None,
                 weight: torch.Tensor | None = None, sample_bs: int | None = None,
                 *, cfg: TrainingConfig | None = None) -> None:
        self.model = model
        self.observed_vars = observed_vars
        self.covariance = Covariance(model, sample_bs=sample_bs, cfg=cfg)
        self.weight = weight

    def evaluate(self, observations: DataBatch, *, sample_bs: int | None = None,
                 beta: float | None = None) -> tuple[ScalarResult, dict]:
        """
        Args:
            observations: named data, selected in observed_vars order.
        Returns:
            (weighted_frobenius_distance, metrics)
        """
        if not isinstance(observations, DataBatch):
            raise TypeError("observations must be a DataBatch.")
        batch = observations if self.observed_vars is None else observations.subset(self.observed_vars)
        covariance, _ = self.covariance.evaluate(sample_bs=sample_bs)
        indices = [i for n in batch.names for i in self.covariance.columns[n]]
        model_cov = covariance[indices][:, indices]
        if len(batch) < 2:
            raise ValueError("Sample covariance requires at least two observations.")
        sample_cov = torch.cov(batch.values.to(model_cov).T).reshape_as(model_cov)

        diff = sample_cov - model_cov
        if self.weight is not None:
            loss = (self.weight.to(device=model_cov.device) * diff ** 2).sum()
        else:
            loss = (diff ** 2).sum()
        return ScalarResult(loss, "weighted_frobenius_distance", batch.variables), {
            "weighted_frobenius_distance": loss.detach()
        }

    def __repr__(self) -> str:
        return "WeightedFrobeniusDistance"
