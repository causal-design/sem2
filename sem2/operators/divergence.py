"""Distributional and covariance discrepancies for model fitting."""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from .operators_base import OperatorInput, OperatorOutput, SEM2Operator
from .covariance import Covariance
from .discrete_energy_loss import DiscreteEnergyLoss, _energy_distance, _energy_values
from ..data.records import DataBatch, ScalarResult
from ..model.construction.sem_types import SEMType
from ..default_configs import TrainingConfig

if TYPE_CHECKING:
    from ..model.models.base import SEM2


class EnergyLoss(SEM2Operator):
    """Compare observations with model draws or finite probabilities.

    Quantitative variables use their supplied scale; categorical variables require
    explicit numeric mappings. No standardization or category scoring is inferred.

    Settings resolve as evaluation override > constructor override > config.
    Per-call overrides do not change the stored defaults.

    Parameters
    ----------
    cfg : TrainingConfig
        Device, default Monte Carlo count and energy exponent.
    model : SEM2
        Model to compare with observations, on the configured device.
    sample_bs : int, optional
        Model draws; overrides ``cfg.sample_bs``. Corrected pairs need two.
    beta : float, optional
        Power applied to distances; overrides ``cfg.energy_score_beta``.
    unbiased : bool, optional
        Pair correction; overrides ``cfg.energy_score_unbiased`` at construction.
    """
    input_type = OperatorInput.DATA
    output_type = OperatorOutput.SCALAR

    def __init__(self, cfg: TrainingConfig, model: SEM2, *,
                 sample_bs: int | None = None, beta: float | None = None,
                 unbiased: bool | None = None) -> None:
        self.model = model
        self.cfg = cfg
        self.device = cfg.device
        self.sample_bs = cfg.sample_bs if sample_bs is None else sample_bs
        self.beta = cfg.energy_score_beta if beta is None else beta
        self.unbiased = cfg.energy_score_unbiased if unbiased is None else unbiased
        self.discrete_loss: DiscreteEnergyLoss | None = None
        self._evaluate_model = (self._evaluate_continuous if SEMType.is_continuous(model.sem_type)
                                else self._evaluate_discrete)

    def evaluate(self, observations: DataBatch, *, sample_bs: int | None = None,
                 beta: float | None = None) -> tuple[ScalarResult, dict]:
        """Compare named observations with the current model distribution.

        Parameters
        ----------
        observations : DataBatch
            Nonempty observations. Numeric mappings define category scores;
            discrete evaluation selects observed endogenous variables.
        sample_bs : int, optional
            Draw count for this call, leaving stored defaults unchanged.
        beta : float, optional
            Distance exponent for this call.

        Returns
        -------
        result : ScalarResult
            Continuous energy score or discrete half-energy distance, with
            gradients. The continuous score omits the data/data constant.
        diagnostics : dict
            Cross and within-model distances; the discrete route also includes
            the within-data term, total variation and largest cell-probability
            error. Values are not detached.
        """
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
        loss = self._prepare_discrete_loss(outcomes)
        prediction, _ = self.model.sample(batch_size=sample_bs)
        return loss.energy_dist(outcomes, prediction, beta=beta)

    def _prepare_discrete_loss(self, observations: DataBatch) -> DiscreteEnergyLoss:
        """Construct once when observed names and scoring are known; reuse thereafter."""
        if self.discrete_loss is None or self.discrete_loss.states.variables != observations.variables:
            states = DataBatch(self.model.all_states, self.model.state_node_names).select(observations.names)
            self.discrete_loss = DiscreteEnergyLoss(
                self.cfg, DataBatch(states, observations.variables), beta=self.beta, unbiased=self.unbiased,
            )
        return self.discrete_loss

    def _energy_score(self, observations: DataBatch, prediction: DataBatch,
                      *, beta: float) -> tuple[ScalarResult, dict]:
        """Compare observations [N, D] with named draws [M, D], allowing N != M."""
        targets = _energy_values(observations)
        draws = _energy_values(DataBatch(prediction.select(observations.names), observations.variables)).to(targets)
        n = draws.shape[0]
        if not len(observations) or n < 1 or (self.unbiased and n < 2):
            raise ValueError("Energy score requires observations and enough model draws for the pair estimator.")
        # Cross term: average every observation/model pair, even when N != M.
        cross = _energy_distance(targets, draws, beta).mean()
        distances = _energy_distance(draws, draws, beta)
        # Model pair term: omit self-pairs for the unbiased finite-sample estimate.
        pair = ((distances.sum() - distances.diagonal().sum()) / (n * (n - 1))
                if self.unbiased else distances.mean())
        value = cross - pair / 2
        return ScalarResult(value, "energy_score", observations.variables), {
            "energy_score": value, "estimated-true distance": cross, "estimated-estimated distance": pair
        }

    def __repr__(self) -> str:
        return "EnergyLoss"


class WeightedFrobeniusDistance(SEM2Operator):
    """Weighted Frobenius distance between the sample covariance and the
    model-implied covariance.

    .. math::

        F = \\sum_{i,j} W_{ij}(S_{ij}-\\Sigma_{ij})^2.

    With W all-ones this is the (squared) Frobenius / unweighted-least-squares
    discrepancy used in SEM estimation (cf. lavaan's ULS objective). A weight
    matrix W supplies elementwise weighted least squares; this corresponds to
    diagonal weighting of vectorized residuals, not a general GLS quadratic form.
    beta is accepted for scalar-objective composition but does not affect this loss.

    Parameters
    ----------
    model : SEM2
        Model providing analytical or sampled covariance.
    observed_vars : list[str], optional
        Selected variables in covariance order; defaults to batch names.
    weight : torch.Tensor, optional
        Elementwise weights broadcastable to the selected covariance matrix.
        Defaults to ones; not a full weight matrix on vectorized residuals.
    sample_bs : int, optional
        Draw count for non-linear models, overriding the config default.
    cfg : TrainingConfig, optional
        Supplies the default draw count.
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
        """Return the weighted squared covariance discrepancy.

        Parameters
        ----------
        observations : DataBatch
            Named values with at least two rows, selected in ``observed_vars``
            order. Uses stored values, not numeric mappings.
        sample_bs : int, optional
            Draw-count override; unused for analytical linear covariance.
        beta : float, optional
            Accepted for composition; unused.

        Returns
        -------
        result : ScalarResult
            Weighted Frobenius discrepancy with parameter gradients.
        diagnostics : dict
            Detached scalar under ``weighted_frobenius_distance``.
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
