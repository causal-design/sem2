"""Linear-model likelihood and covariance-based objectives.

Likelihood and path analysis require linear models. Covariance discrepancy
delegates to the general Covariance operator.
"""
from __future__ import annotations
import math
import warnings
import torch

from .operators_base import OperatorInput, OperatorOutput, SEM2Operator
from .covariance import Covariance
from ..model.construction import SEMType
from ..model.models import SEM2Linear
from ..dataloader.data import DataBatch, ScalarResult
from ..model.models.base import SEM2


class SEM2LinearModelOperator(SEM2Operator):
    """Abstract base for operators that require a linear SEM (SEM2Linear).

    Validates linearity once at construction so concrete operators can rely on
    coef_matrix / cov_matrix being available. Observation names belong to the
    operator or its input batch, not the graph. Named for the *model* it serves,
    not "a linear operator".
    """

    def __init__(self, model: SEM2Linear, observed_vars: list[str] | None = None) -> None:
        inferred_type = getattr(model, "inferred_sem_type", None)

        if not isinstance(model, SEM2Linear):
            if inferred_type is not None and SEMType.is_linear(inferred_type):
                msg = (
                    "Detected you are using a linear SEM through SEM2. "
                    "Best to use the special child class SEM2Linear since it has more "
                    "features available. "
                    "Linear operators require SEM2Linear because SEM2 does not expose "
                    "cov_matrix() / coef_matrix()."
                )
                warnings.warn(msg, stacklevel=2)
                raise TypeError(msg)
            if inferred_type is None:
                raise TypeError(
                    f"{type(self).__name__} requires an SEM2Linear model; "
                    f"got {type(model).__name__}."
                )
            raise ValueError(
                f"{type(self).__name__} requires a linear SEM2 model "
                f"(SEM2Linear); got inferred type "
                f"'{inferred_type.value}'."
            )

        if not SEMType.is_linear(model.inferred_sem_type):
            raise ValueError(
                f"{type(self).__name__} requires a linear SEM2 model "
                f"(SEM2Linear); got inferred type "
                f"'{model.inferred_sem_type.value}'."
            )
        self.model = model
        self.device = getattr(model, "device", "cpu")

        vars_ = list(observed_vars) if observed_vars is not None else None
        if vars_ == []:
            raise ValueError("At least one observed variable is required.")
        missing = [v for v in (vars_ or ()) if v not in model.input_node]
        if missing:
            raise ValueError(f"Observed variables are not in the model: {missing}")
        self.observed_vars = vars_


def _check_observation_shape(observations: torch.Tensor, n_vars: int, operator_name: str) -> None:
    if observations.ndim != 2:
        raise ValueError(
            f"{operator_name} expects observations with shape [n, p]; "
            f"got shape {tuple(observations.shape)}."
        )
    if observations.shape[1] != n_vars:
        raise ValueError(
            f"{operator_name} got {observations.shape[1]} data columns, but "
            f"{n_vars} observed variables were specified."
        )


class LogLikelihood(SEM2LinearModelOperator):
    """Gaussian log-likelihood of observed data under a linear SEM.

    The per-observation log density is computed as for a multivariate normal
    (adapted from torch.distributions.MultivariateNormal.log_prob): with Sigma
    the model-implied covariance over the observed variables and L its Cholesky
    factor,

        log p(x) = -0.5 * (k*log(2*pi) + ||L^-1 (x - loc)||^2) - sum(log diag(L)).

    This is the exact likelihood for LINEAR_GAUSSIAN models. For other linear
    SEMs it is a Gaussian pseudo-log-likelihood based only on the model-implied
    first and second moments; a warning is issued once at construction.

    evaluate() returns the mean log-likelihood over the batch. observed_vars
    defaults to the input batch's names. loc defaults to the
    model-implied mean; use centered=True for centred covariance-style fits.
    """
    input_type = OperatorInput.DATA
    output_type = OperatorOutput.SCALAR

    def __init__(
        self,
        model: SEM2Linear,
        observed_vars: list[str] | None = None,
        loc: torch.Tensor | None = None,
        centered: bool = False,
    ) -> None:
        super().__init__(model, observed_vars)
        if centered and loc is not None:
            raise ValueError("Pass either centered=True or loc=..., not both.")
        self.loc = loc
        self.centered = centered
        if model.inferred_sem_type != SEMType.LINEAR_GAUSSIAN:
            warnings.warn(
                "Using the Gaussian pseudo-log-likelihood for a non-Gaussian linear "
                "SEM. The SEM2Linear covariance formula is valid for linear SEMs with "
                "finite second moments and scalar independent exogenous noise, but "
                "this objective is exact only for LINEAR_GAUSSIAN models and ignores "
                "higher moments.",
                stacklevel=2,
            )

    def _loc(self, dtype: torch.dtype, names: tuple[str, ...]) -> torch.Tensor:
        if self.centered:
            loc = torch.zeros(len(names), device=self.device, dtype=dtype)
        elif self.loc is not None:
            loc = self.loc.to(device=self.device, dtype=dtype)
        else:
            mu = self.model.mean_vector()
            idx = torch.tensor(self.model._index(names), device=self.device)
            loc = mu.index_select(0, idx).to(dtype=dtype)

        if loc.ndim != 1 or loc.shape[0] != len(names):
            raise ValueError(
                f"loc must have shape [{len(names)}]; "
                f"got shape {tuple(loc.shape)}."
            )
        return loc

    def evaluate(self, observations: DataBatch, **kwargs) -> tuple[ScalarResult, dict]:
        """
        Args:
            observations: named data, selected in observed_vars order.
        Returns:
            (mean_log_likelihood, metrics)
        """
        if not isinstance(observations, DataBatch):
            raise TypeError("observations must be a DataBatch.")
        batch = observations if self.observed_vars is None else observations.subset(self.observed_vars)
        x = batch.values.to(self.device)
        _check_observation_shape(x, len(batch.names), type(self).__name__)

        sigma = self.model.cov_matrix(batch.names)               # [p, p]
        x = x.to(dtype=sigma.dtype)
        if not torch.isfinite(sigma).all():
            raise ValueError(
                f"{type(self).__name__} requires finite model-implied covariance. "
                "Check that all exogenous distributions have finite second moments."
            )

        loc = self._loc(dtype=sigma.dtype, names=batch.names)
        if not torch.isfinite(loc).all():
            raise ValueError(
                f"{type(self).__name__} requires finite model-implied means. "
                "Pass loc=... explicitly, use centered=True, or use exogenous "
                "distributions with finite means."
            )

        L = torch.linalg.cholesky(sigma)                         # [p, p]
        diff = (x - loc).T                                       # [p, n]
        # Mahalanobis term: ||L^-1 diff||^2 per observation.
        z = torch.linalg.solve_triangular(L, diff, upper=False)  # [p, n]
        maha = (z ** 2).sum(dim=0)                               # [n]
        half_log_det = torch.log(torch.diagonal(L)).sum()
        k = sigma.shape[0]

        log_probs = -0.5 * (k * math.log(2 * math.pi) + maha) - half_log_det  # [n]
        ll = log_probs.mean()
        return ScalarResult(ll, "log_likelihood", batch.variables), {
            "log_likelihood": ll.detach(), "log_likelihood_sum": log_probs.sum().detach()
        }

    def __repr__(self) -> str:
        return "LogLikelihood"


class WeightedFrobeniusDistance(SEM2Operator):
    """Weighted Frobenius distance between the sample covariance and the
    model-implied covariance.

        F = sum_{i,j} W[i,j] * (S[i,j] - Sigma[i,j])^2

    With W all-ones this is the (squared) Frobenius / unweighted-least-squares
    discrepancy used in SEM estimation (cf. lavaan's ULS objective). A weight
    matrix W can be supplied to recover weighted/GLS-style discrepancies.
    """
    input_type = OperatorInput.DATA
    output_type = OperatorOutput.SCALAR

    def __init__(self, model: SEM2, observed_vars: list[str] | None = None,
                 weight: torch.Tensor | None = None, n_samples: int = 1024) -> None:
        self.model = model
        self.observed_vars = observed_vars
        self.covariance = Covariance(model, n_samples=n_samples)
        self.weight = weight

    def evaluate(self, observations: DataBatch, **kwargs) -> tuple[ScalarResult, dict]:
        """
        Args:
            observations: named data, selected in observed_vars order.
        Returns:
            (weighted_frobenius_distance, metrics)
        """
        if not isinstance(observations, DataBatch):
            raise TypeError("observations must be a DataBatch.")
        batch = observations if self.observed_vars is None else observations.subset(self.observed_vars)
        covariance, _ = self.covariance.evaluate()
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


class PathAnalysis(SEM2LinearModelOperator):
    """Placeholder for future path-analysis functionality."""
    input_type = OperatorInput.SELF
    output_type = OperatorOutput.SCALAR

    def evaluate(self, *args, **kwargs) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError(
            "PathAnalysis is not implemented yet; the API and automatic path "
            "enumeration still need to be designed."
        )

    def __repr__(self) -> str:
        return "PathAnalysis"
