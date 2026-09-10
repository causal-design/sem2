""" Operators specialised for linear SEM2 models (SEM2Linear).

These functionals read the model-implied covariance (and path coefficients) of a
linear SEM. The abstract SEM2LinearModelOperator base guards that the model is in
fact linear; the concrete operators implement specific functionals.
"""
from __future__ import annotations
import math
import warnings
import torch

from .operators_base import OperatorInput, OperatorOutput, SEM2Operator
from ..model.construction import SEMType
from ..model.models import SEM2Linear
from ..data import NamedBatch


class SEM2LinearModelOperator(SEM2Operator):
    """Abstract base for operators that require a linear SEM (SEM2Linear).

    Validates linearity once at construction so concrete operators can rely on
    coef_matrix / cov_matrix being available, and resolves the observed
    variables the operator works over (defaulting to the formula's unobs(...)
    declaration). Named for the *model* it serves (a linear-model operator),
    not "a linear operator".
    """

    def __init__(self, model: SEM2Linear, observed_vars: list[str] | None = None):
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

        vars_ = list(observed_vars) if observed_vars is not None else list(model.observed_vars)
        if not vars_:
            raise ValueError(
                "No observed variables were provided. Pass observed_vars=[...] or "
                "declare unobs(...) in the model formula."
            )
        missing = [v for v in vars_ if v not in model._node_index]
        if missing:
            raise ValueError(f"Observed variables are not in the model: {missing}")
        self.observed_vars = vars_


def _check_observation_shape(observations: torch.Tensor, n_vars: int, operator_name: str):
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
    defaults to the formula's unobs(...) declaration. loc defaults to the
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
    ):
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

    def _loc(self, dtype: torch.dtype) -> torch.Tensor:
        if self.centered:
            loc = torch.zeros(len(self.observed_vars), device=self.device, dtype=dtype)
        elif self.loc is not None:
            loc = self.loc.to(device=self.device, dtype=dtype)
        else:
            mu = self.model.mean_vector()
            idx = torch.tensor(self.model._index(self.observed_vars), device=self.device)
            loc = mu.index_select(0, idx).to(dtype=dtype)

        if loc.ndim != 1 or loc.shape[0] != len(self.observed_vars):
            raise ValueError(
                f"loc must have shape [{len(self.observed_vars)}]; "
                f"got shape {tuple(loc.shape)}."
            )
        return loc

    def evaluate(self, observations: torch.Tensor | NamedBatch, **kwargs) -> tuple[torch.Tensor, dict]:
        """
        Args:
            observations: NamedBatch (aligned by name), or legacy [n, p] tensor
                whose columns the caller explicitly aligns to observed_vars.
        Returns:
            (mean_log_likelihood, metrics)
        """
        if isinstance(observations, NamedBatch):
            observations = observations.select(*self.observed_vars)
        x = observations.to(self.device)
        _check_observation_shape(x, len(self.observed_vars), type(self).__name__)

        sigma = self.model.cov_matrix(self.observed_vars)        # [p, p]
        x = x.to(dtype=sigma.dtype)
        if not torch.isfinite(sigma).all():
            raise ValueError(
                f"{type(self).__name__} requires finite model-implied covariance. "
                "Check that all exogenous distributions have finite second moments."
            )

        loc = self._loc(dtype=sigma.dtype)
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
        return ll, {"log_likelihood": ll.detach(), "log_likelihood_sum": log_probs.sum().detach()}

    def __repr__(self):
        return "LogLikelihood"


class WeightedFrobeniusDistance(SEM2LinearModelOperator):
    """Weighted Frobenius distance between the sample covariance and the
    model-implied covariance.

        F = sum_{i,j} W[i,j] * (S[i,j] - Sigma[i,j])^2

    With W all-ones this is the (squared) Frobenius / unweighted-least-squares
    discrepancy used in SEM estimation (cf. lavaan's ULS objective). A weight
    matrix W can be supplied to recover weighted/GLS-style discrepancies.
    """
    input_type = OperatorInput.DATA
    output_type = OperatorOutput.SCALAR

    def __init__(self, model: SEM2Linear, observed_vars: list[str] | None = None, weight: torch.Tensor | None = None):
        super().__init__(model, observed_vars)
        self.weight = weight

    def evaluate(self, observations: torch.Tensor | NamedBatch, **kwargs) -> tuple[torch.Tensor, dict]:
        """
        Args:
            observations: NamedBatch (aligned by name), or legacy [n, p] tensor
                whose columns the caller explicitly aligns to observed_vars.
        Returns:
            (weighted_frobenius_distance, metrics)
        """
        if isinstance(observations, NamedBatch):
            observations = observations.select(*self.observed_vars)
        x = observations.to(self.device)
        _check_observation_shape(x, len(self.observed_vars), type(self).__name__)
        sample_cov = torch.cov(x.T)                              # [p, p]
        model_cov = self.model.cov_matrix(self.observed_vars)    # [p, p]

        diff = sample_cov - model_cov
        if self.weight is not None:
            loss = (self.weight.to(self.device) * diff ** 2).sum()
        else:
            loss = (diff ** 2).sum()
        return loss, {"weighted_frobenius_distance": loss.detach()}

    def __repr__(self):
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

    def __repr__(self):
        return "PathAnalysis"
