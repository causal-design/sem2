"""Likelihood and path analysis requiring linear models."""
from __future__ import annotations
import math
import warnings
import torch

from .operators_base import OperatorInput, OperatorOutput, SEM2Operator

from ..model.construction.sem_types import SEMType
from ..model.models.linear import SEM2Linear
from ..data.records import DataBatch, ScalarResult


class SEM2LinearModelOperator(SEM2Operator):
    """Validate the model and variable names shared by linear-model operators.

    Parameters
    ----------
    model : SEM2Linear
        Linear structural equation model providing analytical moments.
    observed_vars : list[str], optional
        Nonempty selection of model variable names. If omitted, the concrete
        operator determines the selection from its input.

    Notes
    -----
    Abstract base class; subclasses implement ``evaluate``. Observation names
    belong to the operator, not the model's structural graph.
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
    """Require a two-dimensional tensor with ``n_vars`` columns.

    ``observations`` contains rows of data; ``operator_name`` labels errors.
    Return ``None`` on success; raise ``ValueError`` for an incorrect shape.
    """
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
    """Evaluate mean Gaussian (pseudo-)log-likelihood over the observations using a linear model's first and second moments.

    Parameters
    ----------
    model : SEM2Linear
        Model supplying the mean and covariance of the selected variables.
    observed_vars : list[str], optional
        Variables to evaluate, in order. Defaults to the input batch's names.
    loc : torch.Tensor, optional
        Mean of shape (n_variables,), in the selected variable order.
        Defaults to the model-implied mean.
    centered : bool, default=False
        Use a zero mean instead of the model mean. Does not center the data;
        supply centered observations. Cannot be combined with ``loc``.

    Notes
    -----
    Exact for linear Gaussian models. Otherwise, this is a Gaussian
    pseudo-log-likelihood based on finite first and second moments, and
    construction emits a warning. The selected covariance must be positive
    definite. Negate this objective when using a minimizing optimizer.
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
        """Resolve a mean vector of shape (len(names),) on the model device.

        ``dtype`` sets the output dtype; ``names`` sets the variable order.
        Use zeros, the supplied mean, or model moments according to the
        constructor settings. Raise ``ValueError`` for an incompatible shape.
        """
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

    def evaluate(self, observations: DataBatch, *, sample_bs: int | None = None,
                 beta: float | None = None) -> tuple[ScalarResult, dict]:
        """Return the mean log-likelihood and detached batch diagnostics.

        Parameters
        ----------
        observations : DataBatch
            Named scalar observations with values of shape (n_rows, n_columns).
            Selected columns are aligned by name and converted to the model's
            device and covariance dtype. Supply at least one row.
        sample_bs : int, optional
            Accepted for objective composition; unused by this calculation.
        beta : float, optional
            Accepted for objective composition; unused by this calculation.

        Returns
        -------
        result : ScalarResult
            Mean log-likelihood, retaining gradients and selected variable
            metadata; labelled ``log_likelihood``.
        diagnostics : dict[str, torch.Tensor]
            Detached scalar tensors: ``log_likelihood`` (mean) and
            ``log_likelihood_sum`` (sum over rows).
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


class PathAnalysis(SEM2LinearModelOperator):
    # leave as-is for first release? or should leave out this class
    """Reserve the interface for path analysis; not implemented.

    Parameters
    ----------
    model : SEM2Linear
        Linear model, validated at construction.
    observed_vars : list[str], optional
        Model variable names, validated but not used in a calculation.

    Notes
    -----
    Construction only validates inputs. Calling ``evaluate`` always raises
    ``NotImplementedError``; no path effects are computed.
    """
    input_type = OperatorInput.SELF
    output_type = OperatorOutput.SCALAR

    def evaluate(self, *, sample_bs: int | None = None,
                 beta: float | None = None) -> tuple[torch.Tensor, dict]:
        """Raise ``NotImplementedError`` without computing a result.

        Parameters
        ----------
        sample_bs : int, optional
            Unused composition argument.
        beta : float, optional
            Unused composition argument.

        Raises
        ------
        NotImplementedError
            Always; path analysis is not available.
        """
        raise NotImplementedError(
            "PathAnalysis is not implemented yet; the API and automatic path "
            "enumeration still need to be designed."
        )

    def __repr__(self) -> str:
        return "PathAnalysis"
