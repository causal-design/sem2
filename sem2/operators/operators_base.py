"""Operator input/output contracts and shared numerical representation."""
from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import warnings
import torch

from .._compat import StrEnum
from ..data import DataBatch, FiniteDistribution, ScalarResult

if TYPE_CHECKING:
    from ..model.models import SEM2


class OperatorInput(StrEnum):
    DATA = "Data"       # Requires an observation DataBatch
    SELF = "self"       # Model-only; may take query/intervention arguments


class OperatorOutput(StrEnum):
    SCALAR = "Scalar"   # ScalarResult, retaining gradients
    DATA_BATCH = "DataBatch"
    FINITE_DISTRIBUTION = "FiniteDistribution"


class SEM2Operator(ABC):
    model: SEM2
    input_type: OperatorInput
    output_type: OperatorOutput

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> tuple[ScalarResult | DataBatch | FiniteDistribution, dict]:
        """Return a typed result and a dictionary of diagnostics."""


def _numeric_values(batch: DataBatch) -> torch.Tensor:
    """Use supplied scores without replacing stored category identities."""
    columns = []
    for variable in batch.variables:
        values = batch.select(variable.name)
        if not values.is_floating_point():
            values = values.float()
        if variable.scores is not None:
            if variable.width != 1:
                raise ValueError("Category scores require scalar variables.")
            scored, matched = torch.zeros_like(values), torch.zeros_like(values, dtype=torch.bool)
            for label, score in variable.scores.items():
                code = variable.encoding[label] if variable.encoding is not None else label
                mask = values == code
                scored = torch.where(mask, score, scored)
                matched |= mask
            if not matched.all():
                raise ValueError(f"Missing scores for values of {variable.name!r}.")
            values = scored
        elif variable.categorical:
            warnings.warn(
                f"Using stored category codes numerically for {variable.name!r}; "
                "supply explicit scores or use category-probability contrasts.",
                stacklevel=3,
            )
        columns.append(values)
    return torch.cat(columns, dim=-1)
