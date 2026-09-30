"""Operator input/output contracts and shared numerical representation."""
from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import warnings
import torch

from .._compat import StrEnum
from ..data import DataBatch, DiscreteDistribution, ScalarResult

if TYPE_CHECKING:
    from ..model.models.base import SEM2


class OperatorInput(StrEnum):
    DATA = "Data"       # Requires an observation DataBatch
    SELF = "self"       # Model-only; may take query/intervention arguments


class OperatorOutput(StrEnum):
    SCALAR = "Scalar"   # ScalarResult, retaining gradients
    MATRIX = "Matrix"   # Tensor with axis metadata in diagnostics
    DATA_BATCH = "DataBatch"
    DISCRETE_DISTRIBUTION = "DiscreteDistribution"


class SEM2Operator(ABC):
    model: SEM2
    input_type: OperatorInput
    output_type: OperatorOutput

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> tuple[ScalarResult | DataBatch | DiscreteDistribution | torch.Tensor, dict]:
        """Return a typed result and a dictionary of diagnostics."""


def _apply_numeric_mapping(batch: DataBatch) -> torch.Tensor:
    """Apply numeric mappings without replacing stored category identities.

    Mappings are assumed complete; unmapped values propagate as NaN.
    """
    columns = []
    blocks = batch.values.split([v.width for v in batch.variables], dim=-1)
    for variable, values in zip(batch.variables, blocks):
        if not values.is_floating_point():
            values = values.float()
        if variable.numeric_mapping is not None:
            if variable.width != 1:
                raise ValueError("Numeric mappings require scalar variables.")
            mapped = torch.full_like(values, float("nan"))
            for label, value in variable.numeric_mapping.items():
                code = variable.encoding[label] if variable.encoding is not None else label
                mask = values == code
                mapped = torch.where(mask, value, mapped)

            values = mapped
        elif variable.categorical:
            warnings.warn(
                f"Using stored category codes numerically for {variable.name!r}; "
                "supply a numeric mapping or use category-probability contrasts.",
                stacklevel=3,
            )
        columns.append(values)
    return torch.cat(columns, dim=-1)
