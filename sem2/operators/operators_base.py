"""Operator input/output contracts."""
from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from enum import StrEnum
import torch

from ..data.records import DataBatch, ScalarResult
from ..model.distributions import DiscreteJointDistribution

if TYPE_CHECKING:
    from ..model.models.base import SEM2


class OperatorInput(StrEnum):
    """Input contract: DATA consumes a batch; SELF evaluates the model alone."""
    DATA = "Data"       # Requires an observation DataBatch
    SELF = "self"       # Model-only; may take query/intervention arguments


class OperatorOutput(StrEnum):
    """Result kind: scalar, matrix, named draws or a finite probability law."""
    SCALAR = "Scalar"   # ScalarResult, retaining gradients
    MATRIX = "Matrix"   # Tensor with axis metadata in diagnostics
    DATA_BATCH = "DataBatch"
    DISCRETE_DISTRIBUTION = "DiscreteJointDistribution"


class SEM2Operator(ABC):
    # maybe make even more clear this is shared between operators representing objectives?
    """Common result contract; concrete evaluators declare explicit arguments.

    Scalar objectives share optional sample_bs and beta keywords for composition.
    sample_bs applies to Monte Carlo sampling, beta only to energy calculations.
    Other terms accept these known settings without changing their calculation.
    """
    model: SEM2
    input_type: OperatorInput
    output_type: OperatorOutput

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> tuple[ScalarResult | DataBatch | DiscreteJointDistribution | torch.Tensor, dict]:
        """Return a typed result and a dictionary of diagnostics."""
