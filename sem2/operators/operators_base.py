""" Defining the types of operators
    Operators are defined based on:
        a) Inputs type
        b) Output type
"""
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import torch
from enum import StrEnum

if TYPE_CHECKING:
    from sem2.model.model import SEM2


class OperatorInput(StrEnum):
    """ Defines the input type of an operator, if the operator takes in data."""
    DATA = "Data"       # Data required as input for the operator
    SELF = "self"       # Operator takes in the model itself as input, and no data


class OperatorOutput(StrEnum):
    """ Defines the output type of an operator. """
    SCALAR = "Scalar"   # Output is a scalar value
    VECTOR = "Vector"   # Output is a vector


class SEM2Operator(ABC):
    """" Base class for operators of SEM2. """
    model: SEM2
    input_type: OperatorInput
    output_type: OperatorOutput

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> tuple[torch.Tensor|any, dict]:
        """ Evaluates the operator. Returns a tuple of (output, metrics_dict). The output is the value of the operator,
        and the metrics_dict contains useful debug / tracking information.
        """
