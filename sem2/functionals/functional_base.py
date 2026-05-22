""" Defining the types of functionals / operators
    Functionals are defined based on:
        a) Inputs type
        b) Output type
"""
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import torch

if TYPE_CHECKING:
    from sem2.model import SEM2


class SEM2Functional(ABC):
    """" Base class for functionals of SEM2. """
    model: SEM2

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> tuple[torch.Tensor|any, dict]:
        """ Evaluates the functional. Returns a tuple of (output, metrics_dict). The output is the value of the functional, and the metrics_dict contains useful debug / tracking information.
        """


class SEM2FunctionalOutput(SEM2Functional, ABC):
    """ Defines the output type of a functional. """


class SEM2ScalarFunctional(SEM2FunctionalOutput, ABC):
    """ Takes in the SEM2 model, optional data and returns a scalar.
            SEM2 (+ data) -> R
            for SEM2, optional data V and R output scalar.
        Must always return a scalar value.
        An additional dictionary is returned that contains useful debug / tracking information.

        Commonly used for loss functions and evaluation metrics.
    """

class SEM2VectorFunctional(SEM2FunctionalOutput, ABC):
        """ Takes in the SEM2 model, optional data and returns a vector.
            SEM2 (+ data) -> R^d
            for SEM2, optional data V and R^d output vector.
        An additional dictionary is returned that contains useful debug / tracking information.
        Must always return a scalar value.
        """


class SEM2FunctionalInput(SEM2Functional, ABC):
    """ Defines the input type of a functional. """


class SEM2SelfFunctional(SEM2FunctionalInput, ABC):
    """ Takes in the SEM2 model, and returns anything.
            SEM2 -> any
        An additional dictionary is returned that contains useful debug / tracking information.

        Evaluates a property of the SEM2 model itself, without any external data or inputs.
    """


class SEM2DataFunctional(SEM2FunctionalInput, ABC):
    """ Takes in the SEM2 model, data and returns anything.
        SEM2 + data -> any
    An additional dictionary is returned that contains useful debug / tracking information.
    Must always return a scalar value.
"""


