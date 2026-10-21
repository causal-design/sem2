"""SEM type taxonomy.

SEMType is descriptive metadata describing the structural form of a SEM
(linear/nonlinear, continuous/discrete). It is inferred from the formula by the
DAG layer and used to gate type-specific functionality and guide the user toward
a specialised SEM2 subclass.
"""
from __future__ import annotations
from enum import Enum


class SEMType(Enum):
    """Enum representing the inferred type of a SEM.

    Types:
        - LINEAR_GAUSSIAN: all linear endogenous nodes, all exogenous nodes are Normal
        - LINEAR: all linear endogenous nodes, with non-Gaussian exogenous nodes.
        - NONLINEAR: continuous endogenous mechanisms include at least one MLP.
        - DISCRETE: all endogenous nodes are discrete.
        - MIXED: some endogenous nodes are continuous, some are discrete.

    Notes
    -----
    The label describes formula structure, not observation status or fitted
    parameters. Model subclasses use it to reject incompatible mechanisms or
    suggest a specialization. Gaussian likelihood also checks scalar Normal
    exogenous distributions; the label alone is not sufficient.
    """
    LINEAR_GAUSSIAN     = "linear_gaussian"
    LINEAR              = "linear"
    NONLINEAR           = "nonlinear"
    DISCRETE            = "discrete"
    MIXED               = "mixed"

    @staticmethod
    def is_continuous(sem_type: SEMType) -> bool:
        """Return whether sem_type is LINEAR_GAUSSIAN, LINEAR or NONLINEAR."""
        return sem_type in {
            SEMType.LINEAR_GAUSSIAN,
            SEMType.LINEAR,
            SEMType.NONLINEAR,
        }

    @staticmethod
    def is_discrete(sem_type: SEMType) -> bool:
        """Return whether sem_type is DISCRETE, excluding MIXED models."""
        return sem_type in {
            SEMType.DISCRETE
        }

    @staticmethod
    def is_linear(sem_type: SEMType) -> bool:
        """ If SEM has only linear endogenous mechanisms (Gaussian or not)."""
        return sem_type in {
            SEMType.LINEAR_GAUSSIAN,
            SEMType.LINEAR,
        }
