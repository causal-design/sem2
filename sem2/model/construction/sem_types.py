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
        - NONLINEAR: all endogenous nodes are MLPs.
        - DISCRETE: all endogenous nodes are discrete.
        - MIXED: some endogenous nodes are continuous, some are discrete.

    Note: sem_type is descriptive metadata. It does not change core SEM
    behaviour. Its intended future use is to gate type-specific functionals
    (e.g. closed-form log-likelihood for LINEAR_GAUSSIAN) and guide the user
    toward a specialised implementation when one exists.
    """
    LINEAR_GAUSSIAN     = "linear_gaussian"
    LINEAR              = "linear"
    NONLINEAR           = "nonlinear"
    DISCRETE            = "discrete"
    MIXED               = "mixed"

    @staticmethod
    def is_continuous(sem_type: SEMType):
        """ If SEM is purely continuous"""
        return sem_type in {
            SEMType.LINEAR_GAUSSIAN,
            SEMType.LINEAR,
            SEMType.NONLINEAR,
        }

    @staticmethod
    def is_discrete(sem_type: SEMType):
        """ If SEM is purely discrete"""
        return sem_type in {
            SEMType.DISCRETE
        }

    @staticmethod
    def is_linear(sem_type: SEMType):
        """ If SEM has only linear endogenous mechanisms (Gaussian or not)."""
        return sem_type in {
            SEMType.LINEAR_GAUSSIAN,
            SEMType.LINEAR,
        }
