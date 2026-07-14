"""SEM2 model package: concrete model classes and the construction frontend.

Public surface:
    from sem2.model import SEM2, SEM2Discrete, SEM2Linear, SEMType
"""
from __future__ import annotations

import sys
from types import ModuleType

from . import construction as _construction
from .components import nn_models as _nn_models
from .components import nodes as _nodes
from .construction import SEMType, _DAG
from .models import SEM2, SEM2Discrete, SEM2Linear

__all__ = ["SEM2", "SEM2Discrete", "SEM2Linear", "SEMType"]


def _install_compat_modules() -> None:
    """Keep old notebook imports working without keeping redirect files."""
    package = __name__

    old_model = ModuleType(f"{package}.model")
    old_model.SEM2 = SEM2
    old_model.SEM2Discrete = SEM2Discrete
    old_model.SEM2Linear = SEM2Linear
    old_model._DAG = _DAG
    old_model.SEMType = SEMType
    old_model.__all__ = ["SEM2", "SEM2Discrete", "SEM2Linear", "_DAG", "SEMType"]

    _nn_models.MLP = _nn_models._MLP
    _nn_models.__all__ = [
        "ACTIVATION_FUNCTIONS",
        "MLP",
        "MLPConfig",
        "_LinearConfig",
        "_LinearModel",
        "_MLP",
    ]

    sys.modules.setdefault(f"{package}.model", old_model)
    sys.modules.setdefault(f"{package}.nodes", _nodes)
    sys.modules.setdefault(f"{package}.nn_models", _nn_models)
    sys.modules.setdefault(f"{package}.sem_parsing", _construction)


_install_compat_modules()
