"""Named potential-outcome samples and finite-state distributions.

The model.sample API is unchanged. Operators retain (output, auxiliary) returns;
output is a NamedBatch for draws or a FiniteDistribution for enumerated outcomes.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .operators_base import SEM2Operator, OperatorInput, OperatorOutput
from ..data import NamedBatch, VariableInfo, FiniteDistribution
from ..model.components.nodes import NodeMLPDiscrete
from ..model.construction import SEMType

if TYPE_CHECKING:
    from ..model.models import SEM2


def _interventions(model, batch_size, set_nodes):
    """Resolve named values and validate discrete values before sampler casts."""
    if isinstance(set_nodes, NamedBatch):
        set_nodes = {name: set_nodes.select(name) for name in set_nodes.names}
    result = {} if set_nodes is None else dict(set_nodes)
    unknown = set(result) - set(model.node_dict)
    if unknown:
        raise ValueError(f"Interventions contain unknown nodes: {sorted(unknown)}")
    for name, values in result.items():
        node = model.node_dict[name]
        if not isinstance(values, Tensor) or values.shape != (batch_size, node.out_dim):
            raise ValueError(f"Intervention {name!r} must have shape [batch_size, out_dim].")
        values = values.to(model.device)
        if isinstance(node, NodeMLPDiscrete):
            valid = ((values >= 0) & (values < node.n_classes) & (values == values.floor()))
            if values.shape[1] != 1 or not valid.all():
                raise ValueError(f"Intervention {name!r} contains an invalid discrete state.")
        elif node.dist is not None and not node.dist.support.check(values).all():
            raise ValueError(f"Intervention {name!r} is outside its declared domain.")
        result[name] = values
    return result


def _variables(model, names, supplied=()):
    info = {v.name: v for v in supplied}
    result = []
    for name in names:
        if name in info:
            result.append(info[name])
            continue
        node = model.node_dict[name]
        categorical = isinstance(node, NodeMLPDiscrete) or isinstance(node.dist, torch.distributions.Categorical)
        discrete = True if categorical else (node.dist.support.is_discrete if node.dist is not None else None)
        result.append(VariableInfo(name, True if categorical else None, discrete))
    return tuple(result)


class PotentialOutcome(SEM2Operator):
    """Evaluate one named outcome under optional named interventions.

    Scalar-valued variables only. Continuous evaluation stops at the target.
    Discrete evaluation marginalizes the full state table, retaining one
    probability row per sampled exogenous input. No refitting is performed.
    Pass existing VariableInfo objects to retain external-label encodings;
    these cannot be recovered from the model alone.
    """
    input_type = OperatorInput.DATA
    output_type = OperatorOutput.VECTOR

    def __init__(self, model: SEM2, target_node: str, *, variables: tuple[VariableInfo, ...] = ()):
        if target_node not in model.node_dict:
            raise ValueError(f"Unknown target node: {target_node!r}")
        self.model = model
        self.target_node = target_node
        if len({v.name for v in variables}) != len(variables):
            raise ValueError("Supplied variable metadata must have unique names.")
        self.variables = tuple(variables)
        if SEMType.is_discrete(model.sem_type):
            if target_node not in model.state_node_names:
                raise NotImplementedError("Discrete potential outcomes require an enumerated endogenous target.")
        elif not SEMType.is_continuous(model.sem_type):
            raise NotImplementedError(f"Potential outcomes are not implemented for {model.sem_type}.")

    def evaluate(
        self, batch_size: int = 1, set_nodes: dict[str, Tensor] | NamedBatch | None = None,
    ) -> tuple[NamedBatch | FiniteDistribution, dict]:
        set_nodes = _interventions(self.model, batch_size, set_nodes)
        extra = {"set_nodes": set_nodes}
        if SEMType.is_discrete(self.model.sem_type):
            probabilities, _ = self.model.sample(batch_size=batch_size, set_nodes=set_nodes)
            states = NamedBatch(
                self.model.all_states,
                _variables(self.model, self.model.state_node_names, self.variables),
            )
            return FiniteDistribution(states, probabilities).marginal(self.target_node), extra

        layer = self.model.node_dict
        for name in self.model.topological_order:
            node = layer[name]
            if name in set_nodes:
                node.set_value(set_nodes[name])
            else:
                node([layer[p].get_value() for p in node.predecessors], batch_size)
            if name == self.target_node:
                result = NamedBatch(node.get_value(), _variables(self.model, (name,), self.variables))
                return result, extra
        raise ValueError(f"Target {self.target_node!r} was not encountered during sampling.")
