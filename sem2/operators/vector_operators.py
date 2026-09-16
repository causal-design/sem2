"""Potential outcomes use the same named result formats as model sampling."""
from __future__ import annotations
from typing import TYPE_CHECKING

from .operators_base import SEM2Operator, OperatorInput, OperatorOutput
from ..data import NamedBatch, VariableInfo, FiniteDistribution
from ..model.construction import SEMType

if TYPE_CHECKING:
    from ..model.models import SEM2


class PotentialOutcome(SEM2Operator):
    """Evaluate one named outcome under optional named interventions.

    Continuous evaluation stops at the target and retains its full width.
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
        self.variables = tuple(v for v in variables if v.name == target_node)
        model._variable_info((target_node,), self.variables)  # Validate supplied width up front.
        if SEMType.is_discrete(model.sem_type):
            if target_node not in model.state_node_names:
                raise NotImplementedError("Discrete potential outcomes require an enumerated endogenous target.")
        elif not SEMType.is_continuous(model.sem_type):
            raise NotImplementedError(f"Potential outcomes are not implemented for {model.sem_type}.")

    def evaluate(
        self, batch_size: int = 1, set_nodes: NamedBatch | None = None,
    ) -> tuple[NamedBatch | FiniteDistribution, dict]:
        if SEMType.is_discrete(self.model.sem_type):
            result, extra = self.model.sample(batch_size=batch_size, set_nodes=set_nodes)
            result = result.marginal(self.target_node)
            states = NamedBatch(result.states.values, self.variables or result.states.variables)
            return FiniteDistribution(states, result.probabilities), extra

        result, extra = self.model._sample_nodes(batch_size, set_nodes, self.target_node)
        result = result.subset(self.target_node)
        return NamedBatch(result.values, self.variables or result.variables), extra
