from typing import TYPE_CHECKING
import torch
from .operators_base import SEM2VectorOperator, SEM2SelfOperator, SEM2DataOperator
from ..model import SEMType

if TYPE_CHECKING:
    from torch import Tensor
    from ..model import SEM2


class PotentialOutcome(SEM2DataOperator, SEM2VectorOperator):
    """ Get the potential outcome of a single node in the SEM2 model.
        This exits early as soon as the target node is computed.
    """
    def __init__(self, model: SEM2, target_node: str):
        self.model = model
        self.target_node = target_node

        # Select the correct forward method based on the SEM type
        sem_type = self.model.sem_type
        if SEMType.is_continuous(sem_type):
            # Continuous node
            self.evaluate = self._evaluate_continuous
        else:
            raise NotImplementedError(f"Potential Outcome sampling is not implemented for this SEM type: {sem_type}")

    def _evaluate_continuous(self, batch_size: int = 1, set_nodes: dict[str, Tensor]|None = None) -> tuple[Tensor, dict]:
        """ Run a forward pass through a continuous SEM2 model, stopping at the target node.
            Supports interventions.

        Args:
            batch_size: Number of samples to draw.
            set_nodes: Optional dict of node name -> tensor to pin specific node
                       values (e.g. for interventions). Shape: [batch_size, out_dim].
        Return:
            The computed tensor for the target node.
            {} Dummy dict, no metrics are generated.
        """
        layer = self.model.layer
        topological_order = self.model.topological_order

        assert self.target_node in layer, f"target_node {self.target_node} not found in model"

        set_nodes = set_nodes if set_nodes is not None else {}
        assert set(set_nodes.keys()).issubset(layer.keys()), \
            f"set_nodes contains unknown nodes: {set(set_nodes.keys()) - set(layer.keys())}"

        for node_name in topological_order:
            v = layer[node_name]
            if node_name in set_nodes:
                v.set_value(set_nodes[node_name])
            else:
                pred_values = [layer[p].get_value() for p in v.predecessors]
                v(pred_values, batch_size)

            if node_name == self.target_node:
                return layer[node_name].get_value(), {}

        raise ValueError(f'Target node {self.target_node} was not encountered during sampling. This should not happen. ')

    def evaluate(self, batch_size: int = 1, set_nodes: dict[str, Tensor]|None = None) -> tuple[dict[str, Tensor], dict]:
        """ Run a forward pass through the SEM, stopping at the target node.
            This is a placeholder that gets replaced by the appropriate method in __init__ based on the SEM type."""
        raise NotImplementedError("Sampling is not implemented for this SEM type.")
