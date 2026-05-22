from typing import TYPE_CHECKING
import torch
from sem2.functionals.functional_base import SEM2VectorFunctional, SEM2SelfFunctional, SEM2DataFunctional

if TYPE_CHECKING:
    from model import SEM2
    from torch import Tensor


class SampleContinuous(SEM2DataFunctional, SEM2VectorFunctional):
    """ Sample from the SEM2 model by drawing samples from each node.
        This class allows specifying interventional / fixed nodes.
    """
    def __init__(self, model: SEM2):
        self.model = model

    def evaluate(
        self,
        batch_size: int = 1,
        set_nodes: dict[str, Tensor]|None = None,
    ) -> tuple[dict[str, Tensor], dict]:
        """Run a forward pass through the SEM.

        Args:
            batch_size: Number of samples to draw.
            set_nodes: Optional dict of node name -> tensor to pin specific node
                       values (e.g. for interventions). Shape: [batch_size, out_dim].
        Return:
            {node_name: tensor}.
            {} Dummy dict, no metrics are generated.
        """
        layer = self.model.layer
        topological_order = self.model.topological_order

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

        return {name: layer[name].get_value() for name in topological_order}, {}


class PotentialOutcome(SEM2DataFunctional, SEM2VectorFunctional):
    """ Get the potential outcome of a single node in the SEM2 model.
        This exits early as soon as the target node is computed.
    """
    def __init__(self, model: SEM2, target_node: str):
        self.model = model
        self.target_node = target_node

    def evaluate(
        self,
        batch_size: int = 1,
        set_nodes: dict[str, Tensor]|None = None,
    ) -> tuple[Tensor, dict]:
        """ Run a forward pass through the SEM until target_node is computed, then exit early.
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
