# Defer annotation evaluation so forward references in type hints resolve lazily
# on Python < 3.14.
from __future__ import annotations
import warnings
import torch
from torch import nn
from torch import Tensor

from ..components.nodes import NodeExogenous, NodeLinear, NodeMLP, NodeMLPDiscrete, SEM2Node
from ..construction import _DAG, SEMType


# ════════════════════════════════════════════════════════════════════════════════════
# SEM2 - DAG + nn.Module
# ════════════════════════════════════════════════════════════════════════════════════

class SEM2(_DAG, nn.Module):
    """Structural Equation Model: a DAG whose nodes are PyTorch modules.

    Inherits graph structure and parsing from DAG, and parameter registration
    and training from nn.Module.
    """
    node_dict: nn.ModuleDict[str, SEM2Node]  # Maps node name to its SEM2Node object, registered as submodule for PyTorch tracking
    sem_type = SEMType.NONLINEAR

    def __init__(self, model_formula: str, node_cfgs:dict|None = None, device="cpu"):
        """
        Args:
            model_formula: Semicolon-separated formulas defining the SEM.
            node_cfgs: Optional dict mapping node name to its config.
                - Exogenous node : int (output dimension), default 1.
                - Linear node    : LinearNodeConfig.
                - MLP node       : MLPConfig.
            device: PyTorch device string.
        """
        _DAG.__init__(self, model_formula)
        nn.Module.__init__(self)
        self.device = device

        # Compare the formula's inferred type against the type this class expects,
        # warning (or, in subclasses, erroring) on a mismatch.
        self._check_sem_type()

        node_cfgs = node_cfgs if node_cfgs is not None else {}
        name_to_node = self._build_and_initialize_nodes(node_cfgs, device)

        assert len(node_cfgs) == 0, \
            f"The following nodes were given but not found in formula: {list(node_cfgs.keys())}"

        # Register all nodes as submodules so PyTorch tracks parameters
        self.node_dict = nn.ModuleDict(name_to_node)
        self.node_dict.to(device)

    def _check_sem_type(self):
        """Warn if the formula's inferred sem_type differs from the sem_type this
        class is specialised for, nudging the user toward a more suitable SEM2
        subclass. Subclasses (e.g. SEM2Discrete, SEM2Linear) may override this to
        raise instead, or to accept a family of compatible inferred types.

        self.sem_type is the class-level desired type; self.inferred_sem_type is
        what the DAG detected from the formula.
        """
        if self.inferred_sem_type != self.sem_type:
            if type(self) is SEM2 and SEMType.is_linear(self.inferred_sem_type):
                warnings.warn(
                    f"Detected sem_type '{self.inferred_sem_type.value}': you are using "
                    f"a linear SEM with SEM2. Best to use the special child class "
                    f"SEM2Linear since it has more features available.",
                    stacklevel=3,
                )
            else:
                warnings.warn(
                    f"Inferred sem_type '{self.inferred_sem_type.value}' differs from the "
                    f"'{self.sem_type.value}' expected by {type(self).__name__}. "
                    f"Consider using a more specialised SEM2 subclass.",
                    stacklevel=3,
                )

    def _build_and_initialize_nodes(self, node_cfgs: dict, device: str) -> dict[str, SEM2Node]:
        """Build all SEM2Node objects from DAG.node_specs in topological order.

        Because we iterate in topological order, every node's predecessors are
        guaranteed to already exist in name_to_node by the time we reach it.
        This means input_dim can be computed immediately and each node is fully
        constructed in a single pass.

        Pops configs from node_cfgs as nodes are built and any remaining keys
        after this call indicate unused / misspelled configs.

        For exogenous nodes the distribution is instantiated here from the
        already-validated dist_name and dist_params stored on the NodeSpec,
        keeping NodeExogenous.__init__ free of parsing logic.
        """
        name_to_node = {}

        for node_name in self.topological_order:
            ns = self.node_specs[node_name]
            cfg = node_cfgs.pop(node_name, None)
            predecessor_nodes = [name_to_node[p] for p in ns.predecessors]

            if ns.node_type == "exogenous":
                out_dim = cfg if isinstance(cfg, int) else 1
                # Instantiate the torch distribution from pre-validated spec fields
                dist_class = getattr(torch.distributions, ns.dist_name)
                dist_args, dist_kwargs = ns.dist_params
                dist_args = [
                    torch.tensor(p, device=device, dtype=torch.float32)
                    for p in dist_args
                ]
                dist_kwargs = {k: torch.tensor(v, device=device, dtype=torch.float32) for k, v in dist_kwargs.items()}
                dist = dist_class(*dist_args, **dist_kwargs, validate_args=True)
                node = NodeExogenous(node_name, ns.rhs, dist, out_dim=out_dim, device=device)

            elif ns.node_type == "mlp":
                # Good to note: rhs guaranteed by _expand_rhs to be flat n(a,b,...) since no nesting
                node = NodeMLP(node_name, ns.rhs, cfg, predecessor_nodes, device=device)

            elif ns.node_type == "discrete":
                node = NodeMLPDiscrete(node_name, ns.rhs, cfg, predecessor_nodes, device=device)

            elif ns.node_type == "linear":
                node = NodeLinear(
                    node_name, ns.rhs, ns.coefficients, cfg, predecessor_nodes, device=device
                )

            else:
                raise ValueError(f"Unknown node type: {ns.node_type}")

            name_to_node[node_name] = node

        return name_to_node

    def sample(self, batch_size: int = 1, set_nodes: dict[str, Tensor]|None = None) -> tuple[dict[str, Tensor], dict]:
        """ Forward pass of a continuous SEM2 model. Nodes are sampled from their respective distributions.

        Args:
            batch_size: Number of samples to draw.
            set_nodes: Optional dict of node name -> tensor to pin specific node
                       values (e.g. for interventions). Shape: [batch_size, out_dim].
        Return:
            {node_name: tensor}.
            {} Dummy dict, no metrics are generated.
        """
        layer = self.node_dict
        topological_order = self.topological_order

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

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"call the sample() method instead of forward() or calling the model directly.")

    def likelihood(self, *args, **kwargs):
        raise NotImplementedError(f"likelihood not implemented for this type of SEM2 model: {self.inferred_sem_type.value}")
