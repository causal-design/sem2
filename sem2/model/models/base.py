# Defer annotation evaluation so forward references in type hints resolve lazily
# on Python < 3.14.
from __future__ import annotations
import warnings
import torch
from torch import nn

from ..components.nodes import NodeExogenous, NodeLinear, NodeMLP, NodeMLPDiscrete, SEM2Node
from ..construction import _DAG, SEMType
from ...data import NamedBatch, VariableInfo, FiniteDistribution


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

    def _variable_info(self, names, supplied=()) -> tuple[VariableInfo, ...]:
        """Describe stored values without redefining the nodes' domains."""
        info = {v.name: v for v in supplied}
        result = []
        for name in names:
            node = self.node_dict[name]
            if name in info:
                if info[name].width != node.out_dim:
                    raise ValueError(f"Variable {name!r} width does not match its model node.")
                result.append(info[name])
                continue
            categorical = isinstance(node, NodeMLPDiscrete) or isinstance(node.dist, torch.distributions.Categorical)
            discrete = True if categorical else (node.dist.support.is_discrete if node.dist is not None else None)
            result.append(VariableInfo(name, True if categorical else None, discrete, width=node.out_dim))
        return tuple(result)

    def _interventions(self, batch_size: int, set_nodes: NamedBatch | None) -> NamedBatch | None:
        """Validate named interventions before any node casts or sampling."""
        if set_nodes is None:
            return None
        if not isinstance(set_nodes, NamedBatch):
            raise TypeError("set_nodes must be a NamedBatch, not a tensor or dictionary.")
        unknown = set(set_nodes.names) - set(self.node_dict)
        if unknown:
            raise ValueError(f"Interventions contain unknown nodes: {sorted(unknown)}")
        set_nodes = set_nodes.to(self.device)
        for name in set_nodes.names:
            node = self.node_dict[name]
            values = set_nodes.select(name)
            if values.shape != (batch_size, node.out_dim):
                raise ValueError(f"Intervention {name!r} must have shape [batch_size, out_dim].")
            if isinstance(node, NodeMLPDiscrete):
                valid = (values >= 0) & (values < node.n_classes)
                if values.is_floating_point():
                    valid = valid & (values == values.floor())
                if not valid.all():
                    raise ValueError(f"Intervention {name!r} contains an invalid discrete state.")
            elif node.dist is not None and not node.dist.support.check(values).all():
                raise ValueError(f"Intervention {name!r} is outside its declared domain.")
        return set_nodes

    def sample(self, batch_size: int = 1, set_nodes: NamedBatch | None = None) -> tuple[NamedBatch | FiniteDistribution, dict]:
        """Return named draws in topological order and named intervention provenance.

        ``set_nodes`` is a NamedBatch with ``batch_size`` rows. Each variable's
        width must match its node's output dimension. Numerical node buffers
        remain internal; there is no dictionary or positional-tensor data API.
        """
        return self._sample_nodes(batch_size, set_nodes)

    def _sample_nodes(self, batch_size, set_nodes, target_node=None):
        """Shared sampling engine; causal queries may stop at their target."""
        set_nodes = self._interventions(batch_size, set_nodes)
        interventions = {} if set_nodes is None else {n: set_nodes.select(n) for n in set_nodes.names}
        layer, names, values = self.node_dict, [], []
        for name in self.topological_order:
            node = layer[name]
            if name in interventions:
                node.set_value(interventions[name])
            else:
                node([layer[p].get_value() for p in node.predecessors], batch_size)
            names.append(name)
            values.append(node.get_value())
            if name == target_node:
                break
        supplied = () if set_nodes is None else set_nodes.variables
        result = NamedBatch(torch.cat(values, dim=-1), self._variable_info(names, supplied))
        return result, {"set_nodes": set_nodes}

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"call the sample() method instead of forward() or calling the model directly.")

    def likelihood(self, *args, **kwargs):
        raise NotImplementedError(f"likelihood not implemented for this type of SEM2 model: {self.inferred_sem_type.value}")
