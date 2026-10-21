# Defer annotation evaluation so forward references in type hints resolve lazily
# on Python < 3.14.
from __future__ import annotations
import warnings
from dataclasses import replace
import torch
from torch import nn
from torch import Tensor

from ..components.nodes import NodeExogenous, NodeLinear, NodeMLP, NodeMLPDiscrete, SEM2Node
from ..construction.dag import _DAG
from ..construction.sem_types import SEMType
from ...data.records import DataBatch, VariableInfo
from ...default_configs import LinearNodeConfig, MLPConfig


# ════════════════════════════════════════════════════════════════════════════════════
# SEM2 - DAG + nn.Module
# ════════════════════════════════════════════════════════════════════════════════════


# Docs here should correspond better to docs for the linear/discrete model? 
# Wording is currently quite different
# E.g. here: continuous SEM. Associatied with a DAG whose nodes are PyTorch modules.
# (see linear and discrete model definitions for the other comments)
class SEM2(_DAG, nn.Module):
    """Structural Equation Model: a DAG whose nodes are PyTorch modules.

    Inherits graph structure and parsing from DAG, and parameter registration
    and training from nn.Module.

    Attributes
    ----------
    node_dict : torch.nn.ModuleDict
        Executable nodes keyed by name, including compiler-generated helpers.
    node_specs : dict[str, NodeSpec]
        Structural specifications for the expanded graph.
    input_node : set[str]
        User-declared names, excluding helpers; not observation status.
    topological_order : list[str]
        Parent-before-child order including helpers.
    node_order : list[str]
        Canonical matrix order, copied from topological_order.
    graph : dict[str, list[str]]
        Node names mapped to ordered parent names.
    statements : list
        Typed syntax records produced by formula parsing.
    inferred_sem_type : SEMType
        Structural family inferred from the formula.

    Notes
    -----
    Use ``sample`` rather than calling the model directly: ``forward`` and the
    generic ``likelihood`` method raise ``NotImplementedError``. Distribution
    tensors are constructed on the requested device; construct the model there
    rather than relying on a later module move to relocate distribution objects.
    """
    node_dict: nn.ModuleDict[str, SEM2Node]  # Maps node name to its SEM2Node object, registered as submodule for PyTorch tracking
    sem_type = SEMType.NONLINEAR

    def __init__(self, model_formula: str,
                 node_cfgs: dict[str, int | LinearNodeConfig | MLPConfig] | None = None, device: str | torch.device = "cpu") -> None:
        """Construct and initialize the graph's executable nodes.

        Parameters
        ----------
        model_formula : str
            Semicolon-separated structural equations and exogenous distributions.
        node_cfgs : dict, optional
            Node names mapped to ``LinearNodeConfig``, ``MLPConfig``, or an
            integer output dimension for an exogenous node (default 1).
            Entries are consumed during construction; pass a copy to retain them.
        device : str or torch.device, default="cpu"
            Device for node parameters and exogenous distribution tensors.
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

    def _check_sem_type(self) -> None:
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

    def _build_and_initialize_nodes(self, node_cfgs: dict[str, int | LinearNodeConfig | MLPConfig],
                                    device: str | torch.device) -> dict[str, SEM2Node]:
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
            if ns.is_helper and node_name in node_cfgs:
                raise ValueError(f"Cannot configure internal helper node: {node_name!r}")
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
                # Category count belongs to the formula; copy the architecture config.
                cfg = replace(cfg if cfg is not None else MLPConfig(), output_dim=ns.levels)
                node = NodeMLPDiscrete(node_name, ns.rhs, cfg, predecessor_nodes, device=device)

            elif ns.node_type == "linear":
                node = NodeLinear(
                    node_name, ns.rhs, ns.coefficients, cfg, predecessor_nodes,
                    device=device, intercept=ns.intercept,
                )

            else:
                raise ValueError(f"Unknown node type: {ns.node_type}")

            name_to_node[node_name] = node

        return name_to_node

    def _variable_info(self, name: str) -> VariableInfo:
        node = self.node_dict[name]
        categorical = isinstance(node, NodeMLPDiscrete) or isinstance(node.dist, torch.distributions.Categorical)
        return VariableInfo(name, dim=node.out_dim, categorical=categorical)

    def _unpack_interventions(self, set_nodes: DataBatch | None, batch_size: int) -> dict[str, Tensor]:
        """Unpack a named intervention batch into per-node tensors.

        Input has shape [batch_size, sum(variable.dim)]; each dictionary value
        has shape [batch_size, node.out_dim]. None gives an empty dictionary.
        Check positive batch size, row count, user-declared names and matching
        dimensions. Do not broadcast, encode labels, apply numerical scores or
        validate support values; callers supply valid stored intervention values.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if set_nodes is None:
            return {}
        if not isinstance(set_nodes, DataBatch):
            raise TypeError("set_nodes must be a DataBatch.")
        if len(set_nodes) != batch_size:
            raise ValueError("Intervention rows must match batch_size.")
        values = {}
        for variable in set_nodes.variables:
            name = variable.name
            if name not in self.input_node:
                raise ValueError(f"Unknown or internal intervention node: {name!r}")
            node = self.node_dict[name]
            value = set_nodes.select(name)
            if variable.dim != node.out_dim:
                raise ValueError(f"Intervention dimension does not match {name!r}.")
            values[name] = value
        return values

    def sample(self, batch_size: int = 1, set_nodes: DataBatch | None = None) -> tuple[DataBatch, dict]:
        """Return named draws and intervention provenance, excluding helpers.

        Parameters
        ----------
        batch_size : int, default=1
            Positive number of draws.
        set_nodes : DataBatch, optional
            Intervention values with one row per draw and matching node widths.
            Supply valid stored values on the model device; no broadcasting,
            category encoding or support validation is performed.

        Returns
        -------
        draws : DataBatch
            Values of shape (batch_size, total_width), in topological order.
            Internal helper nodes are excluded.
        details : dict
            ``interventions`` contains the supplied ``set_nodes`` object.

        Notes
        -----
        Consumes randomness and updates cached node values. Interventions replace
        structural equations; they do not establish causal identification.
        """
        # Share the sampling loop with queries that stop at a specific target.
        return self._sample(batch_size, set_nodes)

    def _sample(self, batch_size: int, set_nodes: DataBatch | None,
                target: str | None = None) -> tuple[DataBatch, dict]:
        """One traversal for full samples and target-only queries.

        The private target option lets queries stop early without duplicating
        intervention handling or expanding the public sample interface.
        """
        if target is not None and target not in self.input_node:
            raise ValueError(f"Unknown or internal target node: {target!r}")
        # Unwrap interventions once and retain their supplied variable metadata.
        interventions = self._unpack_interventions(set_nodes, batch_size)
        supplied = {} if set_nodes is None else dict(zip(set_nodes.names, set_nodes.variables))
        values, variables = [], []
        # Evaluate parents first, replacing intervened equations with fixed values.
        for name in self.topological_order:
            node = self.node_dict[name]
            if name in interventions:
                node.set_value(interventions[name])
            else:
                node([self.node_dict[p].get_value() for p in node.predecessors], batch_size)
            # Keep compiler helpers out of the returned batch.
            if not self.node_specs[name].is_helper and (target is None or name == target):
                values.append(node.get_value())
                variables.append(supplied[name] if name in supplied else self._variable_info(name))
            if name == target:
                break
        # Stack variable blocks in the same order as their metadata.
        result = DataBatch(torch.cat(values, dim=-1), variables)
        return result, {"interventions": set_nodes}

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"call the sample() method instead of forward() or calling the model directly.")

    def likelihood(self, *args, **kwargs):
        raise NotImplementedError(f"likelihood not implemented for this type of SEM2 model: {self.inferred_sem_type.value}")
