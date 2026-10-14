# Defer annotation evaluation so forward references in type hints resolve lazily
# on Python < 3.14.
from __future__ import annotations
import logging
import torch
from torch import nn
from torch import Tensor
from collections import defaultdict

from ..components.nodes import NodeMLPDiscrete, SEM2Node, DiscreteModes
from ..construction.sem_types import SEMType
from .base import SEM2
from ...data.records import DataBatch
from ..distributions import DiscreteJointDistribution
from ...default_configs import LinearNodeConfig, MLPConfig


class SEM2Discrete(SEM2):
    """Discrete SEM2 with enumeration of endogenous joint states.

    Parameters
    ----------
    model_formula : str
        Exogenous laws and discrete equations using ``d(parents; levels=N)``.
    node_cfgs : dict, optional
        Per-node configurations, consumed during construction. Discrete nodes
        use ``MLPConfig``; category counts come from the formula.
    device : str or torch.device, default="cpu"
        Device for model parameters, states and probabilities.

    Notes
    -----
    Enumeration grows as the product of endogenous category counts. Sampling
    integrates these states rather than drawing realized categorical outcomes.
    """
    node_dict: nn.ModuleDict[str, SEM2Node]  # Maps node name to its SEM2Node object, registered as submodule for PyTorch tracking
    sem_type = SEMType.DISCRETE

    def _check_sem_type(self) -> None:
        """A discrete SEM requires discrete endogenous nodes. Validate against the
        type inferred from the formula (not the class's desired type) so a non-
        discrete formula is rejected up front rather than failing later in sampling.
        """
        if not SEMType.is_discrete(self.inferred_sem_type):
            raise ValueError(
                f'SEM2Discrete can only be used with a discrete model. '
                f'Found "{self.inferred_sem_type.value}" model.'
            )
        logging.info("Building a discrete SEM2 model.")

    def __init__(self, model_formula: str,
                 node_cfgs: dict[str, int | LinearNodeConfig | MLPConfig] | None = None, device: str | torch.device = "cpu") -> None:
        super().__init__(model_formula, node_cfgs, device=device)
        self.device = device

        # Prepare for all possible configurations of NN nodes for sampling probabilities.
        # Number of NN nodes and their output dimensions
        N_nn_nodes = 0
        n_classes = []  # List of output dimensions for each NN_dist node
        node_names = []   # List of node names corresponding to each column in all_states
        for node_name, node in self.node_dict.items():
            if isinstance(node, NodeMLPDiscrete):
                node.id = N_nn_nodes
                N_nn_nodes += 1
                n_classes.append(node.n_classes)
                node_names.append(node.name)

        self.output_dims = n_classes  # Store for later use

        # Generate all possible discrete configurations of the NN nodes (Cartesian product).
        # N_conf = product of all output_dims
        if N_nn_nodes == 0:
            self.N_conf = 1
            all_states = torch.zeros((1, 0), dtype=torch.long, device=self.device)
        else:
            self.N_conf = 1
            for dim in n_classes:
                self.N_conf *= dim

            # Generate Cartesian product of all possible states
            # all_states shape: [N_conf, N_nn_nodes]
            ranges = [torch.arange(dim, device=self.device) for dim in n_classes]
            grids = torch.meshgrid(*ranges, indexing='ij')
            all_states = torch.stack([g.flatten() for g in grids], dim=1).to(self.device)  # [N_conf, N_nn_nodes]

        # Track order of output nodes.
        self.state_node_names = node_names  # List of node names, where index corresponds to column in all_states
        self.all_states = torch.nn.parameter.Buffer(all_states) # Which state each confguration corresponds to

        # Generate fixed ordering of discrete node parents
        self.disc_parents: dict[str, list[NodeMLPDiscrete]] = defaultdict(list)
        self.cont_parents: dict[str, list[SEM2Node]] = defaultdict(list)
        for node_name in self.topological_order:
            node = self.node_dict[node_name]
            if isinstance(node, NodeMLPDiscrete):
                for p in node.predecessors:
                    pn = self.node_dict[p]
                    if isinstance(pn, NodeMLPDiscrete):
                        self.disc_parents[node_name].append(pn)
                    else:
                        self.cont_parents[node_name].append(pn)

    def sample(self, batch_size: int = 1, set_nodes: DataBatch | None = None) -> tuple[DiscreteJointDistribution, dict]:
        """Enumerate discrete states and compute their conditional probabilities.

        Exogenous inputs are sampled; the returned DiscreteJointDistribution
        integrates discrete outcomes rather than drawing categorical observations.

        Parameters
        ----------
        batch_size : int, default=1
            Positive number of exogenous Monte Carlo draws.
        set_nodes : DataBatch, optional
            Named intervention values with one row per draw. Supply valid state
            IDs or exogenous values on the model device, with matching widths.

        Returns
        -------
        distribution : DiscreteJointDistribution
            Named endogenous states and probabilities of shape
            (batch_size, n_states). Internal helper states are marginalized out.
        details : dict
            ``interventions`` contains the supplied ``set_nodes`` object.

        Notes
        -----
        Consumes randomness and updates cached node values. Average probability
        rows to estimate the marginal law over exogenous inputs.
        """

        # Unwrap valid interventions before enumerating conditional probabilities.
        interventions = self._unpack_interventions(set_nodes, batch_size)

        device = self.device
        all_states = self.all_states  # [N_conf, N_discrete_nodes]
        N_conf = self.N_conf

        # Track in log-space to reduce underflow from many probability multiplications.
        log_probs = torch.zeros((batch_size, N_conf), dtype=torch.float, device=device)
        neg_inf = torch.tensor(-torch.inf, dtype=log_probs.dtype, device=device)

        for node_name in self.topological_order:
            node = self.node_dict[node_name]
            is_discrete = isinstance(node, NodeMLPDiscrete)
            # An intervention replaces this node's probability factor with a point mass.
            if node_name in interventions:
                node.set_value(interventions[node_name])

                if is_discrete:
                    # mask out configurations that don't match the intervened discrete state
                    # all_states[:, v.id] : [N_conf]
                    target = interventions[node_name].long().view(batch_size, 1)  # [bs, 1]
                    col = all_states[:, node.id].view(1, N_conf).expand(batch_size, N_conf)  # [bs, N_conf]
                    mask = (col == target)  # [bs, N_conf]
                    log_probs = torch.where(mask, log_probs, neg_inf)
                continue

            # ---- Build parent_values ----
            if not is_discrete:
                # Non-discrete nodes are sampled like normal
                parent_names = list(node.predecessors)
                parent_values = [self.node_dict[p].get_value() for p in parent_names]
                node(parent_values, batch_size)
                continue
            # node.type == "NodeMLPDiscrete"
            disc_parents = self.disc_parents[node_name]
            cont_parents = self.cont_parents[node_name]

            # 1) Build context keys from *discrete parents only*
            if len(disc_parents) == 0:
                # only one context
                unique_ctx = all_states.new_zeros((1, 0))
                inverse = all_states.new_zeros((N_conf,), dtype=torch.long)
                M = 1
            else:
                disc_ids = torch.tensor([pn.id for pn in disc_parents], device=device, dtype=torch.long)  # [k]
                key = all_states.index_select(1, disc_ids)  # [N_conf, k]
                unique_ctx, inverse = torch.unique(key, dim=0, return_inverse=True)
                M = unique_ctx.shape[0]

            # 2) Build parent_values for M contexts (NOT N_conf)
            parent_values = {}

            # discrete parents: [bs, M, 1] float (as your code expects)
            for j, pn in enumerate(disc_parents):
                col = unique_ctx[:, j].to(torch.float32).view(1, M, 1).expand(batch_size, M, 1)
                parent_values[pn.name] = col

            # continuous parents: [bs, M, dim] (expand, not repeat)
            for pn in cont_parents:
                p_val = pn.get_value()  # [bs, dim]
                p_val = p_val.unsqueeze(1).expand(batch_size, M, p_val.shape[-1])  # [bs, M, dim]
                parent_values[pn.name] = p_val

            # Evaluate each distinct parent context in the formula's declared order.
            node([parent_values[p] for p in node.predecessors], batch_size, mode=DiscreteModes.SOFTMAX)

            # 4) Compute cond prob for each global configuration using inverse map
            # p_unique: [bs, M, n_outcome]
            p_unique = node.get_value().reshape(batch_size, M, -1)

            # which outcome/state this node takes in each global config:
            # state_idx_conf: [N_conf]
            state_idx_conf = all_states[:, node.id].long()

            # map p_unique back to per-config probabilities:
            # p_per_conf: [bs, N_conf, n_outcome]
            p_per_conf = p_unique[:, inverse, :]  # uses inverse to expand logically

            # pick the probability of the realized state for each config:
            idx = state_idx_conf.view(1, N_conf, 1).expand(batch_size, N_conf, 1)
            cond_prob = torch.gather(p_per_conf, dim=-1, index=idx).squeeze(-1)  # [bs, N_conf]

            log_cond_prob = torch.log(cond_prob)
            log_probs = log_probs + log_cond_prob

        probs = torch.exp(log_probs)

        # Pair probability columns with their named states and preserve supplied metadata.
        supplied = {} if set_nodes is None else dict(zip(set_nodes.names, set_nodes.variables))
        variables = [supplied[name] if name in supplied else self._variable_info(name)
                     for name in self.state_node_names]
        result = DiscreteJointDistribution(DataBatch(all_states, variables), probs)
        names = [name for name in self.state_node_names if not self.node_specs[name].is_helper]
        if len(names) != len(self.state_node_names):
            # Sum over internal helpers rather than exposing their state coordinates.
            result = result.marginalize(names)
        return result, {"interventions": set_nodes}
