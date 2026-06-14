import logging
import torch
from torch import nn
from torch import Tensor

from .nodes import NodeExogenous, NodeLinear, NodeMLP, NodeMLPDiscrete, SEM2Node, DiscreteModes
from .sem_parsing import _DAG, SEMType


# ══════════════════════════════════════════════════════════════════════════════
# SEM2 - DAG + nn.Module
# ══════════════════════════════════════════════════════════════════════════════

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
        _DAG.__init__(self, model_formula, self.sem_type)
        nn.Module.__init__(self)
        self.device = device

        node_cfgs = node_cfgs if node_cfgs is not None else {}
        name_to_node = self._build_and_initialize_nodes(node_cfgs, device)

        assert len(node_cfgs) == 0, \
            f"The following nodes were given but not found in formula: {list(node_cfgs.keys())}"

        # Register all nodes as submodules so PyTorch tracks parameters
        self.node_dict = nn.ModuleDict(name_to_node)
        self.node_dict.to(device)

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


class SEM2Discrete(SEM2):
    """Discrete SEM2."""
    node_dict: nn.ModuleDict[str, SEM2Node]  # Maps node name to its SEM2Node object, registered as submodule for PyTorch tracking
    sem_type = SEMType.DISCRETE

    def __init__(self, model_formula: str, node_cfgs: dict = None, device="cpu"):
        super().__init__(model_formula, node_cfgs, device=device)
        self.device = device

        # Check if the model is discrete
        if self.sem_type != SEMType.DISCRETE:
            raise ValueError(f'SEM2Discrete can only be used with a discrete model. Found "{self.sem_type.value}" model.')
        else:
            logging.info(f"Building a discrete SEM2 model.")

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

    def sample(self, batch_size=1, set_nodes: dict[str, torch.Tensor] = None) -> tuple[Tensor, dict]:
        """ Discrete sampling: Sample every possible discrete outcome.
            This involves sampling continuous variables, and computing every permutation of discrete variables.

            Args:
                batch_size: Number of samples to draw.
                set_nodes: Optional dict of node name -> tensor to pin specific node
                           values (e.g. for interventions). Shape: [batch_size, out_dim].
        """

        set_nodes = set_nodes if set_nodes is not None else {}
        assert set(set_nodes.keys()).issubset(self.node_dict.keys()), f"Invalid nodes given: {set_nodes.keys()}"

        device = self.device
        all_states = self.all_states  # [N_conf, N_discrete_nodes]
        N_conf = self.N_conf

        # Track in log-space to reduce underflow from many probability multiplications.
        log_probs = torch.zeros((batch_size, N_conf), dtype=torch.float, device=device)
        neg_inf = torch.tensor(-torch.inf, dtype=log_probs.dtype, device=device)

        for node_name in self.topological_order:
            node = self.node_dict[node_name]
            is_discrete = isinstance(node, NodeMLPDiscrete)
            # ---- Intervention handling ----
            if node_name in set_nodes:
                node.set_value(set_nodes[node_name])

                if is_discrete:
                    # mask out configurations that don't match the intervened discrete state
                    # all_states[:, v.id] : [N_conf]
                    target = set_nodes[node_name].long().view(batch_size, 1)  # [bs, 1]
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

            # v.type == "NodeMLPDiscrete"
            disc_parents = []
            cont_parents = []
            for p in node.predecessors:
                pn = self.node_dict[p]
                if isinstance(pn, NodeMLPDiscrete):
                    disc_parents.append(pn)
                else:
                    cont_parents.append(pn)

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
            parent_values = []

            # discrete parents: [bs, M, 1] float (as your code expects)
            for j, pn in enumerate(disc_parents):
                col = unique_ctx[:, j].to(torch.float32).view(1, M, 1).expand(batch_size, M, 1)
                parent_values.append(col)

            # continuous parents: [bs, M, dim] (expand, not repeat)
            for pn in cont_parents:
                p_val = pn.get_value()  # [bs, dim]
                p_val = p_val.unsqueeze(1).expand(batch_size, M, p_val.shape[-1])  # [bs, M, dim]
                parent_values.append(p_val)

            # 3) Run NN only for M contexts
            # IMPORTANT: v(...) must accept parent_values with second dim = M
            node(parent_values, batch_size, mode=DiscreteModes.SOFTMAX)

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

            log_cond_prob = torch.log(cond_prob.clamp(min=1e-5))
            log_probs = log_probs + log_cond_prob
            # if not torch.all(torch.isfinite(log_probs)):
            #     print("Warning: All probabilities are zero. Check for underflow or incorrect model specification.")
        probs = torch.exp(log_probs)

        # Return extra info
        extra_dict = {"states": all_states}
        extra_dict["full_state"] = {node: self.node_dict[node].get_value() for node in self.topological_order}
        return probs, extra_dict

# ══════════════════════════════════════════════════════════════════════════════
# TO be moved to integration tests later I think
def main():
    from ..default_configs import MLPConfig

    # ── Example 1: Linear SEM ─────────────────────────────────────────────────

    # Fixed coefficients on x1 and x2, learned weight on noise.
    # Example of how to edit Public Linear Config if needed for the experiment: use_bias=False forces the relationship through the origin 
    # Otherwise usually dont need touch LinearNodeConfig at all, dont need to specify and just use defaults it should work ok.
    print("=== Linear SEM ===")
    linear_model = SEM2(
        "x1, x2, noise ~ Normal(0,1); y <- 0.5*x1 + -1*x2 + noise",
        node_cfgs={"y": LinearNodeConfig(use_bias=False)},
    )
    print(linear_model.sample(batch_size=3))

    # ── Example 2: Pure MLP SEM ───────────────────────────────────────────────
    
    # All inputs enter a nonlinear MLP, output is y.
    print("\n=== MLP SEM ===")
    mlp_model = SEM2(
        "x1, x2, noise ~ Normal(0,1); y <- n(x1, x2, noise)",
        node_cfgs={"y": MLPConfig(hidden_dims=[16, 16], output_dim=1)},
    )
    print(mlp_model.sample(batch_size=3))

    # ── Example 3: Mixed linear + MLP SEM (implicit intermediate node) ────────

    # When a formula mixes a nonlinear term and a linear term with '+', _expand_rhs automatically lifts the n(...) part into a separate 
    # intermediate node named "y:0" (MLP) and rewrites y as a linear sum of y:0 and noise. This intermediate node is an internal detail the user never 
    # names or configures it, it uses default MLPConfig.
    #
    # This mixed syntax is supported for convenience but makes the graph harder to inspect. 
    # Honestly, prefer Example 2's way of EXPLICIT style formula typing and that recommended for as best practice:
    #        where routing all inputs into one named MLP node because is clearer and easier to configure. 
    # Have implemented this to just have the option where needed if someone wants the convinence. 
    # (Potentially useful for both power users and less technical users)
    print("\n=== Mixed Linear + MLP SEM (implicit intermediate node) ===")
    mixed_model = SEM2(
        "x1, x2, noise ~ Normal(0,1); y <- n(x1, x2) + 1*noise",
    )
    print(mixed_model.sample(batch_size=3))

    # Example 4: Mixed discrete SEM, with sampling over discrete nodes
    formula1 = "U ~ Normal(0,1); eps_d ~ Uniform(0,1); eps_y ~ Uniform(0,1); Z ~ Categorical(probs=[1, 1, 1]);"
    formula2 = "X <- d(Z, U, eps_d); Y <- d(X, U, eps_y)"
    mlp_cfgs = {"U": 2, "eps_d": 2, "eps_y": 2,
                "X": MLPConfig(input_dim=1 + 2 + 2, hidden_dims=[1024, 1024, 1024], output_dim=2),
                "Y": MLPConfig(input_dim=1 + 2 + 2, hidden_dims=[1024, 1024, 1024], output_dim=3)}

    model = SEM2(formula1 + formula2, node_cfgs=mlp_cfgs, device="cpu")
    model.eval()
    all_states, _ = model.sample(batch_size=2)
    print("\n=== Discrete SEM, with sampling over discrete nodes ===")

    print(f'{all_states = }')


if __name__ == "__main__":
    main()
