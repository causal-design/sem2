import warnings

import torch
from torch import nn
from torch import Tensor

from .nn_models import LinearNodeConfig
from .nodes import NodeExogenous, NodeLinear, NodeMLP, NodeMLPDiscrete
from .sem_parsing import DAG


# ══════════════════════════════════════════════════════════════════════════════
# SEM2 - DAG + nn.Module
# ══════════════════════════════════════════════════════════════════════════════

class SEM2(DAG, nn.Module):
    """Structural Equation Model: a DAG whose nodes are PyTorch modules.

    Inherits graph structure and parsing from DAG, and parameter registration
    and training from nn.Module.
    """

    def __init__(self, model_formula: str, node_cfgs: dict = None, device="cpu"):
        """
        Args:
            model_formula: Semicolon-separated formulas defining the SEM.
            node_cfgs: Optional dict mapping node name to its config.
                - Exogenous node : int (output dimension), default 1.
                - Linear node    : LinearNodeConfig.
                - MLP node       : MLPConfig.
            device: PyTorch device string.
        """
        DAG.__init__(self, model_formula)
        nn.Module.__init__(self)
        self.device = device

        node_cfgs = node_cfgs if node_cfgs is not None else {}
        name_to_node = self._build_and_initialize_nodes(node_cfgs, device)

        assert len(node_cfgs) == 0, \
            f"The following nodes were given but not found in formula: {list(node_cfgs.keys())}"

        # Register all nodes as submodules so PyTorch tracks parameters
        self.layer = nn.ModuleDict(name_to_node)
        self.layer.to(device)

        # Inform the user what type of SEM they have built
        warnings.warn(
            f"[SEM2] Model '{model_formula}' detected as sem_type='{self.sem_type.value}'.",
            stacklevel=2,
        )

    def _build_and_initialize_nodes(self, node_cfgs: dict, device: str) -> dict:
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
        layer = self.layer
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
        raise NotImplementedError(f"likelihood not implemented for this type of SEM2 model: {self.sem_type.value}")


class SEM2Discrete(SEM2):
    """Discrete SEM2."""
    def __init__(self, model_formula: str, node_cfgs: dict = None, device="cpu"):
        super().__init__(model_formula, node_cfgs, device=device)
        self.device = device

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
    formula2 = "D <- d(Z, U, eps_d); Y <- d(D, U, eps_y)"
    mlp_cfgs = {"U": 2, "eps_d": 2, "eps_y": 2,
                "D": MLPConfig(input_dim=1 + 2 + 2, hidden_dims=[1024, 1024, 1024], output_dim=2),
                "Y": MLPConfig(input_dim=1 + 2 + 2, hidden_dims=[1024, 1024, 1024], output_dim=3)}

    model = SEM2(formula1 + formula2, node_cfgs=mlp_cfgs, device="cpu")
    model.eval()
    all_states, _ = model.sample(batch_size=2)
    print("\n=== Discrete SEM, with sampling over discrete nodes ===")

    print(f'{all_states = }')


if __name__ == "__main__":
    main()
