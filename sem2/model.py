import re
from dataclasses import dataclass, field
from graphlib import TopologicalSorter, CycleError
import torch
from torch import nn

from .nn_models import LinearNodeConfig
from .nodes import NodeExogenous, NodeLinear, NodeMLP

#things to change in this file : Move main to integration tests. Got 3 examples for now to show the interface

# ══════════════════════════════════════════════════════════════════════════════
# DAG
# ══════════════════════════════════════════════════════════════════════════════

class DAG:
    """Directed Acyclic Graph built from a model formula string.

    Owns all formula parsing and topological sorting. Stores only raw graph
    info (node names, rhs strings, predecessor lists, fixed coefficients)
    So no tensors, no device, no SEM2Node objects. (no pytorch related things basically)

    Attributes:
        model_formula:     The original formula string.
        formula_list:      Expanded, one-node-per-entry formula list.
        input_node:        Set of node names that appear on any LHS.
        node_specs:        Dict of node_name -> NodeSpec. Single source of
                           truth for graph structure — built once, read by SEM2.
        graph:             Dict of node_name -> list of predecessor names.
                           Derived from node_specs for topological sorting.
        topological_order: Node names in topologically sorted order.
    """

    @dataclass
    class NodeSpec:
        """Parsed specification of a single node from the formula string.
        NESTED inside DAG as it is a DAG implementation detail it is not part
        of the public API and should NOT be used outside of DAG and SEM2.

        Fields:
            name:         Node name (LHS of formula).
            rhs:          Raw RHS string after expansion.
            node_type:    One of "exogenous", "linear", "mlp".
            predecessors: Ordered list of predecessor node names.
            coefficients: One entry per predecessor for linear nodes.
                            float -> fixed at that value (e.g. 1.0, -0.5)
                            None  -> learned weight
                          Empty list for exogenous and mlp nodes.
        """
        name: str
        rhs: str
        node_type: str
        predecessors: list[str]
        coefficients: list[float | None] = field(default_factory=list)

    def __init__(self, model_formula: str):
        self.model_formula = model_formula
        self.formula_list, self.input_node = self._parse_formula(model_formula)

        # Build node_specs: single parse, stored as NodeSpec objects
        self.node_specs: dict[str, DAG.NodeSpec] = {}
        for formula in self.formula_list:
            if "~" in formula:
                lhs, rhs = formula.split("~")
                self.node_specs[lhs] = DAG.NodeSpec(
                    name=lhs, rhs=rhs, node_type="exogenous", predecessors=[],
                )
            elif "<-" in formula:
                lhs, rhs = formula.split("<-")
                if rhs.startswith("n("):
                    predecessors = rhs[2:-1].split(",")
                    self.node_specs[lhs] = DAG.NodeSpec(
                        name=lhs, rhs=rhs, node_type="mlp", predecessors=predecessors,
                    )
                else:
                    terms = rhs.split("+")
                    parsed_terms = [self._parse_linear_term(t) for t in terms]
                    self.node_specs[lhs] = DAG.NodeSpec(
                        name=lhs,
                        rhs=rhs,
                        node_type="linear",
                        predecessors=[name for _, name in parsed_terms],
                        coefficients=[coeff for coeff, _ in parsed_terms],
                    )

        # Derive graph from node_specs for topological sorting
        self.graph: dict[str, list[str]] = {
            name: ns.predecessors for name, ns in self.node_specs.items()
        }

        try:
            self.topological_order: list[str] = list(
                TopologicalSorter(self.graph).static_order()
            )
        except CycleError as e:
            raise ValueError(
                f"Formula contains a cycle. Please check your formula: {e.args[1]}"
            ) from None

    def static_order(self):
        return iter(self.topological_order)

    # ── Linear term parsing helper fx ───────────────────────────────────────

    def _parse_linear_term(self, term: str) -> tuple[float | None, str]:
        """Parse a single linear RHS term into (coefficient, predecessor_name).

        coefficient is None  -> weight is learned (no prefix).
        coefficient is float -> weight is fixed at that value.

        Examples:
            'x'      -> (None,  'x')
            '1*x'    -> (1.0,   'x')
            '-1*x'   -> (-1.0,  'x')
            '0.5*x'  -> (0.5,   'x')
            '-0.5*x' -> (-0.5,  'x')
        """
        match = re.match(r'^(-?[0-9]*\.?[0-9]+)\*(.+)$', term)
        if match:
            return float(match.group(1)), match.group(2)
        return None, term

    # ── Formula parsing ───────────────────────────────────────────────────────

    def _parse_formula(self, model_formula: str):
        formula_list = model_formula.replace(" ", "").split(";")

        expanded_list = []
        input_node = set()
        for formula in formula_list:
            vars_, formulas = self._expand_lhs(formula)
            input_node.update(vars_)
            expanded_list.extend(formulas)

        formula_list = []
        for formula in expanded_list:
            formula_list.extend(self._expand_rhs(formula))

        return formula_list, input_node

    def _expand_lhs(self, formula: str):
        if "~" in formula:
            sep = "~"
        elif "<-" in formula:
            sep = "<-"
        else:
            raise ValueError(f"Formula must contain '~' or '<-': {formula!r}")

        lhs, rhs = formula.replace(" ", "").split(sep)
        variables = lhs.split(",")
        return variables, [var + sep + rhs for var in variables]

    def _split_rhs(self, rhs: str, sep: str):
        """Split by sep but ignore separators inside parentheses."""
        rhs = rhs.replace(" ", "")
        terms, current, depth = [], "", 0
        for ch in rhs:
            if ch == sep and depth == 0:
                terms.append(current)
                current = ""
            else:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                current += ch
        if current:
            terms.append(current)
        return terms

    def _expand_rhs(self, formula: str):
        formula = formula.replace(" ", "")

        if "~" in formula:
            return [formula]
        if formula.count("<-") != 1:
            raise ValueError(f"Formula must contain exactly one '<-': {formula!r}")

        lhs, rhs = formula.split("<-")

        if "n(" not in rhs:
            return [formula]

        plus_terms = self._split_rhs(rhs, "+")
        if len(plus_terms) == 1 and rhs.startswith("n(") and rhs.count("n(") == 1:
            return [formula]  # single flat nonlinear term, nothing to expand

        expanded = []
        if len(plus_terms) == 1 and rhs.startswith("n("):
            # Single nested nonlinear term
            new_formula = lhs + "<-n("
            comma_terms = self._split_rhs(rhs[2:-1], ",")
            for i, term in enumerate(comma_terms):
                if term.startswith("n(") or "+" in term:
                    new_rv = lhs + ":" + str(i)
                    expanded.extend(self._expand_rhs(new_rv + "<-" + term))
                    new_formula += new_rv
                else:
                    new_formula += term
                if i < len(comma_terms) - 1:
                    new_formula += ","
            new_formula += ")"
        else:
            # Mix of linear and nonlinear plus-terms
            new_formula = lhs + "<-"
            for i, term in enumerate(plus_terms):
                if term.startswith("n("):
                    new_rv = lhs + ":" + str(i)
                    expanded.extend(self._expand_rhs(new_rv + "<-" + term))
                    new_formula += new_rv
                else:
                    new_formula += term
                if i < len(plus_terms) - 1:
                    new_formula += "+"

        expanded.append(new_formula)
        return expanded


# ══════════════════════════════════════════════════════════════════════════════
# SEM2 - DAG + nn.Module
# ══════════════════════════════════════════════════════════════════════════════

class SEM2(DAG, nn.Module):
    """Structural Equation Model — a DAG whose nodes are PyTorch modules.

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

    def _build_and_initialize_nodes(self, node_cfgs: dict, device: str) -> dict:
        """Build all SEM2Node objects from DAG.node_specs in topological order.

        Because we iterate in topological order, every node's predecessors are
        guaranteed to already exist in name_to_node by the time we reach it.
        This means input_dim can be computed immediately and each node is fully
        constructed in a single pass.

        Pops configs from node_cfgs as nodes are built — any remaining keys
        after this call indicate unused / misspelled configs.
        """
        name_to_node = {}

        for node_name in self.topological_order:
            ns = self.node_specs[node_name]
            cfg = node_cfgs.pop(node_name, None)
            predecessor_nodes = [name_to_node[p] for p in ns.predecessors]

            if ns.node_type == "exogenous":
                out_dim = cfg if isinstance(cfg, int) else 1
                node = NodeExogenous(node_name, ns.rhs, out_dim=out_dim, device=device)

            elif ns.node_type == "mlp":
                # Good to note: rhs guaranteed by _expand_rhs to be flat n(a,b,...) since no nesting
                node = NodeMLP(node_name, ns.rhs, ns.predecessors, cfg, predecessor_nodes, device=device)

            elif ns.node_type == "linear":
                node = NodeLinear(
                    node_name, ns.rhs, ns.predecessors, ns.coefficients, cfg, predecessor_nodes, device=device
                )

            name_to_node[node_name] = node

        return name_to_node

    def forward(
        self,
        batch_size: int = 1,
        set_nodes: dict[str, torch.Tensor] = None,
        return_dict: bool = False,
    ):
        """Run a forward pass through the SEM.

        Args:
            batch_size: Number of samples to draw.
            set_nodes: Optional dict of node name -> tensor to pin specific node
                       values (e.g. for interventions). Shape: [batch_size, out_dim].
            return_dict: If True, return {node_name: tensor}. Otherwise return a
                         single concatenated tensor of shape [batch_size, total_out_dim].
        """
        set_nodes = set_nodes if set_nodes is not None else {}
        assert set(set_nodes.keys()).issubset(self.layer.keys()), \
            f"set_nodes contains unknown nodes: {set(set_nodes.keys()) - set(self.layer.keys())}"

        for node_name in self.topological_order:
            v = self.layer[node_name]
            if node_name in set_nodes:
                v.set_value(set_nodes[node_name])
            else:
                pred_values = [self.layer[p].get_value() for p in v.predecessors]
                v(pred_values, batch_size)

        if return_dict:
            return {name: self.layer[name].get_value() for name in self.topological_order}
        return torch.cat([self.layer[name].get_value() for name in self.topological_order], dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# TO be moved to integration tests later I think
def main():
    from .nn_models import MLPConfig

    # ── Example 1: Linear SEM ─────────────────────────────────────────────────

    # Fixed coefficients on x1 and x2, learned weight on noise.
    # Example of how to edit Public Linear Config if needed for the experiment: use_bias=False forces the relationship through the origin 
    # Otherwise usually dont need touch LinearNodeConfig at all, dont need to specify and just use defaults it should work ok.
    print("=== Linear SEM ===")
    linear_model = SEM2(
        "x1, x2, noise ~ Normal(0,1); y <- 0.5*x1 + -1*x2 + noise",
        node_cfgs={"y": LinearNodeConfig(use_bias=False)},
    )
    print(linear_model(batch_size=3, return_dict=True))

    # ── Example 2: Pure MLP SEM ───────────────────────────────────────────────
    
    # All inputs enter a nonlinear MLP, output is y.
    print("\n=== MLP SEM ===")
    mlp_model = SEM2(
        "x1, x2, noise ~ Normal(0,1); y <- n(x1, x2, noise)",
        node_cfgs={"y": MLPConfig(hidden_dims=[16, 16], output_dim=1)},
    )
    print(mlp_model(batch_size=3, return_dict=True))

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
    print(mixed_model(batch_size=3, return_dict=True))


if __name__ == "__main__":
    main()
