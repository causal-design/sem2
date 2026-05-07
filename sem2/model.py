from graphlib import TopologicalSorter
import torch
from torch import nn

from .nodes import NodeDistribution, NodeLinear, NodeMLP


class SEM2(TopologicalSorter, nn.Module):
    def __init__(self, model_formula:str, node_cfgs:dict=None, device="cpu"):
        """ model_formula: a string with the formulas for the model. The formulas should be separated by semicolons.
            node_cfgs: Dict of node str to config for nodes.
                If node is NN, the config should be MLPConfig or MLP instance (for already initialised MLP).
                If node is linear, the config should be LinearConfig.
                if node is distribution, the config should be an int specifying the output dimension of the distribution.
        """

        # initialize as a neural network and as a DAG
        nn.Module.__init__(self)
        TopologicalSorter.__init__(self, graph = None)
        self.device = device

        node_cfgs = node_cfgs if node_cfgs is not None else {}

        # Initialize node info by formula
        self.model_formula = model_formula
        self.formula_list, self.input_node = self._parse_formula(model_formula)

        # Initialize all nodes (which may contain new latent variables)
        rvs = []
        for formula in self.formula_list:
            if "~" in formula:
                lhs, rhs = formula.split("~")
                out_dim = node_cfgs.pop(lhs, 1)
                rvs.append(NodeDistribution(lhs, rhs, out_dim=out_dim, device=device))
            elif "<-" in formula:
                lhs, rhs = formula.split("<-")
                if rhs.startswith("n("):        # For NN node, use MLP with config if given, else default.
                    cfg = node_cfgs.pop(lhs, None)
                    rvs.append(NodeMLP(lhs, rhs, cfg, device=device))
                else:
                    lin_cfg = node_cfgs.pop(lhs, None)
                    rvs.append(NodeLinear(lhs, rhs, lin_cfg, device=device))

        assert len(node_cfgs) == 0, f"The following nodes were given but not set: {node_cfgs.keys()}"
        self._node2info = {rv.node: rv for rv in rvs}               # This is used by TopologicalSorter as well.

        # Find successors
        for node, v in self._node2info.items():
            v.successors = []
        for node, v in self._node2info.items():
            for p in v.predecessors:
                self._node2info[p].successors.append(node)

        self._topological_order = list(super(SEM2, self).static_order())

        # Initialize models inside nodes
        for node_name in self._topological_order:
            node = self._node2info[node_name]
            predecessor_nodes = [self._node2info[p] for p in node.predecessors]
            node.initialize_model(predecessor_nodes)

        # Register modules to the model buffers / parameters
        self.layer = nn.ModuleDict(self._node2info)
        self.layer.to(device)

    def forward(self, batch_size=1, set_nodes:dict[str, torch.Tensor] = None, return_dict:bool = False):
        """ If set nodes is given, replace the values of the nodes in the model with the given values.
                set_nodes.keys(): the names of the nodes to be set
                set_nodes.values(): Shape = [BS, 1]. Values of the nodes to be set
            return_dict: If True, return a dictionary of node names to their values. Otherwise, return tensor.
                shape = [BS, output_dim], where output_dim is the output of each node.
        """

        set_nodes = set_nodes if set_nodes is not None else {}
        # Check that all set_nodes are valid and in the model:
        assert set(set_nodes.keys()).issubset(self.layer.keys()), f"Invalid nodes given: {set_nodes.keys()}"

        for node in self._topological_order:
            v = self.layer[node]

            if node in set_nodes.keys():
                v.set_value(set_nodes[node])
            else:
                pred_values = [self._node2info[p].get_value() for p in v.predecessors]
                v(pred_values, batch_size)

        if return_dict:
            return {node: self.layer[node].get_value() for node in self._topological_order}
        else:
            return torch.cat([self.layer[node].get_value() for node in self._topological_order], dim=-1)

    def _parse_formula(self, model_formula):
        "x1 <- ind60 + ex1; x2 <- ind60 + ex2; x3 <- ind60 + ex3; ind60, ex1, ex2 , ex3 ~ Normal(0,1)"
        formula_list = model_formula.replace(" ", "").split(";")

        # Expand LHS of formula
        expanded_list = []
        input_node = set() # Record all variables in the input
        for formula in formula_list:
            new_formula = self._expand_lhs(formula)
            input_node.update(new_formula[0])
            expanded_list.extend(new_formula[1])
        formula_list = expanded_list

        # Expand RHS of formula
        expanded_list = []
        for formula in formula_list:
            expanded_list.extend(self._expand_rhs(formula))
        formula_list = expanded_list

        return formula_list, input_node

    def _expand_lhs(self, formula: str):
        # Determine the separator (could be ~ or <-)
        if "~" in formula:
            sep = "~"
        elif "<-" in formula:
            sep = "<-"
        else:
            raise ValueError("Formula must contain ~ or <-")

        lhs, rhs = formula.replace(" ", "").split(sep)
        variables = lhs.split(",") # split terms on LHS by comma

        return [variables, [var + sep + rhs for var in variables]]

    def _split_rhs(self, rhs: str, sep: str):
        """Split by sep ("+" or ",") but preserve content inside parentheses."""
        rhs = rhs.replace(" ", "")
        terms = []
        current_term = ""
        paren_count = 0
        for char in rhs:
            if char == sep and paren_count == 0:
                # Add a new term
                terms.append(current_term)
                current_term = ""
                continue
            elif char == "(":
                paren_count += 1
            elif char == ")":
                paren_count -= 1
            current_term += char
        if current_term:
            # Add the last term
            terms.append(current_term)
        return terms

    def _expand_rhs(self, formula: str):

        formula = formula.replace(" ", "")

        if "~" in formula:
            # RHS should be a distribution, no need to expand
            return [formula]
        elif formula.count("<-") != 1:
            raise ValueError("Formula must contain exactly one ~ or <-")

        lhs, rhs = formula.split("<-")

        if "n(" not in rhs:
            # formula is linear, no need to expand
            return [formula]

        plus_terms = self._split_rhs(rhs, "+")
        if len(plus_terms) == 1 and rhs.startswith("n(") and rhs.count("n(") == 1 and "+" not in rhs:
            # formula is a single, non-nested, nonlinear term
            return [formula]

        expanded_list = []
        if len(plus_terms) == 1 and rhs.startswith("n("):
            # a single, nested, nonlinear term
            new_formula = lhs + "<-n("
            comma_terms = self._split_rhs(rhs[2:-1], ",")
            for i, term in enumerate(comma_terms):
                if term.startswith("n(") or "+" in term:
                    # a nested term
                    new_rv_name = lhs + ":" + str(i)
                    expanded_list.extend(self._expand_rhs(new_rv_name + "<-" + term)) # nested expansion
                    new_formula = new_formula + new_rv_name
                else:
                    # a non-nested term
                    new_formula = new_formula + term
                if i < len(comma_terms) - 1:
                    new_formula = new_formula + ","
            new_formula = new_formula + ")"
        else:
            # a number of linear terms
            new_formula = lhs + "<-"
            for i, term in enumerate(plus_terms):
                if term.startswith("n("):
                    # a nested term
                    new_rv_name = lhs + ":" + str(i)
                    expanded_list.extend(self._expand_rhs(new_rv_name + "<-" + term)) # nested expansion
                    new_formula = new_formula + new_rv_name
                else:
                    # a non-nested term
                    new_formula = new_formula + term

                if i < len(plus_terms) - 1:
                    new_formula = new_formula + "+"

        # add the transformed formula
        expanded_list.append(new_formula)

        return expanded_list

    def _get_nodeinfo(self, node: str):
        """Override original private method, raise an error if node does not exist
        """
        return self._node2info[node]

    def add(self, node, *predecessors):
        """Override original methods, don't do anything
        """
        return None

    def static_order(self):
        """Override static_order to return cached topological order"""
        return iter(self._topological_order)

def main():
    # formula1 = "e1, e2, e3, e4, e5 ~ Normal(0,1);"
    # formula2 = "X1 <- e1; X2 <- e2; X3 <- 1 * X1 + X2 + e3; X4 <- n(X1, X3) + n(X2) + e4; X5 <- n(X1 + X2, n(X1 + X3, X4), e5)"
    # model = SEM2(formula1 + formula2, device="cpu")
    # model.formula_list
    #
    # # [model._input_node2index]
    # set_nodes = {"e1": torch.tensor([[1.], [2.], [3.]], device="cpu"),}
    # print(model(3))

    from .nodes import LinearConfig
    model_formula = "x1 <- ind60 + ex1; x2 <- ind60 + ex2; x3 <- ind60 + ex3; ind60, ex1, ex2 , ex3 ~ Normal(0,1); x4 <- x1"
    node_cfgs = {"x1": LinearConfig(output_dim=2, use_bias=False),
        "x4": LinearConfig(output_dim=1, init_weight_val=torch.tensor([[1.0, 2.]], dtype=torch.float32), use_bias=False)}
    model = SEM2(model_formula, node_cfgs=node_cfgs)
    sample1 = model(2, return_dict=True)
    print(sample1)

if __name__ == "__main__":
    main()
