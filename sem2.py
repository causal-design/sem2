import re
from graphlib import TopologicalSorter, _NodeInfo
import numpy as np
import torch
from torch import nn
from torch import distributions as dist
from nn_models import MLP

class rv_sem2(_NodeInfo):
    def __init__(self, formula: str, device="cpu"):

        formula = formula.replace(" ", "") # remove spaces
        self.formula = formula

        if "<-" in formula:
            # this is a non-random node
            lhs, rhs = formula.split("<-")
            # lhs is the variable name, rhs contains all parents of the node
            super().__init__(lhs)

            if rhs.startswith("n("):
                # this is a nonlinear node
                self.predecessors = rhs[2:-1].split(",")
                self.is_linear = False
                self.fix_weights = None
            else:
                # this is a linear node
                predecessors = rhs.split("+")
                self.fix_weights = [p.startswith("1*") for p in predecessors]
                self.predecessors = [p.lstrip("1*") for p in predecessors]
                self.is_linear = True
            self.dist = None
            self.is_random = False
        elif "~" in formula:
            # this is a random node
            lhs, rhs = formula.split("~")
            super().__init__(lhs)
            self.predecessors = []
            self.is_random = True
            self.fix_weights = None
            self.is_linear = None

            # match the name of distribution and its parameters
            match = re.match(r'^([A-Za-z0-9]+)\((.*)\)$', rhs)
            dist_name = match.group(1)  # e.g., 'Beta'
            dist_class = getattr(dist, dist_name)
            params_str = match.group(2)  # e.g., '0.5,0.5'
            params = [torch.tensor(float(p), device = device, dtype=torch.float32) for p in params_str.split(',')]

            self.dist = dist_class(*params)
        else:
            raise ValueError("Formula does not contain <- or ~.")

        self.npredecessors = len(self.predecessors)
        # NOTE: npredecessors will be changed in topological ordering and should not be used for other purposes.

    def __repr__(self):
        return self.formula


class sem2(TopologicalSorter, nn.Module):

    def __init__(self, model_formula:str, model_components:dict=None, device="cpu"):
        """ model_formula: a string with the formulas for the model. The formulas should be separated by semicolons.
            model_components: Dict of node str to nn.Module. If a node is given in the model_components, it will be used, otherwise default linear layer is used.
                                Can be used to specify MLPs for non-linear models.
            Example:
        """

        # initialize as a neural network and as a DAG
        nn.Module.__init__(self)
        TopologicalSorter.__init__(self, graph = None)

        self.device = device
        model_components = model_components if model_components is not None else {}

        # Initialize node info by formula
        self.model_formula = model_formula
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

        # Save the expanded formulas
        self.formula_list = formula_list

        # Initialize all nodes (which may contain new latent variables)
        rvs = [rv_sem2(formula, device=device) for formula in formula_list]
        self._node2info = {rv.node: rv for rv in rvs}

        # Find successors
        for node, v in self._node2info.items():
            for p in v.predecessors:
                self._node2info[p].successors.append(node)

        # Construct the neural network
        self.layer = nn.ModuleDict()
        for node, v in self._node2info.items():
            if v.is_random:
                self.layer[node] = None
            elif node in model_components.keys():       # if the node is given in model_components, use it
                self.layer[node] = model_components.pop(node)
            elif v.is_linear:
                self.layer[node] = nn.Linear(len(v.predecessors), 1)
                for i in range(len(v.predecessors)):
                    if v.fix_weights[i]:
                        with torch.no_grad():
                            self.layer[node].weight[0, i] = 1.0
            else:
                self.layer[node] = MLP(len(v.predecessors), [128, 128, 128], 1)
        assert len(model_components) == 0, f"The following nodes were given but not set: {model_components.keys()}"
        self.layer.to(device)

        self._topological_order = list(super(sem2, self).static_order())
        self._node2index = {node: index for index, node in enumerate(self._topological_order)}
        self._input_node2index = {node: index for index, node in enumerate(self._topological_order) if node in input_node}

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

    def forward(self, batch_size=1, set_nodes:dict[str, torch.Tensor] = None):
        """ If set nodes is given, replace the values of the nodes in the model with the given values.
                set_nodes.keys(): the names of the nodes to be set
                set_nodes.values(): Shape = [BS, 1]. Values of the nodes to be set
        """
        set_nodes = set_nodes if set_nodes is not None else {}

        n_nodes = len(self._node2info)
        x = torch.zeros(batch_size, n_nodes, device=self.device)
        for node in self._topological_order:

            v = self._node2info[node]
            index = self._node2index[node]

            if node in set_nodes.keys():
                x[:, index] = set_nodes.pop(node)
            elif v.is_random is True:
                x[:, index] = v.dist.sample([batch_size])
            else:
                pred_indices = [self._node2index[p] for p in v.predecessors]
                x[:, index] = self.layer[node](x[:, pred_indices]).squeeze(-1)

        assert len(set_nodes) == 0, f"The following nodes are not in the model: {set_nodes.keys()}"
        return x[:, list(self._input_node2index.values())]

    def fix_weights(self):
        """Set weights of fixed coefficients to 1; call this after any update to the model."""
        for node, v in self._node2info.items():
            if not v.is_random and not node in model_components.keys() and v.is_linear:
                for i in range(len(v.predecessors)):
                    if v.fix_weights[i]:
                        with torch.no_grad():
                            self.layer[node].weight[0, i] = 1.0


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
