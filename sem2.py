import re
from graphlib import TopologicalSorter, _NodeInfo
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
from torch import distributions as dist
import torch.nn.functional as F
from nn_models import MLP, MLPConfig


class SEM2Node(ABC, _NodeInfo, nn.Module):
    """ Base class for nodes. All nodes should inherit from this class.
        All nodes are initialised with 2 step proces:
            1) Graph is initialised
            2) Node attributes are set once the graph structure is known.
    """
    # Node attributes
    device: str
    rhs: str
    value: torch.Tensor     # shape = [batch_size, 1]

    # TopologicalSorter attributes:
    npredecessors: int
    predecessors: list[str]

    def __init__(self, lhs, rhs, predecessors, device="cpu"):
        _NodeInfo.__init__(self, lhs)
        nn.Module.__init__(self)

        self.rhs = rhs     # remove spaces
        self.predecessors = predecessors
        self.npredecessors = len(self.predecessors)
        self.device = device

    def __repr__(self):
        return self.formula

    def reset_value(self):
        self.value = None

    def set_value(self, value):
        """ Set the value of the node. """
        self.value = value

    def get_value(self):
        return self.value

    def _collate_fn(self, x: list[torch.Tensor]):
        """ Collate predeccesor values into a single tensor.
            x: List[Tensor], shape = [n_predecessors][bs]

            return: Tensor, shape = [bs, input_dim]
            TODO: Different sort of collate. Currently, we just concatenate the tensors.
        """
        return torch.cat(x, dim=-1)

    def forward(self, x: list[torch.Tensor], batch_size=1):
        self.value = self._forward_inner(x, batch_size)
        return self.value

    @abstractmethod
    def _forward_inner(self, x: list[torch.Tensor], batch_size=1):
        """ Function that sets node value. """
        pass


class NodeDistribution(SEM2Node):
    """ Distribution node. Samples from random distribution, no parent nodes."""
    dist: torch.distributions.Distribution

    def __init__(self, node_name, rhs, device="cpu"):
        predecessors = []
        super().__init__(node_name, rhs, predecessors, device=device)

        """ Initialise the distribution. Get parameters by splitting the rhs string."""
        match = re.match(r'^([A-Za-z0-9]+)\((.*)\)$', self.rhs)     # Split apart string name from bracket parameter
        dist_name = match.group(1)  # e.g., 'Beta'
        dist_class = getattr(dist, dist_name)
        params_str = match.group(2)  # e.g., '0.5,0.5'
        params = [torch.tensor(float(p), device=self.device, dtype=torch.float32) for p in params_str.split(',')]
        self.dist = dist_class(*params)

    def _forward_inner(self, x, batch_size=1):
        # Sample from the distribution (parents not used)
        return self.dist.sample([batch_size, 1])


class NodeLinear(SEM2Node):
    is_random: bool = False
    is_linear: bool = True
    _fix_weights: torch.Tensor    # shape = [n_predecessors]
    _weight: torch.Tensor

    def __init__(self, node_name, rhs, device="cpu"):
        """ Linear layer.
            In order to fix some weights, some of the weights are masked off.
            Use effective weight matrix: A_eff = A * (1 - mask) + 1 * mask
        """
        rhs_vars = rhs.split("+")
        predecessors = [p.lstrip("1*") for p in rhs_vars]
        super().__init__(node_name, rhs, predecessors)

        # Fix weights by masking off some weights
        self.fix_weights = torch.tensor([p.startswith("1*") for p in rhs_vars], dtype=bool, device=device)
        self._ones = torch.ones(1, len(predecessors), device=device)

        self._weight = torch.nn.Parameter(torch.empty(1, len(predecessors), device=device))
        self._bias = torch.nn.Parameter(torch.empty(1, 1, device=device))

        # Match default pytorch init for nn.Linear layer
        nn.init.kaiming_uniform_(self._weight, a=np.sqrt(5))
        fan_in = nn.init._calculate_fan_in_and_fan_out(self._weight)[0]
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self._bias, -bound, bound)


    def weight_matrix(self):
        full_weight = self._weight * (~self.fix_weights) + self._ones * self.fix_weights
        return full_weight, self._bias

    def _forward_inner(self, x: list[torch.Tensor], batch_size=1):
        """ x: List[Tensor], shape = [n_predecessors][bs, 1] """
        full_weight, bias = self.weight_matrix()
        x = self._collate_fn(x)
        return F.linear(x, full_weight, bias)


class NodeNN(SEM2Node):
    """ MLP node. Use default config if not given"""
    dist: torch.distributions.Distribution

    def __init__(self, node_name, rhs, cfg: MLPConfig, device="cpu"):
        predecessors = rhs[2:-1].split(",")
        super().__init__(node_name, rhs, predecessors)

        if cfg is None:
            cfg = MLPConfig()
        cfg.input_dim = len(self.predecessors)
        self.mlp = MLP(cfg, device=device)


    def _forward_inner(self, x: list[torch.Tensor], batch_size=1):
        """ x: List[Tensor], shape = [n_predecessors][bs, 1] """
        x = self._collate_fn(x)
        return self.mlp(x)


class SEM2(TopologicalSorter, nn.Module):
    def __init__(self, model_formula:str, mlp_cfgs:dict=None, device="cpu"):
        """ model_formula: a string with the formulas for the model. The formulas should be separated by semicolons.
            model_components: Dict of node str to nn.Module. If a node is given in the model_components, it will be used, otherwise default linear layer is used.
                                Can be used to specify MLPs for non-linear models.
        """

        # initialize as a neural network and as a DAG
        nn.Module.__init__(self)
        TopologicalSorter.__init__(self, graph = None)
        self.device = device

        mlp_cfgs = mlp_cfgs if mlp_cfgs is not None else {}

        # Initialize node info by formula
        self.model_formula = model_formula
        self.formula_list, self.input_node = self._parse_formula(model_formula)

        # Initialize all nodes (which may contain new latent variables)
        rvs = []
        for formula in self.formula_list:
            if "~" in formula:
                lhs, rhs = formula.split("~")
                rvs.append(NodeDistribution(lhs, rhs, device=device))
            elif "<-" in formula:
                lhs, rhs = formula.split("<-")
                if rhs.startswith("n("):        # For NN node, use MLP with config if given, else default.
                    cfg = mlp_cfgs.pop(lhs, None)
                    rvs.append(NodeNN(lhs, rhs, cfg, device=device))
                else:
                    rvs.append(NodeLinear(lhs, rhs, device=device))

        assert len(mlp_cfgs) == 0, f"The following nodes were given but not set: {mlp_cfgs.keys()}"
        self._node2info = {rv.node: rv for rv in rvs}               # This is used by TopologicalSorter as well.

        # Find successors
        for node, v in self._node2info.items():
            for p in v.predecessors:
                self._node2info[p].successors.append(node)

        # Register modules to the model buffers / parameters
        self.layer = nn.ModuleDict(self._node2info)
        self.layer.to(device)

        self._topological_order = list(super(SEM2, self).static_order())

    def forward(self, batch_size=1, set_nodes:dict[str, torch.Tensor] = None):
        """ If set nodes is given, replace the values of the nodes in the model with the given values.
                set_nodes.keys(): the names of the nodes to be set
                set_nodes.values(): Shape = [BS, 1]. Values of the nodes to be set
        """
        set_nodes = set_nodes if set_nodes is not None else {}
        for node in self._topological_order:
            v = self._node2info[node]

            if node in set_nodes.keys():
                v.set_value(set_nodes.pop(node))
            else:
                pred_values = [self._node2info[p].get_value() for p in v.predecessors]
                v(pred_values, batch_size)

        assert len(set_nodes) == 0, f"The following nodes are not in the model: {set_nodes.keys()}"
        return torch.cat([self._node2info[node].get_value() for node in self._topological_order], dim=-1)

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
    formula1 = "e1, e2, e3, e4, e5 ~ Normal(0,1);"
    formula2 = "X1 <- e1; X2 <- e2; X3 <- 1 * X1 + X2 + e3; X4 <- n(X1, X3) + n(X2) + e4; X5 <- n(X1 + X2, n(X1 + X3, X4), e5)"
    model = SEM2(formula1 + formula2, device="cuda")
    model.formula_list

    # [model._input_node2index]
    print(model(3).detach())

if __name__ == "__main__":
    main()
