import re
from graphlib import TopologicalSorter, _NodeInfo
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
from torch import distributions as dist
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class LinearConfig:
    """ Configuration for linear layers. """
    use_bias: bool = True           # If to use bias in the linear layer.
    input_dim: int = None              # Input dimension of the linear layer. Defaults to number of predecessors.
    output_dim: int = 1             # Output dimension of the linear layer.
    init_weight_val: torch.Tensor = None    # Initial weights for the linear layer. If not given, weights are initialized randomly.
    init_bias_val: torch.Tensor = None      # Initial bias for the linear layer. If not given, bias is initialized randomly.


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
    out_dim: int            # Output dimension of node.

    # TopologicalSorter attributes:
    npredecessors: int
    predecessors: list[str]

    def __init__(self, lhs, rhs, predecessors, device="cpu"):
        _NodeInfo.__init__(self, lhs)
        nn.Module.__init__(self)

        self.rhs = rhs     # remove spaces
        self.predecessors = predecessors
        self.npredecessors = len(self.predecessors)
        # NOTE: npredecessors will be changed in topological ordering and should not be used for other purposes.
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
            x: List[Tensor], shape = [n_predecessors][bs, ..., input_dim]

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

    @abstractmethod
    def initialize_model(self, predecessor_nodes: list['SEM2Node']):
        """ Initialise the model parameters. """
        pass


class NodeDistribution(SEM2Node):
    """ Distribution node. Samples from random distribution, no parent nodes."""
    dist: torch.distributions.Distribution
    def __init__(self, node_name, rhs, out_dim: int=1, device="cpu"):
        predecessors = []
        super().__init__(node_name, rhs, predecessors, device=device)
        self.out_dim = out_dim

        """ Initialise the distribution. Get parameters by splitting the rhs string."""
        match = re.match(r'^([A-Za-z0-9]+)\((.*)\)$', self.rhs)     # Split apart string name from bracket parameter
        dist_name = match.group(1)  # e.g., 'Beta'
        dist_class = getattr(dist, dist_name)
        params_str = match.group(2)  # e.g., '0.5,0.5'
        params = [torch.tensor(float(p), device=self.device, dtype=torch.float32) for p in params_str.split(',')]
        self.dist = dist_class(*params)

    def _forward_inner(self, x, batch_size=1):
        # Sample from the distribution (parents not used)
        value = self.dist.sample([batch_size, self.out_dim])
        return value

    def initialize_model(self, predecessor_nodes: list['SEM2Node']):
        """ No-op for distribution nodes as they have no predecessors. """
        pass


class NodeLinear(SEM2Node):
    is_random: bool = False
    is_linear: bool = True
    _fix_weights: torch.Tensor    # shape = [n_predecessors]
    _weight: torch.Tensor

    def __init__(self, node_name, rhs, lin_cfg: LinearConfig, device="cpu"):
        """ Linear layer.
            In order to fix some weights, some of the weights are masked off.
            Use effective weight matrix: A_eff = A * (1 - mask) + 1 * mask.

            lin_cfg: dict with configuration for the linear layer. Options:
                use_bias: bool, whether to use bias in the linear layer. Default is True.
                output_dim: int, output dimension of the linear layer. Default is 1.
                init_weight_val: torch.Tensor, initial weights for the linear layer. If not given, weights are initialized randomly.
                init_bias_val: torch.Tensor, initial bias for the linear layer. If not given, bias is initialized randomly.

        """
        rhs_vars = rhs.split("+")
        predecessors = [p.lstrip("1*") for p in rhs_vars]
        super().__init__(node_name, rhs, predecessors, device=device)

        if lin_cfg is None:
            lin_cfg = LinearConfig()
        self.lin_cfg = lin_cfg
        self.out_dim = lin_cfg.output_dim

        # Fix weights by masking off some weights
        self.fix_weights = torch.tensor([p.startswith("1*") for p in rhs_vars], dtype=torch.bool, device=device)

    def initialize_model(self, predecessor_nodes: list['SEM2Node']):
        """ Initialize the linear layer's weights and biases. """
        if self.lin_cfg.input_dim is not None:
            input_dim = self.lin_cfg.input_dim
        else:
            input_dim = sum(p.out_dim for p in predecessor_nodes)

        self._ones = torch.ones(1, input_dim, device=self.device)

        # If weights are given, use them. Otherwise, initialize them with the default pytorch init for nn.Linear layer
        if self.lin_cfg.init_weight_val is not None:
            assert isinstance(self.lin_cfg.init_weight_val, torch.Tensor), "init_weight must be a torch.Tensor"
            assert self.lin_cfg.init_weight_val.shape == (self.out_dim, input_dim), f"init_weight must have shape ({self.out_dim}, {input_dim})"
            self._weight = torch.nn.Parameter(self.lin_cfg.init_weight_val.to(self.device))
        else:
            self._weight = torch.nn.Parameter(torch.empty(self.out_dim, input_dim, device=self.device))
            nn.init.kaiming_uniform_(self._weight, a=np.sqrt(5))

        # Initialize bias / use given bias if needed
        if self.lin_cfg.use_bias:
            if self.lin_cfg.init_bias_val is not None:
                self._bias = torch.nn.Parameter(self.lin_cfg.init_bias_val.to(self.device))
            else:
                self._bias = torch.nn.Parameter(torch.empty(self.out_dim, device=self.device))
                fan_in = nn.init._calculate_fan_in_and_fan_out(self._weight)[0]
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(self._bias, -bound, bound)
        else:
            self._bias = None
            assert self.lin_cfg.init_bias_val is None, "init_bias_val must be None if use_bias is False"

    def weight_matrix(self):
        full_weight = self._weight * (~self.fix_weights) + self._ones * self.fix_weights
        return full_weight, self._bias

    def _forward_inner(self, x: list[torch.Tensor], batch_size=1):
        """ x: List[Tensor], shape = [n_predecessors][bs, in_dim_n] for parent n.
            return.shape = [bs, out_dim]
        """

        full_weight, bias = self.weight_matrix()
        x = self._collate_fn(x)
        return F.linear(x, full_weight, bias)


class NodeNN(SEM2Node):
    """ MLP node. Use default config if not given"""
    dist: torch.distributions.Distribution
    mlp: MLP

    def __init__(self, node_name, rhs, cfg: MLPConfig, device="cpu"):
        predecessors = rhs[2:-1].split(",")
        super().__init__(node_name, rhs, predecessors)

        if cfg is None:
            cfg = MLPConfig()
        self.cfg = cfg
        self.device = device

    def initialize_model(self, predecessor_nodes: list['SEM2Node']):
        """ Initialize the MLP model. """
        input_dim = sum(p.out_dim for p in predecessor_nodes)
        if isinstance(self.cfg, MLPConfig):
            if self.cfg.input_dim is None:
                self.cfg.input_dim = input_dim
            self.mlp = MLP(self.cfg, device=self.device)
            self.out_dim = self.cfg.output_dim
        elif isinstance(self.cfg, MLP):
            assert self.mlp.cfg.input_dim == input_dim, "Given MLP input dimension is different from the expected input dimension based on predecessors."
            self.mlp = self.cfg
            self.out_dim = self.mlp.cfg.output_dim
        else:
            raise ValueError(f"Invalid MLP configuration: {self.cfg}. Must be MLPConfig or MLP instance.")

    def _forward_inner(self, x: list[torch.Tensor], batch_size=1):
        """ x: List[Tensor], shape = [n_predecessors][bs, in_dim_n] for parent n
            return.shape = [bs, out_dim]
         """
        x = self._collate_fn(x)
        return self.mlp(x)


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
                    rvs.append(NodeNN(lhs, rhs, cfg, device=device))
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

    def forward_single_node(self, node_name:str, inputs:list[torch.Tensor], batch_size=1):
        """ Forward a single node with given inputs.
            inputs: List of tensors, each tensor is the value of a predecessor node.
            batch_size: Batch size to use for the node.
        """
        if node_name not in self.layer:
            raise ValueError(f"Node {node_name} does not exist in the model.")
        node = self.layer[node_name]
        return node(inputs, batch_size)

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

    model_formula = "x1 <- ind60 + ex1; x2 <- ind60 + ex2; x3 <- ind60 + ex3; ind60, ex1, ex2 , ex3 ~ Normal(0,1); x4 <- x1"
    node_cfgs = {"x1": LinearConfig(output_dim=2, use_bias=False),
        "x4": LinearConfig(output_dim=1, init_weight_val=torch.tensor([[1.0, 2.]], dtype=torch.float32), use_bias=False)}
    model = SEM2(model_formula, node_cfgs=node_cfgs)
    sample1 = model(2, return_dict=True)
    print(sample1)

if __name__ == "__main__":
    main()
