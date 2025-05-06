import torch
from torch import nn
from torch import distributions as dist

import scipy.stats as stats
from graphlib import TopologicalSorter, _NodeInfo
import numpy as np


class rv_distribution:
    """ Random distributions for nodes. Can use torch distributions or scipy distributions"""
    dist_map = {
        "norm": (dist.Normal, (0.0, 1.0)),
        "uniform": (dist.Uniform, (0.0, 1.0)),
        "bernoulli": (dist.Bernoulli, (0.5)),
        "exponential": (dist.Exponential, (1.0)),
        "gamma": (dist.Gamma, (1.0, 1.0)),
        "beta": (dist.Beta, (1.0, 1.0)),
    }

    def __init__(self, dist_name, dist_args=None, device="cpu"):
        self.device = device

        if dist_name in self.dist_map:
            torch_dist, args = self.dist_map[dist_name]
            if dist_args is not None:
                args = dist_args
            args = [torch.tensor(arg, device=self.device, dtype=torch.float32) for arg in args]
            self.dist = torch_dist(*args)
            self._using_torch = True
        else:
            # Fallback for custom distribution specifications
            self.dist = eval(f"stats.{dist_name}()")
            self._using_torch = False

    def sample(self, bs):
        if self._using_torch:
            return self.dist.sample([bs])
        else:
            # Handle custom distribution sampling
            return torch.tensor(self.dist.rvs(bs), device=self.device, dtype=torch.float32)
            # NOTE: With device = "mps", the last step can give an error later if not converted to float32


class rv_sem2(_NodeInfo):
    def __init__(self, formula: str, device="cpu"):
        if "<-" in formula:
            lhs, rhs = formula.replace(" ", "").split("<-")
            super().__init__(lhs) # lhs is the variable name
            self.predecessors = rhs.split("+") # rhs is the parents of the node
            self.dist = None
            self.is_random = False
        elif "~" in formula:
            lhs, rhs = formula.replace(" ", "").split("~")
            super().__init__(lhs) # lhs is the variable name
            self.predecessors = []
            self.is_random = True
            self.dist = rv_distribution(rhs, device=device)
        else:
            raise ValueError("Formula does not contain <- or ~.")

        self.npredecessors = len(self.predecessors) # NOTE: this will be changed when obtaining topological order
        self.formula = formula

    def __repr__(self):
        return self.formula


class sem2(TopologicalSorter, nn.Module):
    def __init__(self, model_formula:str, model_components:dict=None, device="cpu"):
        """ model_formula: a string with the formulas for the model. The formulas should be separated by semicolons.
            model_components: Dict of node str to nn.Module. If a node is given in the model_components, it will be used, otherwise default linear layer is used.
                                Can be used to specify MLPs for non-linear models.
        """
        nn.Module.__init__(self)
        TopologicalSorter.__init__(self, graph = None)

        self.device = device
        model_components = model_components if model_components is not None else {}

        # Initialize node info by formula
        formula_list = model_formula.replace(" ", "").split(";")
        rvs = [rv_sem2(formula, device=device) for formula in formula_list]
        self._node2info = {rv.node: rv for rv in rvs}

        for node, v in self._node2info.items():
            for p in v.predecessors:
                self._node2info[p].successors.append(node)

        self.layer = nn.ModuleDict()
        for node, v in self._node2info.items():
            if v.is_random is True:
                self.layer[node] = None
            elif node in model_components.keys():       # if the node is given in model_components, use it
                self.layer[node] = model_components.pop(node)
            else:
                self.layer[node] = nn.Linear(len(v.predecessors), 1)
        assert len(model_components) == 0, f"The following nodes were given but not set: {model_components.keys()}"
        self.layer.to(device)

        self._topological_order = list(super(sem2, self).static_order())
        self._node2index = {node: index for index, node in enumerate(self._topological_order)}


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
                x[:, index] = v.dist.sample(batch_size)
            else:
                pred_indices = [self._node2index[p] for p in v.predecessors]
                x[:, index] = self.layer[node](x[:, pred_indices]).squeeze(-1)

        assert len(set_nodes) == 0, f"The following nodes are not in the model: {set_nodes.keys()}"
        return x

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

