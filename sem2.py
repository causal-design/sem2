import torch
from torch import nn
import scipy.stats as stats
from graphlib import TopologicalSorter, _NodeInfo

class rv_sem2(_NodeInfo):
    def __init__(self, formula: str):

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
            self.dist = eval("stats."+rhs+"()") # rhs is the distribution (must be a class in scipy.stats)
        else:
            raise ValueError("Formula does not contain <- or ~.")

        self.npredecessors = len(self.predecessors) # NOTE: this will be changed when obtaining topological order
        self.formula = formula

    def __repr__(self):
        return self.formula


class sem2(TopologicalSorter, nn.Module):
    def __init__(self, model_formula=None, model_components:dict=None, device="cpu"):
        self.device = device

        nn.Module.__init__(self)
        TopologicalSorter.__init__(self, graph = None)

        formula_list = model_formula.replace(" ", "").split(";")

        # Initialize node info by formula
        self._node2info = {rv_sem2(formula).node: rv_sem2(formula) for formula in formula_list}

        for node, v in self._node2info.items():
            for p in v.predecessors:
                self._node2info[p].successors.append(node)

        self.layer = nn.ModuleDict()
        for node, v in self._node2info.items():
            if v.is_random is True:
                self.layer[node] = None
            else:
                self.layer[node] = nn.Linear(len(v.predecessors), 1)
        self.layer.to(device)

        self._topological_order = list(super(sem2, self).static_order())
        self._node2index = {node: index for index, node in enumerate(self._topological_order)}

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

    def forward(self, batch_size = 1):
        n_nodes = len(self._node2info)
        x = torch.zeros(batch_size, n_nodes, device=self.device)
        for node in self._topological_order:
            v = self._node2info[node]
            index = self._node2index[node]
            if v.is_random is True:
                 x[:, index] = torch.tensor(v.dist.rvs(batch_size), device=self.device)
            else:
                pred_indices = [self._node2index[p] for p in v.predecessors]
                x[:, index] = self.layer[node](x[:, pred_indices]).squeeze(-1)

        return x