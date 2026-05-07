import re
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
from torch import distributions as dist
import torch.nn.functional as F
from dataclasses import dataclass

from .nn_models import MLP, MLPConfig


@dataclass
class LinearConfig:
    """ Configuration for linear layers. """
    use_bias: bool = True           # If to use bias in the linear layer.
    input_dim: int = None              # Input dimension of the linear layer. Defaults to number of predecessors.
    output_dim: int = 1             # Output dimension of the linear layer.
    init_weight_val: torch.Tensor = None    # Initial weights for the linear layer. If not given, weights are initialized randomly.
    init_bias_val: torch.Tensor = None      # Initial bias for the linear layer. If not given, bias is initialized randomly.



class SEM2Node(ABC, nn.Module):
    """ Base class for nodes. All nodes should inherit from this class.
        All nodes are initialised with 2 step proces:
            1) Graph is initialised
            2) Node attributes are set once the graph structure is known.
    """
    # Node attributes
    device: str
    name: str
    rhs: str
    value: torch.Tensor     # shape = [batch_size, 1]
    out_dim: int            # Output dimension of node.

    # TopologicalSorter attributes:
    npredecessors: int
    predecessors: list[str]

    def __init__(self, lhs, rhs, predecessors, device="cpu"):
        super().__init__()

        self.name = lhs
        self.rhs = rhs     # remove spaces
        self.predecessors = predecessors
        self.npredecessors = len(self.predecessors)
        # NOTE: npredecessors will be changed in topological ordering and should not be used for other purposes.
        self.device = device

        self.formula = lhs + "=" + rhs

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

    def __init__(self, node_name, rhs, lin_cfg: LinearConfig|None, device="cpu"):
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
        x_collated = self._collate_fn(x)
        return F.linear(x_collated, full_weight, bias)


class NodeMLP(SEM2Node):
    """ MLP node. Use default config if not given"""
    dist: torch.distributions.Distribution
    mlp: MLP

    def __init__(self, node_name, rhs, cfg: MLPConfig|None, device="cpu"):
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
        else:
            raise ValueError(f"Invalid MLP configuration: {self.cfg}. Must be MLPConfig or MLP instance.")

    def _forward_inner(self, x: list[torch.Tensor], batch_size=1):
        """ x: List[Tensor], shape = [n_predecessors][bs, in_dim_n] for parent n
            return.shape = [bs, out_dim]
         """
        x_collated = self._collate_fn(x)
        return self.mlp(x_collated)
