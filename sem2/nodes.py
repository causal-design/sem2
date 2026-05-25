import re
from abc import ABC, abstractmethod
from typing import Optional
import torch
from torch import nn
from torch.distributions import Distribution

from .nn_models import _LinearConfig, _LinearModel, LinearNodeConfig, MLPConfig, _MLP


# ── RandomVariable ────────────────────────────────────────────────────────────

class RandomVariable(ABC):
    """Abstract base for any node that is conceptually a random variable.
    Carries an optional dist attribute — concrete subclasses that wrap a
    torch.distributions.Distribution should set it; purely deterministic
    endogenous nodes leave it as None.
    """
    dist: Optional[Distribution] = None


# ── Base ──────────────────────────────────────────────────────────────────────

class SEM2Node(RandomVariable, ABC, nn.Module):
    """Base class for all SEM nodes.
    Conceptually every node is a random variable:

    Each node is fully initialised in __init__ ===> predecessor nodes must be passed in at construction time 
    so input dimensions can be computed immediately.
    """

    name: str
    rhs: str
    value: torch.Tensor
    out_dim: int
    predecessors: list[str]

    def __init__(self, name: str, rhs: str, predecessors: list[str], device="cpu"):
        super().__init__()
        self.name = name
        self.rhs = rhs
        self.predecessors = predecessors
        self.npredecessors = len(predecessors)
        self.device = device
        self.formula = name + "=" + rhs
        self.value = None

    def __repr__(self):
        return self.formula

    def set_value(self, value: torch.Tensor):
        self.value = value

    def get_value(self) -> torch.Tensor:
        return self.value

    def reset_value(self):
        self.value = None

    def _collate_fn(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate predecessor values along the last dimension."""
        return torch.cat(x, dim=-1)

    def forward(self, x: list[torch.Tensor], batch_size: int = 1):
        self.value = self._forward_inner(x, batch_size)
        return self.value

    @abstractmethod
    def _forward_inner(self, x: list[torch.Tensor], batch_size: int = 1) -> torch.Tensor:
        pass


# ── Exogenous ─────────────────────────────────────────────────────────────────

class NodeExogenous(SEM2Node):
    """Exogenous node with a known distribution.
    An exogenous node has no parents. The distribution is instantiated upstream
    in SEM2._build_and_initialize_nodes and passed in here directly.

    Attributes:
        dist: the instantiated torch Distribution object.
    """

    def __init__(
        self,
        name: str,
        rhs: str,
        dist: Distribution,
        out_dim: int = 1,
        device="cpu",
    ):
        super().__init__(name, rhs, predecessors=[], device=device)
        self.out_dim = out_dim
        self.dist = dist

    def _forward_inner(self, x: list[torch.Tensor], batch_size: int = 1) -> torch.Tensor:
        return self.dist.sample([batch_size, self.out_dim])


# ── Endogenous (abstract) ─────────────────────────────────────────────────────

class NodeEndogenous(SEM2Node, ABC):
    """Abstract base for endogenous nodes.
    An endogenous node is a deterministic function of its parent nodes.
    """

    def __init__(self, name: str, rhs: str, predecessors: list[str], device="cpu"):
        super().__init__(name, rhs, predecessors, device=device)


# ── Linear ────────────────────────────────────────────────────────────────────

class NodeLinear(NodeEndogenous):
    """Linear endogenous node. Thin wrapper around _LinearModel.

    coefficients: one entry per predecessor, parsed upstream in DAG.
        - float  -> fixed at that value (e.g. 1.0, -1.0, 0.5)
        - None   -> learned weight

    Coefficients are expanded to match input_dim (accounting for predecessors
    with out_dim > 1) and split into a boolean mask and a fixed-value tensor.

    fix_weights, _fixed_vals, and _ones are registered as buffers so they move
    correctly with model.to(device) and are not counted as learnable parameters.
    """

    def __init__(
        self,
        name: str,
        rhs: str,
        predecessors: list[str],
        coefficients: list[float | None],
        cfg: LinearNodeConfig | None,
        predecessor_nodes: list[SEM2Node],
        device="cpu",
    ):
        super().__init__(name, rhs, predecessors, device=device)
        self.out_dim = cfg.output_dim if cfg is not None else 1
        self.use_bias = cfg.use_bias if cfg is not None else True

        input_dim = sum(p.out_dim for p in predecessor_nodes)

        # Expand coefficients from [n_predecessors] -> [input_dim],
        # repeating each entry for all of that predecessor's output dims.
        expanded = [
            coeff
            for coeff, p in zip(coefficients, predecessor_nodes)
            for _ in range(p.out_dim)
        ]

        # Split into mask and fixed values tensor, both shape [1, input_dim].
        # Registered as buffers so they move with model.to(device).
        fix_mask = [c is not None for c in expanded]
        fix_vals = [c if c is not None else 0.0 for c in expanded]

        self.register_buffer(
            "fix_weights",
            torch.tensor(fix_mask, dtype=torch.bool, device=device).unsqueeze(0),
        )
        self.register_buffer(
            "_fixed_vals",
            torch.tensor(fix_vals, dtype=torch.float32, device=device).unsqueeze(0),
        )
        self.register_buffer(
            "_ones",
            torch.ones(1, input_dim, device=device),
        )

        internal_cfg = _LinearConfig(
            input_dim=input_dim,
            output_dim=self.out_dim,
            use_bias=self.use_bias,
        )
        self.linear = _LinearModel(internal_cfg, device=device)

    def _effective_weight(self) -> torch.Tensor:
        """Learned weights where free, fixed values where pinned."""
        return self.linear.weight * (~self.fix_weights) + self._fixed_vals * self.fix_weights

    def _forward_inner(self, x: list[torch.Tensor], batch_size: int = 1) -> torch.Tensor:
        x_cat = self._collate_fn(x)
        return torch.nn.functional.linear(x_cat, self._effective_weight(), self.linear.bias)


# ── Nonlinear (abstract) ──────────────────────────────────────────────────────

class NodeNonlinear(NodeEndogenous, ABC):
    """Abstract nonlinear endogenous node.
    Subclass this to plug in a different linking function (e.g. GP, spline).
    """

    def __init__(self, name: str, rhs: str, predecessors: list[str], device="cpu"):
        super().__init__(name, rhs, predecessors, device=device)

    @abstractmethod
    def _forward_inner(self, x: list[torch.Tensor], batch_size: int = 1) -> torch.Tensor:
        pass


# ── NodeMLP ───────────────────────────────────────────────────────────────────

class NodeMLP(NodeNonlinear):
    """Default nonlinear node — MLP as the linking function.
    Predecessors are parsed upstream in DAG (rhs is flat n(a,b,...) post-expansion).
    """

    def __init__(
        self,
        name: str,
        rhs: str,
        predecessors: list[str],
        cfg: MLPConfig | None,
        predecessor_nodes: list[SEM2Node],
        device="cpu",
    ):
        super().__init__(name, rhs, predecessors, device=device)
        self.cfg = cfg if cfg is not None else MLPConfig()

        input_dim = sum(p.out_dim for p in predecessor_nodes)
        if self.cfg.input_dim is None:
            self.cfg.input_dim = input_dim
        self.mlp = _MLP(self.cfg, device=device)
        self.out_dim = self.cfg.output_dim

    def _forward_inner(self, x: list[torch.Tensor], batch_size: int = 1) -> torch.Tensor:
        return self.mlp(self._collate_fn(x))
