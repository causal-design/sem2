# Defer annotation evaluation so forward references (e.g. SEM2Node used in its
# own method signatures) resolve lazily on Python < 3.14.
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import torch
import warnings
from torch import nn
from torch.distributions import Distribution

from ..._compat import StrEnum
from .nn_models import _LinearConfig, _LinearModel, _MLP
from ...default_configs import LinearNodeConfig, MLPConfig

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

    def __init__(self, name: str, rhs: str, predecessor_nodes: list[SEM2Node], device="cpu"):
        super().__init__()
        self.name = name
        self.rhs = rhs
        self.predecessors = [n.name for n in predecessor_nodes]
        self.npredecessors = len(self.predecessors)
        self.device = device
        self.formula = name + "=" + rhs
        self.value = None

        # Number of input variables.
        self.input_dim = sum(p.out_dim for p in predecessor_nodes)

    def __repr__(self):
        return f'SEM2Node: {self.formula}'

    def set_value(self, value: torch.Tensor):
        self.value = value

    def get_value(self) -> torch.Tensor:
        return self.value

    def reset_value(self):
        self.value = None

    def _collate_fn(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate predecessor values along the last dimension."""
        return torch.cat(x, dim=-1)

    def forward(self, x: list[torch.Tensor], batch_size: int = 1, **kwargs):
        self.value = self._forward_inner(x, batch_size=batch_size, **kwargs)
        return self.value

    @abstractmethod
    def _forward_inner(self, x: list[torch.Tensor], batch_size: int = 1, **kwargs) -> torch.Tensor:
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
        super().__init__(name, rhs, predecessor_nodes=[], device=device)
        self.out_dim = out_dim
        self.dist = dist

    def _forward_inner(self, x: list[torch.Tensor], batch_size: int = 1) -> torch.Tensor:
        return self.dist.sample([batch_size, self.out_dim])


# ── Endogenous (abstract) ─────────────────────────────────────────────────────

class NodeEndogenous(SEM2Node, ABC):
    """Abstract base for endogenous nodes.
    An endogenous node is a deterministic function of its parent nodes.
    """

    def __init__(self, name: str, rhs: str, predecessor_nodes: list['SEM2Node'], device="cpu"):
        super().__init__(name, rhs, predecessor_nodes, device=device)


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
    An explicit intercept is also a buffer and takes precedence over cfg.use_bias.
    """

    def __init__(
        self,
        name: str,
        rhs: str,
        coefficients: list[float | None],
        cfg: LinearNodeConfig | None,
        predecessor_nodes: list[SEM2Node],
        device="cpu",
        intercept: float | None = None,
    ):
        super().__init__(name, rhs, predecessor_nodes, device=device)
        self.out_dim = cfg.output_dim if cfg is not None else 1
        self.use_bias = cfg.use_bias if cfg is not None else True
        self.register_buffer(
            "_fixed_intercept",
            torch.full((self.out_dim,), float(intercept), device=device) if intercept is not None else None,
        )

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

        self.fix_weights = nn.parameter.Buffer(torch.tensor(fix_mask, dtype=torch.bool, device=device).unsqueeze(0))
        self._fixed_vals = nn.parameter.Buffer(torch.tensor(fix_vals, dtype=torch.float32, device=device).unsqueeze(0))
        self._ones = nn.parameter.Buffer(torch.ones(1, input_dim, device=device))

        internal_cfg = _LinearConfig(
            input_dim=input_dim,
            output_dim=self.out_dim,
            use_bias=self.use_bias and intercept is None,
        )
        self.linear = _LinearModel(internal_cfg, device=device)

    def _effective_weight(self) -> torch.Tensor:
        """Learned weights where free, fixed values where pinned."""
        return self.linear.weight * (~self.fix_weights) + self._fixed_vals * self.fix_weights

    def effective_weight(self) -> torch.Tensor:
        """Public accessor for the node's effective weight matrix, shape
        [out_dim, input_dim], combining learned weights (where free) and fixed
        coefficients (where pinned). Columns follow predecessor order, with each
        predecessor occupying `out_dim` consecutive columns.
        """
        return self._effective_weight()

    def bias(self) -> torch.Tensor | None:
        """Effective fixed or learned intercept [out_dim], or None if absent."""
        return self._fixed_intercept if self._fixed_intercept is not None else self.linear.bias

    def _forward_inner(self, x: list[torch.Tensor], batch_size: int = 1) -> torch.Tensor:
        x_cat = self._collate_fn(x) if x else self.linear.weight.new_empty(batch_size, 0)
        return torch.nn.functional.linear(x_cat, self._effective_weight(), self.bias())


# ── Nonlinear (abstract) ──────────────────────────────────────────────────────

class NodeNonlinear(NodeEndogenous, ABC):
    """Abstract nonlinear endogenous node.
    Subclass this to plug in a different linking function (e.g. GP, spline).
    """

    def __init__(self, name: str, rhs: str, predecessor_nodes: list[SEM2Node], device="cpu"):
        super().__init__(name, rhs, predecessor_nodes, device=device)

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
        cfg: MLPConfig | None,
        predecessor_nodes: list[SEM2Node],
        device="cpu",
    ):
        super().__init__(name, rhs, predecessor_nodes, device=device)
        self.cfg = cfg if cfg is not None else MLPConfig()

        self.cfg.input_dim = self.input_dim
        self.mlp = _MLP(self.cfg, device=device)
        self.out_dim = self.cfg.output_dim

    def _forward_inner(self, x: list[torch.Tensor], **kwargs) -> torch.Tensor:
        """ Input shape:
                x: List[Tensor], shape = [n_predecessors][bs, output_dim]
            Return shape:
                return.shape = [bs, output_dim]
        """
        return self.mlp(self._collate_fn(x))


# ── Discrete Node ───────────────────────────────────────────────────────────────────

class DiscreteModes(StrEnum):
    GUMBEL = "gumbel"                   # Gumbel-Softmax, a soft approximation of a categorical distribution
    SOFTMAX = "softmax"                 # Return probabilities for each outcome
    SAMPLE_ONEHOT = "onehot"            # Return one-hot vectors sampled from the distribution
    SAMPLE = "sample"                   # Return a single sample from the distribution. Does not support training.


class NodeMLPDiscrete(SEM2Node):
    """ Discrete NN node. """
    id: int = 0
    out_dim = 1         # The discrete node outputs 1 value, with n_classes categories.

    def __init__(self, name: str, rhs: str, cfg: MLPConfig, predecessor_nodes: list[SEM2Node], mode=None, device="cpu"):
        super().__init__(name, rhs, predecessor_nodes, device=device)
        self.cfg = cfg if cfg is not None else MLPConfig()

        cfg.input_dim = self.input_dim
        if isinstance(cfg, MLPConfig):
            self.nn_module = _MLP(cfg, device=device)
        else:
            raise ValueError(f"Invalid MLP configuration: \n{cfg}\n. Must be MLPConfig object, or None if default config is needed.")

        self.n_classes = cfg.output_dim

    def _forward_inner(self, x: list[torch.Tensor], mode=DiscreteModes.SAMPLE, **kwargs) -> torch.Tensor:
        """ Input shape:
                x: List[Tensor], shape = [n_predecessors][bs, output_dim]
                mode: How to treat the discretization.
            Return shape depends on the mode,
                if DiscreteModes.SAMPLE:
                    return.shape = [bs, 1]
                if DiscreteModes.SOFTMAX:
                    return.shape = [bs, n_classes]
                if DiscreteModes.Gumbel:
                    return.shape = TBD
                if DiscreteModes.SAMPLE_ONEHOT:
                    return.shape = TBD
        """
        x = self._collate_fn(x)
        out = self.nn_module(x)
        out = self._discretize(out, mode)
        return out

    def _discretize(self, x, mode):
        """ Turn input logits into a representation of discrete values. """
        if mode == DiscreteModes.SOFTMAX:
            # Return probabilities for each outcome. Shape = [..., n_outcomes]
            return torch.softmax(x, dim=-1)
        elif mode == DiscreteModes.SAMPLE:
            # Draw from the observed distribution. Shape = [..., 1]
            if self.training:
                warnings.warn("Drawing sample for discrete nodes does not support backpropagation during training. \n"
                              " To train discrete nodes, use another sampling strategy, via a different SEM2 class. ")
            probs = torch.softmax(x, dim=-1)
            sample = torch.distributions.Categorical(probs=probs).sample()
            return sample.unsqueeze(-1)
        else:
            raise NotImplementedError(f"Discretisation mode {mode} not implemented yet.")
