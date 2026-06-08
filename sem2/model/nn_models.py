from typing import TYPE_CHECKING
from dataclasses import dataclass
from typing import Optional
from enum import StrEnum
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
if TYPE_CHECKING:
    from torch import Tensor

# LinearNodeConfig and MLPConfig are the public user-facing node configs.
# They live in default_configs and are re-exported here so internal node
# constructors can import from a single place.
from default_configs import LinearNodeConfig, MLPConfig


# ── Internal linear config + model ────────────────────────────────────────────

@dataclass
class _LinearConfig:
    """Internal configuration for _LinearModel. Not part of the public API."""
    input_dim: int = None
    output_dim: int = 1
    use_bias: bool = True
    init_weight_val: Optional[torch.Tensor] = None
    init_bias_val: Optional[torch.Tensor] = None


class _LinearModel(nn.Module):
    """Internal weighted linear layer. No SEM-specific logic. (masking is not done here)
    Not part of the public API and used internally by NodeLinear.
    """

    def __init__(self, cfg: _LinearConfig, device="cpu"):
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.input_dim
        output_dim = cfg.output_dim

        if cfg.init_weight_val is not None:
            assert isinstance(cfg.init_weight_val, torch.Tensor)
            assert cfg.init_weight_val.shape == (output_dim, input_dim), (
                f"init_weight_val must have shape ({output_dim}, {input_dim})"
            )
            self.weight = nn.Parameter(cfg.init_weight_val.to(device))
        else:
            self.weight = nn.Parameter(torch.empty(output_dim, input_dim, device=device))
            nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        if cfg.use_bias:
            if cfg.init_bias_val is not None:
                self.bias = nn.Parameter(cfg.init_bias_val.to(device))
            else:
                self.bias = nn.Parameter(torch.empty(output_dim, device=device))
                fan_in = nn.init._calculate_fan_in_and_fan_out(self.weight)[0]
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            assert cfg.init_bias_val is None, "init_bias_val must be None if use_bias is False"
            self.bias = None

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# ── MLP ───────────────────────────────────────────────────────────────────────

ACTIVATION_FUNCTIONS = {
    "linear": nn.Identity,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "selu": nn.SELU,
}



class _MLP(nn.Module):
    """Internal MLP implementation. Not part of the public API and used internally by NodeMLP.

    Args:
        config (MLPConfig): Configuration object for the MLP. If None, uses default values.
    """
    def __init__(self, config: MLPConfig = None, device="cpu"):
        super().__init__()

        layers = []
        # Combine input_dim and hidden_dims to create the architecture
        all_dims = [config.input_dim] + config.hidden_dims
        activation_fn = ACTIVATION_FUNCTIONS[config.activation]

        # Add hidden layers with selected activation and optional dropout
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            layers.append(activation_fn())
            if config.dropout > 0:
                layers.append(nn.Dropout(p=config.dropout))

        layers.append(nn.Linear(all_dims[-1], config.output_dim))

        # Wrap the layers in a Sequential module
        self.net = nn.Sequential(*layers)
        self.cfg = config
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, dim: int = -1,
                    ) -> Tensor:
    r"""
    Sample from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretize.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparameterization trick.
        ret = y_soft
    return ret


class DiscreteModes(StrEnum):
    GUMBEL = "gumbel"                   # Gumbel-Softmax, a soft approximation of a categorical distribution
    SOFTMAX = "softmax"                 # Return probabilities for each outcome
    SAMPLE_ONEHOT = "onehot"            # Return one-hot vectors sampled from the distribution
    SAMPLE = "sample"                   # Return a single sample from the distribution. Does not support training.
    
class _MLPDiscrete(nn.Module):
    """MLP for discrete nodes. Not part of the public API and used internally by NodeMLPDiscrete.

    Args:
        config (MLPConfig): Configuration object for the MLP. If None, uses default values.
    """
    def __init__(self, config: MLPConfig, mode: DiscreteModes, device="cpu"):
        super().__init__()
        self.mode = mode 
        
        # Combine input_dim and hidden_dims to create the architecture
        all_dims = [config.input_dim] + config.hidden_dims
        self.activation_fn = ACTIVATION_FUNCTIONS[config.activation]()
        self.dropout = config.dropout

        # Add hidden layers with selected activation and optional dropout
        self.layers = torch.nn.ModuleList()
        for i in range(len(all_dims) - 1):
            self.layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
        self.out_layer = nn.Linear(all_dims[-1], config.output_dim)

        self.cfg = config
        self.to(device)

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
        
    def forward(self, x: torch.Tensor, mode=DiscreteModes.SAMPLE) -> torch.Tensor:
        """ x.shape = [..., n_predecessors]
            mode: How to process the discrete node.

            if we are in SAMPLE mode:
                returns shape = [..., 1], where we draw from the observed distribution.
            Otherwise,
                returns shape = [..., n_outcomes], where we return probabilities for each outcome.
        """
        for layer in self.layers:
            x = layer(x)
            x = self.activation_fn(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout)

        x = self.out_layer(x)
        x = self._discretize(x, mode)
        return x

