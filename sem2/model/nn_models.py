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
from ..default_configs import LinearNodeConfig, MLPConfig


# ── Internal linear config + model ────────────────────────────────────────────

@dataclass
class _LinearConfig:
    """Internal configuration for _LinearModel. Not part of the public API."""
    input_dim: int = None
    output_dim: int = 1
    use_bias: bool = True
    init_weight_val: Optional[Tensor] = None
    init_bias_val: Optional[Tensor] = None


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

    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    
# class _MLPDiscrete(nn.Module):
#     """MLP for discrete nodes. Not part of the public API and used internally by NodeMLPDiscrete.
#
#     Args:
#         config (MLPConfig): Configuration object for the MLP. If None, uses default values.
#     """
#     def __init__(self, config: MLPConfig, mode: DiscreteModes, device="cpu"):
#         super().__init__()
#         self.mode = mode
#
#         # Combine input_dim and hidden_dims to create the architecture
#         all_dims = [config.input_dim] + config.hidden_dims
#         self.activation_fn = ACTIVATION_FUNCTIONS[config.activation]()
#         self.dropout = config.dropout
#
#         # Add hidden layers with selected activation and optional dropout
#         self.layers = torch.nn.ModuleList()
#         for i in range(len(all_dims) - 1):
#             self.layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
#         self.out_layer = nn.Linear(all_dims[-1], config.output_dim)
#
#         self.cfg = config
#         self.to(device)
#
#     def _discretize(self, x, mode):
#         """ Turn input logits into a representation of discrete values. """
#         if mode == DiscreteModes.SOFTMAX:
#             # Return probabilities for each outcome. Shape = [..., n_outcomes]
#             return torch.softmax(x, dim=-1)
#         elif mode == DiscreteModes.SAMPLE:
#             # Draw from the observed distribution. Shape = [..., 1]
#             if self.training:
#                 warnings.warn("Drawing sample for discrete nodes does not support backpropagation during training. \n"
#                               " To train discrete nodes, use another sampling strategy, via a different SEM2 class. ")
#             probs = torch.softmax(x, dim=-1)
#             sample = torch.distributions.Categorical(probs=probs).sample()
#             return sample.unsqueeze(-1)
#         else:
#             raise NotImplementedError(f"Discretisation mode {mode} not implemented yet.")
#
#     def forward(self, x: Tensor, mode=DiscreteModes.SAMPLE) -> Tensor:
#         """ x.shape = [..., n_predecessors]
#             mode: How to process the discrete node.
#
#             if we are in SAMPLE mode:
#                 returns shape = [..., 1], where we draw from the observed distribution.
#             Otherwise,
#                 returns shape = [..., n_outcomes], where we return probabilities for each outcome.
#         """
#         for layer in self.layers:
#             x = layer(x)
#             x = self.activation_fn(x)
#             if self.dropout > 0:
#                 x = F.dropout(x, p=self.dropout)
#
#         x = self.out_layer(x)
#         x = self._discretize(x, mode)
#         return x
