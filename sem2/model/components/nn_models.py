# Defer annotation evaluation so the Tensor type hint (imported only under
# TYPE_CHECKING) used in _LinearConfig's dataclass fields does not crash on
# Python < 3.14.
from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
if TYPE_CHECKING:
    from torch import Tensor

# LinearNodeConfig and MLPConfig are the public user-facing node configs.
# They live in default_configs and are re-exported here so internal node
# constructors can import from a single place.
from ...default_configs import MLPConfig


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
    def __init__(self, config: MLPConfig, device="cpu"):
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
        # Discrete parents can supply only integer state IDs. Convert at the
        # neural-kernel boundary, without changing their external representation.
        return self.net(x.to(next(self.net.parameters())))
