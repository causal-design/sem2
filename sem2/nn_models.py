from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#Things to change in this file : The Public LinearNoddeConfig and the MLPConfig have to be moved to the default configs file once we make it

# ── Linear ────────────────────────────────────────────────────────────────────

@dataclass
class LinearNodeConfig:
    """Public Configuration for linear SEM nodes. (to move to our default configs folder once we decide on it)
    Will have to discuss but my thoughts: 
    1) input_dim is not exposed —> it is inferred from the graph structure.
    2) init_weight_val and init_bias_val are not exposed since fixed coefficients are specified inline in the formula syntax (e.g. '1*x', '-0.5*x') 
        rather than through config, keeping the model formula self-contained and readable.
    """

    # I want to expose output_dim and not forced to 1 since a linear node may output probabilities over discrete states (e.g. a discrete latent variable). The multi-output linear case
    # is I think essentially a linear classifier / softmax layer, which is a valid SEM component (e.g. for use in exact marginalisation and other methods over discrete variables).
    output_dim: int = 1

    # Bias is the intercept term SO without it you force the linear relationship through the origin, which is a strong structural constraint. Default is True to allow learned offsets. 
    # Set to False only if your model explicitly requires this constraint. (good to expose for some types of assumptions in experiments I think)
    use_bias: bool = True



@dataclass
class _LinearConfig:
    """Internal configuration for _LinearModel. Not part of the public API."""
    input_dim: int = None
    output_dim: int = 1
    use_bias: bool = True
    init_weight_val: Optional[torch.Tensor] = None
    init_bias_val: Optional[torch.Tensor] = None


class _LinearModel(nn.Module):
    """Internal weighted linear layer. No SEM-specific logic. (basically masking is not done here)
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

@dataclass
class MLPConfig:
    """Configuration for the MLP model.  (to move to our default configs folder once we decide on it) """
    input_dim: Optional[int] = None                                     # Input dimension
    hidden_dims: list = field(default_factory=lambda: [128, 128, 128]) # List of hidden layer dimensions
    output_dim: int = 1                                                  # Output dimension
    prob_output: bool = False                                           # If True, applies Sigmoid to the output to output within range [0, 1]
    dropout: float = 0.0                                                # Dropout probability for regularization. 0 means no dropout
    activation: str = "relu"                                            # Activation function to use.

    def __post_init__(self):
        self.hidden_dims = list(self.hidden_dims)


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


class MLP(nn.Module):
    """
    Args:
        config (MLPConfig): Configuration object for the MLP. If None, uses default values.
    """
    def __init__(self, config: MLPConfig = None, device="cpu"):
        super().__init__()

        if config is None:
            config = MLPConfig()

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

        if config.prob_output:
            layers.append(nn.Sigmoid())

        # Wrap the layers in a Sequential module
        self.net = nn.Sequential(*layers)
        self.cfg = config
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)