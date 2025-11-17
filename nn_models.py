from dataclasses import dataclass
import torch.nn as nn

@dataclass()
class MLPConfig:
    """ Configuration for the MLP model."""
    input_dim: int = None                  # Input dimension
    hidden_dims: list[int] = (128, 128, 128)      # List of hidden layer dimensions
    output_dim: int = 1                 # Output dimension
    prob_output: bool = False           # If True, applies Sigmoid to the output to output within range [0, 1]
    dropout: float = 0.0                # Dropout probability for regularization. 0 means no dropout
    activation: str = "relu"            # Activation function to use.

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
    def __init__(self, config: MLPConfig=None, device="cpu"):
        """
        Args:
            config (MLPConfig): Configuration object for the MLP. If None, uses default values.
        """
        super(MLP, self).__init__()

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

        self.to(device)
        self.cfg = config

    def forward(self, x):
        return self.net(x)
