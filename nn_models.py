import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, prob=False, dropout=0.0):
        """
        Args:
            input_dim (int): Dimension of the input features.
            hidden_dims (list of int): List containing the number of units in each hidden layer.
            output_dim (int): Dimension of the output.
            prob (boolean): Add a final Sigmoid layer?
            dropout (float): Dropout probability (default: 0.0, no dropout).
        """
        super(MLP, self).__init__()
        layers = []
        # Combine input_dim and hidden_dims to create the architecture
        all_dims = [input_dim] + hidden_dims

        # Add hidden layers with ReLU activation and optional dropout
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        # Add the final output layer (without activation; adjust if needed)
        layers.append(nn.Linear(all_dims[-1], output_dim))

        if prob:
            layers.append(nn.Sigmoid())

        # Wrap the layers in a Sequential module
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
