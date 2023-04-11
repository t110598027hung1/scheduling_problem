import torch
import torch.nn as nn
import torch.nn.functional as F


# MLP with linear output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1,
                        this reduces to linear model.
            input_dim: dimensionality of input features.
            hidden_dim: dimensionality of hidden units at ALL layers.
            output_dim: number of classes for prediction.
        """

        super(MLP, self).__init__()

        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear_or_not = True
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.multi_linear = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            # Create input layer, hidden layer and output layer
            self.multi_linear.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.multi_linear.append(nn.Linear(hidden_dim, hidden_dim))
            self.multi_linear.append(nn.Linear(hidden_dim, output_dim))

            # Create Batch Normalization between each layer
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.multi_linear[layer](h)))
            return self.multi_linear[self.num_layers - 1](h)


class MLPActor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1,
                        this reduces to linear model.
            input_dim: dimensionality of input features.
            hidden_dim: dimensionality of hidden units at ALL layers.
            output_dim: number of classes for prediction.
            device: which device to use.
        """

        super(MLPActor, self).__init__()

        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear_or_not = True
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.multi_linear = torch.nn.ModuleList()

            self.multi_linear.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.multi_linear.append(nn.Linear(hidden_dim, hidden_dim))
            self.multi_linear.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = torch.tanh(self.multi_linear[layer](h))
            return self.multi_linear[self.num_layers - 1](h)


class MLPCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1,
                        this reduces to linear model.
            input_dim: dimensionality of input features.
            hidden_dim: dimensionality of hidden units at ALL layers.
            output_dim: number of classes for prediction.
        """

        super(MLPCritic, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.multi_linear = torch.nn.ModuleList()

            self.multi_linear.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.multi_linear.append(nn.Linear(hidden_dim, hidden_dim))
            self.multi_linear.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = torch.tanh(self.multi_linear[layer](h))
            return self.multi_linear[self.num_layers - 1](h)
