import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithm.policy.mlp import MLP


class GraphCNN(nn.Module):
    def __init__(self,
                 num_layers,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 device,
                 aggregate_type='none'):
        """
        num_layers: number of layers in the neural networks. (INCLUDING the input layer)
        num_mlp_layers: number of layers in MLPs. (EXCLUDING the input layer)
        input_dim: dimensionality of input features.
        hidden_dim: dimensionality of hidden units at ALL layers.
        device: which device to use.
        aggregate_type: how to aggregate neighbors (sum, average, or none)
        """

        super(GraphCNN, self).__init__()

        self.num_layers = num_layers
        self.device = device
        self.aggregate_type = aggregate_type
        # List of MLPs
        self.mlp_layers = torch.nn.ModuleList()
        # List of batch_norms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        self.mlp_layers.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        for layer in range(self.num_layers - 1):
            self.mlp_layers.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def next_layer(self, h, layer, adj_block):
        # average pooling neighboring nodes and center nodes altogether
        if self.aggregate_type == 'sum':
            h = torch.mm(adj_block, h)
        elif self.aggregate_type == 'average':
            degree = torch.mm(adj_block, torch.ones((adj_block.shape[0], 1)).to(self.device))
            h = torch.mm(adj_block, h) / degree
        # representation of neighboring and center nodes
        pooled_rep = self.mlp_layers[layer](h)
        h = self.batch_norms[layer](pooled_rep)
        # non-linearity
        h = F.relu(h)
        return h

    def forward(self, x_concat, graph_pool, adj_block):
        # list of hidden representation at each layer (including input)
        h = x_concat
        for layer in range(self.num_layers):
            h = self.next_layer(h, layer, adj_block=adj_block)
        h_nodes = h.clone()
        pooled_h = torch.sparse.mm(graph_pool, h)
        return pooled_h, h_nodes
