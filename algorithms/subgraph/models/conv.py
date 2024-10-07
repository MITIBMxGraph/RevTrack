import torch.nn as nn
from torch_geometric.nn import (
    GCNConv,
    GINConv as _GINConv,
    GATConv,
    TransformerConv,
    SuperGATConv,
    MessagePassing,
)
import torch.nn.functional as F
from torch_geometric.utils import degree


# NGCFConv, LightGCNConv from:
# https://medium.com/stanford-cs224w/recommender-systems-with-gnns-in-pyg-d8301178e377


class NGCFConv(MessagePassing):
    def __init__(self, in_channels, dropout, bias=True, **kwargs):
        super(NGCFConv, self).__init__(aggr="add", **kwargs)
        self.dropout = dropout
        self.lin_1 = nn.Linear(in_channels, in_channels, bias=bias)
        self.lin_2 = nn.Linear(in_channels, in_channels, bias=bias)
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.lin_1.weight)
        nn.init.xavier_uniform_(self.lin_2.weight)

    def forward(self, x, edge_index):
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
        # Start propagating messages
        out = self.propagate(edge_index, x=(x, x), norm=norm)
        out += self.lin_1(x)
        out = F.dropout(out, self.dropout, self.training)
        return F.leaky_relu(out)

    def message(self, x_j, x_i, norm):
        return norm.view(-1, 1) * (self.lin_1(x_j) + self.lin_2(x_j * x_i))


class LightGCNConv(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(aggr="add")

    def forward(self, x, edge_index):
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
        # Start propagating messages
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GINConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=nn.ELU):
        super().__init__()
        self.conv = _GINConv(
            nn.Sequential(
                nn.Linear(in_channels, out_channels),
                activation(),
                nn.Linear(out_channels, out_channels),
            )
        )

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


def conv_fn(conv_klass):
    return lambda input_dim, hidden_dim, activation: conv_klass(input_dim, hidden_dim)


conv_registry = {
    "GCN": conv_fn(GCNConv),
    "GIN": GINConv,
    "GAT": conv_fn(GATConv),
    "Transformer": conv_fn(TransformerConv),
    "SuperGAT": conv_fn(SuperGATConv),
}

cf_conv_registry = {
    "NGCF": NGCFConv,
    "LightGCN": lambda input_dim, dropout: LightGCNConv(),
}
