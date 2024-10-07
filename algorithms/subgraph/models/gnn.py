from omegaconf import DictConfig
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GraphNorm
from .feature_encoder import NodeIdFeatureEncoder
from .conv import conv_registry
from .pool import pool_registry
from .activation import activation_registry


class GNN(nn.Module):
    feature_encoder: nn.Module
    convs: nn.ModuleList
    gns: nn.ModuleList
    final_layers: nn.Sequential

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.emb_path = cfg.emb_path
        self.num_classes = cfg.num_classes
        self.num_layers = cfg.num_layers
        self.input_dim = cfg.input_dim
        self.hidden_dim = cfg.hidden_dim
        self.dropout = cfg.dropout
        self.conv_type = cfg.conv
        self.activation_type = cfg.activation
        self.pool_type = cfg.pool
        self._build_model()

    def _build_model(self):
        self.feature_encoder = NodeIdFeatureEncoder(self.emb_path)

        conv_klass = conv_registry[self.conv_type]
        activation_klass = activation_registry[self.activation_type]

        self.convs = nn.ModuleList(
            [
                conv_klass(
                    self.input_dim if i == 0 else self.hidden_dim,
                    self.hidden_dim,
                    activation_klass,
                )
                for i in range(self.num_layers)
            ]
        )

        self.gns = nn.ModuleList(
            [GraphNorm(self.hidden_dim) for _ in range(self.num_layers - 1)]
        )

        self.activation = activation_klass()

        self.pred_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            activation_klass(),
            nn.Linear(self.hidden_dim, self.num_classes if self.num_classes > 2 else 1),
        )

        self.pool = pool_registry[self.pool_type]

    def forward(self, batched_data):
        node_idx, edge_index = batched_data.node_idx, batched_data.edge_index

        x = self.feature_encoder(node_idx)
        for gn, conv in zip(self.gns, self.convs[:-1]):
            x = conv(x, edge_index)
            x = gn(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        h_graph = self.pool(x, batched_data.batch)
        return self.pred_mlp(h_graph)
