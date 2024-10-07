from omegaconf import DictConfig
import torch
from torch import nn
import torch.nn.functional as F
from .feature_encoder import NodeIdEmbedding
from .conv import cf_conv_registry


class CFGNN(nn.Module):
    feature_encoder: nn.Module
    convs: nn.ModuleList

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.emb_path = cfg.emb_path
        self.edge_index_path = cfg.edge_index_path
        self.num_classes = cfg.num_classes
        self.num_layers = cfg.num_layers
        self.input_dim = cfg.input_dim
        self.dropout = cfg.dropout
        self.conv_type = cfg.conv
        self.aggr_type = cfg.aggr
        self.normailze = cfg.normalize
        self._build_model()

    def _build_model(self):
        self.feature_encoder = NodeIdEmbedding(self.emb_path)
        mlp_dim = (
            self.input_dim * (self.num_layers + 1)
            if self.aggr_type == "concat"
            else self.input_dim
        )
        self.post_mlp = nn.Linear(
            mlp_dim,
            mlp_dim,
        )
        self.register_buffer(
            "all_node_idx",
            torch.arange(
                self.feature_encoder.embedding.weight.shape[0], dtype=torch.long
            ),
        )
        self.register_buffer("all_edge_index", torch.load(self.edge_index_path))
        conv_klass = cf_conv_registry[self.conv_type]

        self.convs = nn.ModuleList(
            [
                conv_klass(
                    self.input_dim,
                    self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, batched_data):
        z = []
        x = self.feature_encoder(self.all_node_idx)
        z.append(F.normalize(x, p=2, dim=-1) if self.normailze else x)
        for conv in self.convs:
            x = conv(x, self.all_edge_index)
            z.append(F.normalize(x, p=2, dim=-1) if self.normailze else x)

        match self.aggr_type:
            case "concat":
                x = torch.cat(z, dim=-1)
            case "mean":
                x = torch.mean(torch.stack(z), dim=0)

        x = self.post_mlp(x)

        senders, receivers = batched_data.senders, batched_data.receivers

        return x[senders], x[receivers]
