from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm
from .feature_encoder import NodeIdFeatureEncoder
from .activation import activation_registry


class MLP(nn.Module):
    layers: nn.Sequential

    def __init__(self, cfg: DictConfig):
        super(MLP, self).__init__()
        self.cfg = cfg
        self.input_dim = cfg.input_dim
        self.hidden_dim = cfg.hidden_dim
        self.num_layers = cfg.num_layers
        self.dropout = cfg.dropout
        self.activation_type = cfg.activation
        self._build_model()

    def _build_model(self):
        activation_klass = activation_registry[self.activation_type]

        self.layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(
                        self.input_dim if i == 0 else self.hidden_dim, self.hidden_dim
                    ),
                    *(
                        [
                            GraphNorm(self.hidden_dim),
                            activation_klass(),
                            nn.Dropout(self.dropout),
                        ]
                        if i < self.num_layers - 1
                        else []
                    )
                )
                for i in range(self.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
