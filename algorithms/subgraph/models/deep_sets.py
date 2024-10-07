from omegaconf import DictConfig
import torch
import torch.nn as nn
from .activation import activation_registry
from .pool import pool_registry


class DeepSets(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(DeepSets, self).__init__()
        self.cfg = cfg
        self.input_dim = cfg.input_dim
        self.hidden_dim = cfg.hidden_dim
        self.num_layers = cfg.num_layers
        self._build_model()

    def _build_model(self):
        self.init_mlp = nn.Linear(self.input_dim, self.hidden_dim)
        self.equivarant_layers = nn.ModuleList(
            [EquivariantLayer(self.cfg) for _ in range(self.num_layers)]
        )
        self.invariant_layer = InvariantLayer(self.cfg)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.init_mlp(x)
        for layer in self.equivarant_layers:
            x = layer(x, batch)
        return self.invariant_layer(x, batch)


class EquivariantLayer(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.activation_type = cfg.activation
        self.pool_type = cfg.pool
        self._build_model()

    def _build_model(self):
        self.lamb = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.activation = activation_registry[self.activation_type]()
        self.pool = pool_registry[self.pool_type]

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x, batch)
        return self.activation(
            self.lamb * x + self.gamma * pooled.index_select(dim=0, index=batch)
        )


class InvariantLayer(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.activation_type = cfg.activation
        self.pool_type = cfg.pool
        self.dropout = cfg.dropout
        self._build_model()

    def _build_model(self):
        activation_klass = activation_registry[self.activation_type]
        self.phi = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            activation_klass(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(p=self.dropout),
        )
        self.pool = pool_registry[self.pool_type]
        self.rho = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            activation_klass(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return self.rho(self.pool(self.phi(x), batch))
