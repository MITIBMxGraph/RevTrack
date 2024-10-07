import torch
import torch.nn as nn
from omegaconf import DictConfig
from .deep_sets import DeepSets
from .feature_encoder import NodeIdFeatureEncoder
from .activation import activation_registry


class DoubleDeepSets(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.emb_path = cfg.emb_path
        self.num_classes = cfg.num_classes
        self.hidden_dim = cfg.hidden_dim
        self.activation_type = cfg.activation
        self.dropout = cfg.dropout
        self.cfg = cfg
        self._build_model()

    def _build_model(self):
        self.feature_encoder = NodeIdFeatureEncoder(self.emb_path)
        self.sender_deep_sets = DeepSets(self.cfg)
        self.receiver_deep_sets = DeepSets(self.cfg)
        self.pred_mlp = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            activation_registry[self.activation_type](),
            nn.Linear(self.hidden_dim, self.num_classes if self.num_classes > 2 else 1),
        )

    def forward(self, batched_data):
        senders, receivers, senders_batch, receivers_batch = (
            batched_data.senders,
            batched_data.receivers,
            batched_data.senders_batch,
            batched_data.receivers_batch,
        )
        senders = self.feature_encoder(senders)
        receivers = self.feature_encoder(receivers)
        senders = self.sender_deep_sets(senders, senders_batch)
        receivers = self.receiver_deep_sets(receivers, receivers_batch)

        return self.pred_mlp(torch.cat([senders, receivers], dim=-1))
