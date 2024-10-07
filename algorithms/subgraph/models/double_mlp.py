from omegaconf import DictConfig
import torch.nn as nn
from .mlp import MLP
from .feature_encoder import NodeIdFeatureEncoder


class DoubleMLP(nn.Module):
    feature_encoder: nn.Module
    sender_mlp: nn.Module
    receiver_mlp: nn.Module

    def __init__(self, cfg: DictConfig):
        super(DoubleMLP, self).__init__()
        self.cfg = cfg
        self.emb_path = cfg.emb_path
        self._build_model()

    def _build_model(self):
        self.feature_encoder = NodeIdFeatureEncoder(self.emb_path)
        self.sender_mlp = MLP(self.cfg)
        self.receiver_mlp = MLP(self.cfg)

    def forward(self, batched_data):
        senders, receivers = batched_data.senders, batched_data.receivers
        senders = self.feature_encoder(senders)
        receivers = self.feature_encoder(receivers)
        senders = self.sender_mlp(senders)
        receivers = self.receiver_mlp(receivers)
        return senders, receivers
