import os
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseFeatureEncoder(nn.Module, ABC):

    @abstractmethod
    def forward(self, node_idx: torch.Tensor):
        """
        node_idx: 1D tensor of node indices
        """


class OneFeatureEncoder(BaseFeatureEncoder):
    """
    Every node has the same feature [1]
    """

    def __init__(self):
        super().__init__()

    def forward(self, node_idx: torch.Tensor):
        return torch.ones(
            node_idx.size(0), 1, device=node_idx.device, dtype=torch.float
        )


class NodeIdFeatureEncoder(BaseFeatureEncoder):
    """
    Feature encoder that returns node id embeddings
    """

    def __init__(self, emb_path: str):
        super().__init__()
        # load embeddings from path
        self.emb_path = emb_path
        self.emb = None
        if os.path.exists(emb_path):  # this might happen at the first run
            self.emb = torch.load(emb_path)

    def forward(self, node_idx: torch.Tensor):
        if self.emb is None:
            self.emb = torch.load(self.emb_path)
        if self.emb.device != node_idx.device:
            self.emb = self.emb.to(node_idx.device)

        return self.emb[node_idx]


class NodeIdEmbedding(BaseFeatureEncoder):
    """
    Feature encoder that returns node id embeddings
    Uses an embedding layer initialized with embeddings from path
    """

    def __init__(self, emb_path: str):
        super().__init__()
        self.emb_path = emb_path
        self.embedding = None
        self._init_embedding()

    def _init_embedding(self, device=None):
        if os.path.exists(self.emb_path):
            emb = torch.load(self.emb_path)
            self.embedding = nn.Embedding.from_pretrained(emb, freeze=False)
            if device is not None:
                self.embedding = self.embedding.to(device)

    def forward(self, node_idx: torch.Tensor):
        if self.embedding is None:
            self._init_embedding(node_idx.device)
        return self.embedding(node_idx)
