from functools import partial
import torch
from omegaconf import DictConfig
from torch_geometric.data import Batch
from einops import einsum
from datasets.elliptic.data import SenderToReceiverData
from algorithms.subgraph.utils.edge_recommendation_evaluator import (
    EdgeRecommendationEvaluator,
)
from .subgraph_algo import SubgraphAlgo
from .models import DoubleMLP, CFGNN

model_registry = dict(mlp=DoubleMLP, ngcf=CFGNN, lightgcn=CFGNN)


class DotProductAlgo(SubgraphAlgo):
    def __init__(self, cfg: DictConfig):
        self.top_k = cfg.top_k  # number of top edges to recommend
        self.test_every_n_epoch = cfg.test_every_n_epoch
        self.evaluator = EdgeRecommendationEvaluator()
        super().__init__(cfg)

    def _get_model_cls(self):
        return model_registry[self.cfg._name]

    def forward(self, batch):
        senders, receivers = self.model(batch)
        scores = (senders * receivers).sum(dim=-1)

        loss = self.criterion(scores, batch.y)
        return loss, scores

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            super().validation_step(batch, batch_idx, dataloader_idx)
        elif self.current_epoch % self.test_every_n_epoch == 0:
            self.test_step(batch, batch_idx, namespace="test")

    def test_step(self, batch, batch_idx: int, namespace: str = "final_test"):
        senders, receivers, illicit_edge_indices = batch
        batch_size = len(senders)

        new_batch = Batch.from_data_list(
            [
                SenderToReceiverData.from_data(s, r, torch.tensor([1]))
                for s, r in zip(senders, receivers)
            ],
            follow_batch=["senders", "receivers"],
        )
        senders, receivers, senders_batch, receivers_batch = (
            new_batch.senders,
            new_batch.receivers,
            new_batch.senders_batch,
            new_batch.receivers_batch,
        )
        sender_features, receiver_features = self.model(new_batch)
        # for each batch, find top k edges
        top_k_edges = []

        for i in range(batch_size):
            curr_top_k_edges = []
            curr_senders = senders[senders_batch == i]
            curr_receivers = receivers[receivers_batch == i]
            curr_sender_features = sender_features[senders_batch == i]
            curr_receiver_features = receiver_features[receivers_batch == i]
            # compute dot product for all pairs of senders and receivers
            scores = einsum(
                curr_sender_features, curr_receiver_features, "i d, j d -> i j"
            )
            # get top k edges
            top_k_indices = torch.topk(
                scores.flatten(), min(self.top_k, scores.size(0) * scores.size(1))
            ).indices
            top_k_senders = curr_senders[top_k_indices // scores.size(1)]
            top_k_receivers = curr_receivers[top_k_indices % scores.size(1)]
            for s, r in zip(top_k_senders, top_k_receivers):
                curr_top_k_edges.append((s.item(), r.item()))
            top_k_edges.append(curr_top_k_edges)

        hit_ratio, ndcg = self.evaluator(top_k_edges, illicit_edge_indices)

        log_fn = partial(
            self.log,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
            prog_bar=True,
        )

        log_fn(f"{namespace}/HR", hit_ratio)
        log_fn(f"{namespace}/NDCG", ndcg)
