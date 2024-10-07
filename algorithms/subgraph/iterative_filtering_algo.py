import math
import numpy as np
from typing import List
from functools import partial
import torch
from omegaconf import DictConfig
from torch_geometric.data import Batch
from algorithms.subgraph.utils.edge_recommendation_evaluator import (
    EdgeRecommendationEvaluator,
)
from datasets.elliptic.data import SenderToReceiverData
from .models import DoubleDeepSets
from .subgraph_algo import SubgraphAlgo


class IterativeFilteringAlgo(SubgraphAlgo):
    def __init__(self, cfg: DictConfig):
        self.top_k = cfg.top_k  # number of top edges to recommend
        self.keep_top_k = int(
            self.top_k * cfg.keep_multiplier
        )  # keep at most this many edges after each iteration
        self.test_every_n_epoch = cfg.test_every_n_epoch
        self.evaluator = EdgeRecommendationEvaluator()
        super().__init__(cfg)

    def _get_model_cls(self):
        return DoubleDeepSets

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            super().validation_step(batch, batch_idx, dataloader_idx)
        elif self.current_epoch % self.test_every_n_epoch == 0:
            self.test_step(batch, batch_idx, namespace="test")

    def test_step(self, batch, batch_idx: int, namespace: str = "final_test"):
        senders, receivers, illicit_edge_indices = batch
        batch_size = len(senders)

        groups = [
            [SenderToReceiverData.from_data(s, r, torch.tensor([1]))]
            for s, r in zip(senders, receivers)
        ]

        estimated_iters = math.ceil(
            math.log(groups[0][0].senders.size(0) * groups[0][0].receivers.size(0), 4)
        )
        keep_top_k = self.keep_top_k
        decrease_k_by = 2 * (self.keep_top_k - self.top_k) / estimated_iters

        while not all(self._is_done(groups)):
            is_done = self._is_done(groups)

            # done group remains the same
            # undone group split into 4
            groups = [
                group if done else self._split_group(group)
                for group, done in zip(groups, is_done)
            ]

            # choose groups to forward pass (compute scores)
            should_forward = [len(group) > self.top_k for group in groups]
            # choose data to forward pass (all data in should_forward groups, by zipping with should_forward)
            data_list = [
                data
                for group, forward in zip(groups, should_forward)
                if forward
                for data in group
            ]

            if len(data_list) == 0:
                continue

            batch = Batch.from_data_list(
                data_list, follow_batch=["senders", "receivers"]
            )

            # sort each forwarded group by scores
            scores = self.model(batch).flatten().detach().cpu().tolist()
            groups = self._sort_by_scores(groups, scores, should_forward)

            # keep keep_top_k data in each forwarded group
            groups = [
                group[:keep_top_k] if forward else group
                for group, forward in zip(groups, should_forward)
            ]

            # for each forwarded group, if top-k edges are all 1-1, only keep the top-k edges (mark as done)
            groups = [
                (
                    group[: self.top_k]
                    if forward and all(self._is_data_1_1(data) for data in group)
                    else group
                )
                for group, forward in zip(groups, should_forward)
            ]

            keep_top_k = int(max(self.top_k, keep_top_k - decrease_k_by))

        top_k_edges = [
            [(data.senders[0].item(), data.receivers[0].item()) for data in group]
            for group in groups
        ]

        hit_ratio, ndcg = self.evaluator(top_k_edges, illicit_edge_indices)

        log_fn = partial(
            self.log,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        log_fn(f"{namespace}/HR", hit_ratio)
        log_fn(f"{namespace}/NDCG", ndcg)

    @staticmethod
    def _sort_by_scores(
        groups: List[List[SenderToReceiverData]],
        scores: List[float],
        forwarded: List[bool],
    ) -> List[List[SenderToReceiverData]]:
        """
        Sort each group by scores and return the sorted groups
        """
        num_data = [
            len(group) if forward else 0 for group, forward in zip(groups, forwarded)
        ]
        score_start_indices = [0] + list(np.cumsum(num_data))[:-1]
        return [
            (
                [
                    data
                    for _, data in sorted(
                        zip(
                            scores[
                                score_start_indices[i] : score_start_indices[i]
                                + len(group)
                            ],
                            group,
                        ),
                        reverse=True,
                        key=lambda x: x[0],
                    )
                ]
                if forward
                else group
            )
            for i, (group, forward) in enumerate(zip(groups, forwarded))
        ]

    def _is_done(self, groups: List[List[SenderToReceiverData]]) -> List[bool]:
        """
        Given a batch of SenderToReceiverData objects, return a list of booleans indicating whether each group is done recommending top-k edges
        """
        return [
            all(IterativeFilteringAlgo._is_data_1_1(data) for data in group)
            and len(group) <= self.top_k
            for group in groups
        ]

    @staticmethod
    def _split_data(
        data: SenderToReceiverData,
    ) -> List[SenderToReceiverData]:
        """
        Split a SenderToReceiverData object into 4 SenderToReceiverData objects
        by splitting the senders and receivers into 2 parts each.
        """
        if len(data.senders) == 1 and len(data.receivers) == 1:
            return [data]

        def split_nodes(nodes):
            return torch.chunk(nodes, 2) if len(nodes) > 1 else (nodes,)

        all_senders = split_nodes(data.senders)
        all_receivers = split_nodes(data.receivers)
        return [
            SenderToReceiverData.from_data(senders, receivers, data.y)
            for senders in all_senders
            for receivers in all_receivers
        ]

    @classmethod
    def _split_group(
        cls, group: List[SenderToReceiverData]
    ) -> List[SenderToReceiverData]:
        return [d for data in group for d in cls._split_data(data)]

    @staticmethod
    def _is_data_1_1(data: SenderToReceiverData) -> bool:
        return len(data.senders) == 1 and len(data.receivers) == 1
