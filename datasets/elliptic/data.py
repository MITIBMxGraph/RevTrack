from typing import Union
import torch
from torch_geometric.data import Data


class SenderToReceiverData(Data):
    """
    A data object representing a sender-to-receiver bipartite graph.
    node_idx: node indices (senders + receivers)
    senders: node indices of senders
    receivers: node indices of receivers
    edge_index: (relabeled) edge indices (senders -> receivers)
    num_nodes: number of nodes (senders + receivers)
    y: label
    """

    @staticmethod
    def from_data(senders: torch.Tensor, receivers: torch.Tensor, y: torch.Tensor):
        num_senders = senders.size(0)
        num_receivers = receivers.size(0)
        edge_index = torch.stack(
            torch.meshgrid(
                torch.arange(num_senders),
                torch.arange(num_senders, num_senders + num_receivers),
                indexing="ij",
            )
        ).reshape(2, -1)
        return SenderToReceiverData(
            node_idx=torch.cat([senders, receivers]),
            senders=senders,
            receivers=receivers,
            edge_index=edge_index,
            num_nodes=num_senders + num_receivers,
            y=y,
        )

    def __add__(self, other: Union["SenderToReceiverData", int]):
        if isinstance(other, int):
            return self
        senders = torch.cat([self.senders, other.senders]).unique()
        receivers = torch.cat([self.receivers, other.receivers]).unique()
        y = self.y or other.y
        return SenderToReceiverData.from_data(senders, receivers, y)

    def __radd__(self, other):
        return self + other
