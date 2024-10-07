from typing import Sequence, List
import os.path as osp
import random
from functools import cached_property
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_undirected
from torch_geometric.io import fs
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
from omegaconf import DictConfig
from utils.print_utils import cyan
from .data import SenderToReceiverData


class EllipticDataset(InMemoryDataset):
    def __init__(self, cfg: DictConfig, root="data/elliptic"):
        super(EllipticDataset, self).__init__(root)
        print(cyan("Loading Elliptic dataset..."))
        self.load(self.processed_paths[2])
        self.mask = fs.torch_load(self.processed_paths[1])
        print(cyan("Done loading Elliptic dataset!"))
        self.cfg = cfg

    @property
    def raw_file_names(self):
        return ["data_df.pkl", "node_idx_map.pt", "raw_emb.pt"]

    @property
    def processed_file_names(self):
        return ["emb.pt", "mask.pt", "data.pt"]

    def download(self):
        # TODO: implement download from url
        raise NotImplementedError

    def process(self):
        print(cyan("Processing Elliptic dataset..."))
        # load raw embeddings and generate new embeddings
        # using node_id_map
        print(cyan("Generating embeddings..."))
        node_id_map = torch.load(self.raw_paths[1])
        raw_emb = torch.load(self.raw_paths[2])
        emb = raw_emb[node_id_map]
        fs.torch_save(emb, self.processed_paths[0])

        print(cyan("Generating graph data..."))
        # load data_df and generate SenderToReceiverData objects & masks
        data_df = pd.read_pickle(self.raw_paths[0])
        data_list = []
        mask = []
        for _, row in data_df.iterrows():
            senders = torch.tensor(row.senders_mapped, dtype=torch.long)
            receivers = torch.tensor(row.receivers_mapped, dtype=torch.long)
            y = torch.tensor(row.labels, dtype=torch.long)
            data_list.append(SenderToReceiverData.from_data(senders, receivers, y))

            split = 0 if row.split == "TRN" else 1 if row.split == "VAL" else 2
            mask.append(split)

        fs.torch_save(torch.tensor(mask, dtype=torch.long), self.processed_paths[1])
        self.save(data_list, self.processed_paths[2])
        print(cyan("Done processing Elliptic dataset!"))

    def split(self):
        """
        returns train_dataset, val_dataset, test_dataset
        """
        return (
            self.apply_few_shot(self[self.mask == 0], self.cfg.shot_size),
            self[self.mask == 1],
            self[self.mask == 2],
        )

    @staticmethod
    def apply_few_shot(dataset: Sequence[SenderToReceiverData], shot_size: int = -1):
        """
        Apply few-shot learning to the dataset
        """
        if shot_size == -1:
            return dataset
        licit_dataset = [data for data in dataset if data.y == 0]
        illicit_dataset = [data for data in dataset if data.y == 1]
        print(cyan(f"Applying few-shot learning with shot size {shot_size}"))
        licit_shot = random.sample(licit_dataset, shot_size)
        illicit_shot = random.sample(illicit_dataset, shot_size)
        few_shot_dataset = licit_shot + illicit_shot
        random.shuffle(few_shot_dataset)
        return few_shot_dataset


class EllipticRecommendationEvalDataset(Dataset):
    """
    Dataset for edge recommendation task of the Elliptic dataset
    """

    def __init__(self, dataset: EllipticDataset, cfg: DictConfig):
        self.dataset = dataset
        self.num_illicits = cfg.num_illicits
        self.num_licits = cfg.num_licits
        self.num_samples = cfg.num_samples
        self._generate_data()

    def _generate_data(self):
        print(cyan("Generating Elliptic recommendation evaluation dataset..."))
        # identify illicit 1-1 transactions
        self._illicit_1_1_data_list = [
            data
            for data in self.dataset
            if data.y == 1 and data.senders.size(0) == 1 and data.receivers.size(0) == 1
        ]
        self._licit_data_list = [data for data in self.dataset if data.y == 0]
        print(
            f"Found {len(self._illicit_1_1_data_list)} illicit 1-1 transactions and {len(self._licit_data_list)} licit transactions"
        )

        data_list = []
        for _ in tqdm(range(self.num_samples)):
            data_list.append(self._generate_sample())
        self.data_list = data_list
        print(f"Avg density: {self.density:.4f}")
        print(
            f"which is {self.num_illicits} illicit edges out of {self.num_illicits / self.density:.1f} possible edges"
        )
        print(cyan("Done generating Elliptic recommendation evaluation dataset!"))

    @cached_property
    def density(self):
        density = [
            self.num_illicits / (senders.size(0) * receivers.size(0))
            for senders, receivers, _ in self.data_list
        ]
        return sum(density) / len(density)

    def _generate_sample(self):
        chosen_illicit = random.sample(self._illicit_1_1_data_list, self.num_illicits)
        chosen_licit = random.sample(self._licit_data_list, self.num_licits)

        # merge all chosen data into one SenderToReceiverData object
        senders = set()
        receivers = set()

        # take both senders and receivers from illicit transactions
        # and only senders from licit transactions
        for data in chosen_illicit:
            senders.update(set(data.senders.tolist()))
            receivers.update(set(data.receivers.tolist()))
        for data in chosen_licit:
            senders.update(set(data.senders.tolist()))

        senders = torch.tensor(list(senders), dtype=torch.long)
        receivers = torch.tensor(list(receivers), dtype=torch.long)
        illicit_senders = torch.tensor([], dtype=torch.long)
        illicit_receivers = torch.tensor([], dtype=torch.long)
        for illicit_data in chosen_illicit:
            illicit_senders = torch.cat((illicit_senders, illicit_data.senders.clone()))
            illicit_receivers = torch.cat(
                (illicit_receivers, illicit_data.receivers.clone())
            )
        illicit_edge_index = torch.stack(
            [illicit_senders, illicit_receivers], dim=0
        ).contiguous()
        assert (
            illicit_edge_index.size(0) == 2
            and illicit_edge_index.size(1) == self.num_illicits
        ), "Invalid illicit edge index shape"

        return senders, receivers, illicit_edge_index

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_list[idx]


class EllipticRecommendationDataset(EllipticDataset):
    def __init__(self, cfg: DictConfig, root="data/elliptic"):
        super().__init__(cfg, root)
        if cfg.use_edge_index and not fs.exists(self.processed_edge_index_path):
            self._generate_edge_index()

    @property
    def processed_edge_index_path(self):
        return osp.join(self.processed_dir, "edge_index.pt")

    def _generate_edge_index(self):
        illicit_edges = []
        print(cyan("Generating edge index..."))
        train_dataset = self[self.mask == 0]
        for data in self.filter_1_1(train_dataset[train_dataset.y == 1]):
            illicit_edges.append([data.senders.item(), data.receivers.item()])
        edge_index = torch.tensor(illicit_edges, dtype=torch.long).t()
        edge_index = to_undirected(edge_index).contiguous()
        fs.torch_save(edge_index, self.processed_edge_index_path)
        print(cyan("Done generating edge index:"), f"{edge_index.size(1)} edges")

    @staticmethod
    def filter_1_1(dataset: Sequence[SenderToReceiverData]):
        return [
            data
            for data in dataset
            if data.senders.size(0) == 1 and data.receivers.size(0) == 1
        ]

    def split(self):
        filter_fn = (
            EllipticRecommendationDataset.filter_1_1
            if self.cfg.filter_1_1
            else lambda x: x
        )

        return (
            filter_fn(self.augment(self[self.mask == 0])),
            filter_fn(self.augment(self[self.mask == 1])),
            EllipticRecommendationEvalDataset(self[self.mask == 2], self.cfg),
        )

    def augment(self, dataset: Sequence[SenderToReceiverData]):
        """
        Augment the dataset by merging samples into a single sample
        The number of samples to merge should follow a exponential distribution with min_merge and max_merge
        """
        if not self.cfg.augment.enabled:
            return dataset
        min_merge, max_merge, gamma = (
            self.cfg.augment.min,
            self.cfg.augment.max,
            self.cfg.augment.gamma,
        )
        print(
            cyan(
                f"Augmenting dataset by merging samples (min: {min_merge}, max: {max_merge}, gamma: {gamma})"
            )
        )
        shuffled_dataset = dataset[torch.randperm(len(dataset))]
        augmented_dataset = []
        curr_idx = 0
        while curr_idx < len(dataset):
            num_merge = min(
                self._sample_exponential_decay(gamma, min_merge, max_merge),
                len(dataset) - curr_idx,
            )
            merged_data = sum(shuffled_dataset[curr_idx + i] for i in range(num_merge))
            augmented_dataset.append(merged_data)
            curr_idx += num_merge
        return augmented_dataset

    @staticmethod
    def _sample_exponential_decay(gamma: float, min_val: int, max_val: int):
        values = np.arange(min_val, max_val + 1)
        probs = np.exp(-gamma * values)
        probs /= probs.sum()
        return np.random.choice(values, p=probs)
