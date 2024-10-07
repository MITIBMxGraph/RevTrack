import os
from functools import partial
from pathlib import Path
from typing import Optional, Union
import torch
from omegaconf import DictConfig
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch_geometric.loader import DataLoader as _DataLoader
from datasets import EllipticDataset
from algorithms.subgraph import SubgraphAlgo
from .exp_base import BaseLightningExperiment

DataLoader = partial(_DataLoader, follow_batch=["subgraph_id"])


class SubgraphClassificationExperiment(BaseLightningExperiment):
    compatible_algorithms = dict(
        bipartite_gnn=SubgraphAlgo,
        deepsets=SubgraphAlgo,
    )

    compatible_datasets = dict(
        elliptic=EllipticDataset,
    )

    def __init__(
        self,
        root_cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(root_cfg, logger, ckpt_path)
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def _build_dataset(self, split: str) -> Optional[torch.utils.data.Dataset]:
        if (
            self.train_dataset is None
            or self.val_dataset is None
            or self.test_dataset is None
        ):
            dataset = self.compatible_datasets[self.root_cfg.dataset._name](
                cfg=self.root_cfg.dataset
            )
            self.train_dataset, self.val_dataset, self.test_dataset = dataset.split()

        match split:
            case "training":
                return self.train_dataset
            case "validation":
                return self.val_dataset
            case "test":
                return self.test_dataset
            case _:
                raise ValueError(
                    f"split {split} not available for subgraph classification datasets"
                )

    def _build_training_loader(
        self,
    ) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        train_dataset = self._build_dataset("training")
        shuffle = (
            False
            if isinstance(train_dataset, torch.utils.data.IterableDataset)
            else self.cfg.training.data.shuffle
        )
        if train_dataset:
            return DataLoader(
                train_dataset,
                batch_size=(
                    self.cfg.training.batch_size
                    if self.cfg.training.batch_size > 0
                    else len(train_dataset)
                ),
                num_workers=min(os.cpu_count(), self.cfg.training.data.num_workers),
                shuffle=shuffle,
                persistent_workers=True,
                follow_batch=["senders", "receivers"],
            )
        else:
            return None

    def _build_validation_loader(
        self,
    ) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        validation_dataset = self._build_dataset("validation")
        shuffle = (
            False
            if isinstance(validation_dataset, torch.utils.data.IterableDataset)
            else self.cfg.validation.data.shuffle
        )
        if validation_dataset:
            return DataLoader(
                validation_dataset,
                batch_size=(
                    self.cfg.validation.batch_size
                    if self.cfg.validation.batch_size > 0
                    else len(validation_dataset)
                ),
                num_workers=min(os.cpu_count(), self.cfg.validation.data.num_workers),
                shuffle=shuffle,
                persistent_workers=True,
                follow_batch=["senders", "receivers"],
            )
        else:
            return None

    def _build_test_loader(
        self,
    ) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        test_dataset = self._build_dataset("test")
        shuffle = (
            False
            if isinstance(test_dataset, torch.utils.data.IterableDataset)
            else self.cfg.test.data.shuffle
        )
        if test_dataset:
            return DataLoader(
                test_dataset,
                batch_size=(
                    self.cfg.test.batch_size
                    if self.cfg.test.batch_size > 0
                    else len(test_dataset)
                ),
                num_workers=min(os.cpu_count(), self.cfg.test.data.num_workers),
                shuffle=shuffle,
                persistent_workers=True,
                follow_batch=["senders", "receivers"],
            )
        else:
            return None
