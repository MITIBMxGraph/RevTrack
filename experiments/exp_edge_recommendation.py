import os
from typing import Optional, Union
import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from algorithms.subgraph import IterativeFilteringAlgo, DotProductAlgo
from datasets import EllipticRecommendationDataset
from .exp_subgraph_classification import SubgraphClassificationExperiment
from utils.distributed_utils import is_rank_zero


class EdgeRecommendationExperiment(SubgraphClassificationExperiment):
    compatible_algorithms = dict(
        iterative_filtering=IterativeFilteringAlgo,
        mlp=DotProductAlgo,
        ngcf=DotProductAlgo,
        lightgcn=DotProductAlgo,
    )

    compatible_datasets = dict(
        elliptic_recommendation=EllipticRecommendationDataset,
    )

    def test(self) -> None:
        super().test()

        if is_rank_zero and self.logger:
            self.logger.experiment.log(
                {
                    "final_test/density": self.test_dataset.density,
                    "final_test/sparsity": 1 - self.test_dataset.density,
                }
            )

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
            return torch.utils.data.DataLoader(
                test_dataset,
                batch_size=(
                    self.cfg.test.batch_size
                    if self.cfg.test.batch_size > 0
                    else len(test_dataset)
                ),
                num_workers=min(os.cpu_count(), self.cfg.test.data.num_workers),
                shuffle=shuffle,
                persistent_workers=False,
                collate_fn=self._collate_fn,
            )
        else:
            return None

    def _collate_fn(self, batch):
        return tuple(zip(*batch))
