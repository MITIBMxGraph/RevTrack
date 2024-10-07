import random
import numpy as np
import torch
from omegaconf import DictConfig


def set_deterministic_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def recompute_batch_size(cfg: DictConfig):
    """
    Recompute batch size based on the number of gpus available.
    (e.g. for a batch size of 64, and 4 gpus, the batch size per gpu will be 16)
    """
    num_gpus = torch.cuda.device_count()

    def get_new_batch_size(batch_size: int):
        if batch_size == -1:
            return -1
        if batch_size % num_gpus != 0:
            raise ValueError(
                f"Effective batch size ({batch_size}) must be divisible by the number of gpus ({num_gpus})"
            )
        return batch_size // num_gpus

    cfg.experiment.training.batch_size = get_new_batch_size(
        cfg.experiment.training.batch_size
    )
    cfg.experiment.validation.batch_size = get_new_batch_size(
        cfg.experiment.validation.batch_size
    )
    cfg.experiment.test.batch_size = get_new_batch_size(cfg.experiment.test.batch_size)
    cfg.experiment.num_gpus = num_gpus


def override_exp_edge_recommendation_cfg(cfg: DictConfig):
    """
    If shortcut field is present in the experiment config,
    override the corresponding fields in the dataset and algorithm config
    shortcut is a string in the form of "{dataset.num_illicits}+{dataset.num_licits}@{algorithm.top_k}"
    """
    if hasattr(cfg, "shortcut"):
        shortcut = cfg.shortcut
        dataset_str, top_k = shortcut.split("@")
        num_illicits, num_licits = dataset_str.split("+")
        cfg.dataset.num_illicits = int(num_illicits)
        cfg.dataset.num_licits = int(num_licits)
        cfg.algorithm.top_k = int(top_k)
