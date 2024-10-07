from functools import partial
from omegaconf import DictConfig
import torch
import wandb
from torch import nn
from torchmetrics import F1Score
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryConfusionMatrix,
)
from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from .models import GNN, DoubleDeepSets


model_registry = dict(bipartite_gnn=GNN, deepsets=DoubleDeepSets)


class SubgraphAlgo(BasePytorchAlgo):
    def __init__(self, cfg: DictConfig):
        self.num_classes = cfg.dataset.num_classes
        self.is_multilabel = cfg.dataset.is_multilabel
        self.is_binary = self.num_classes == 2 and not self.is_multilabel
        super().__init__(cfg)

    def _get_model_cls(self):
        return model_registry[self.cfg._name]

    def _build_model(self):
        model_cls = self._get_model_cls()
        self.model = model_cls(self.cfg.model)

        loss_fn = (
            nn.CrossEntropyLoss()
            if not self.is_multilabel and not self.is_binary
            else nn.BCEWithLogitsLoss()
        )
        self.criterion = (
            loss_fn
            if not self.is_multilabel and not self.is_binary
            else lambda x, y: (loss_fn(x.flatten(), y.flatten().float()))
        )

        self.train_f1, self.train_auroc, self.train_prauc, self.train_conf_matrix = (
            self._build_metrics()
        )
        self.val_f1, self.val_auroc, self.val_prauc, self.val_conf_matrix = (
            self._build_metrics()
        )
        self.test_f1, self.test_auroc, self.test_prauc, self.test_conf_matrix = (
            self._build_metrics()
        )

    def _build_metrics(self):
        f1 = (
            BinaryF1Score()
            if self.is_binary
            else F1Score(
                task="multilabel" if self.is_multilabel else "multiclass",
                average="micro",
                **{
                    (
                        "num_labels" if self.is_multilabel else "num_classes"
                    ): self.num_classes
                },
            )
        )
        auroc = BinaryAUROC() if self.is_binary else None
        prauc = BinaryAveragePrecision() if self.is_binary else None
        conf_matrix = BinaryConfusionMatrix() if self.is_binary else None
        return f1, auroc, prauc, conf_matrix

    def _get_metrics(self, namespace: str):
        match namespace:
            case "training":
                return (
                    self.train_f1,
                    self.train_auroc,
                    self.train_prauc,
                    self.train_conf_matrix,
                )
            case "validation":
                return self.val_f1, self.val_auroc, self.val_prauc, self.val_conf_matrix
            case "test" | "final_test":
                return (
                    self.test_f1,
                    self.test_auroc,
                    self.test_prauc,
                    self.test_conf_matrix,
                )

    def forward(self, batch):
        output = self.model(batch)
        loss = self.criterion(output, batch.y)
        return loss, output

    def training_step(self, batch):
        return self._step(batch, "training")

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        namespace = "validation" if dataloader_idx == 0 else "test"
        return self._step(batch, namespace)

    def test_step(self, batch):
        return self._step(batch, "final_test")

    def on_test_start(self) -> None:
        for metric in self._get_metrics("final_test"):
            metric.reset()
        return super().on_test_start()

    def on_test_end(self) -> None:
        conf_matrix = self.test_conf_matrix.compute()
        print("conf_matrix", conf_matrix)
        self.log_conf_matrix("final_test/conf_matrix", conf_matrix)
        return super().on_test_end()

    def _step(self, batch, namespace: str = "training"):
        loss, output = self.forward(batch)

        self.log(
            f"{namespace}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        f1, auroc, prauc, conf_matrix = self._get_metrics(namespace)
        target = batch.y

        if self.is_binary:
            target = target.flatten()
            output = output.flatten()

        f1(output, target)
        logging_fn = partial(
            self.log, on_step=False, on_epoch=True, add_dataloader_idx=False
        )

        logging_fn(
            f"{namespace}/f1",
            f1,
        )

        if self.is_binary:
            auroc(output, target)
            prauc(output, target)

            logging_fn(
                f"{namespace}/auroc",
                auroc,
            )

            logging_fn(
                f"{namespace}/prauc",
                prauc,
            )

        if self.is_binary and namespace == "final_test":
            conf_matrix(output, target)

        return loss

    def log_conf_matrix(self, key: str, conf_matrix: torch.Tensor):
        conf_matrix = conf_matrix.detach().cpu().numpy()
        data = []
        class_names = (
            ["licit", "suspicious"]
            if self.cfg.dataset.name == "elliptic"
            else ["0", "1"]
        )

        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                data.append([class_names[i], class_names[j], conf_matrix[i, j]])

        fields = {
            "Actual": "Actual",
            "Predicted": "Predicted",
            "nPredictions": "nPredictions",
        }

        self.logger.experiment.log(
            {
                key: self.logger.experiment.plot_table(
                    "wandb/confusion_matrix/v1",
                    wandb.Table(
                        columns=["Actual", "Predicted", "nPredictions"], data=data
                    ),
                    fields,
                    {"title": key},
                    split_table=False,
                ),
            }
        )
