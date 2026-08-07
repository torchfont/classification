from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from torch import Tensor, nn
from torch.optim import AdamW

from classification.module import FontClassifier
from classification.optim import WarmupCosineAnnealingLR


class LitFontClassifier(LightningModule):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        num_layers: int = 3,
        num_classes: int = 26,
        lr: float = 3e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = FontClassifier(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            num_classes=num_classes,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.test_preds: list[Tensor] = []
        self.test_labels: list[Tensor] = []
        self.num_classes = num_classes

    def forward(
        self,
        ops: Tensor,
        coords: Tensor,
    ) -> Tensor:
        return self.model(
            ops=ops,
            coords=coords,
            attention_mask=ops.ne(0).long(),
        )

    def training_step(
        self,
        batch: Any,
        _batch_idx: int,
    ) -> Tensor:
        ops, coords, labels = batch
        logits = self(ops, coords)

        loss = self.criterion(logits, labels)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=torch.distributed.is_initialized(),
        )
        return loss

    def validation_step(
        self,
        batch: Any,
        _batch_idx: int,
    ) -> None:
        ops, coords, labels = batch
        logits = self(ops, coords)

        loss = self.criterion(logits, labels)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=torch.distributed.is_initialized(),
        )

    def test_step(
        self,
        batch: Any,
        _batch_idx: int,
    ) -> None:
        ops, coords, labels = batch
        logits = self(ops, coords)

        loss = self.criterion(logits, labels)
        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=torch.distributed.is_initialized(),
        )
        preds = logits.argmax(dim=-1)
        self.test_preds.append(preds.detach().cpu())
        self.test_labels.append(labels.detach().cpu())

    def on_test_start(self) -> None:
        self.test_preds = []
        self.test_labels = []

    def on_test_epoch_end(self) -> None:
        if not self.test_preds:
            return

        preds_local = torch.cat(self.test_preds).to(self.device)
        labels_local = torch.cat(self.test_labels).to(self.device)

        preds_gathered = self.all_gather(preds_local)
        labels_gathered = self.all_gather(labels_local)

        preds = cast("Tensor", preds_gathered).reshape(-1)
        labels = cast("Tensor", labels_gathered).reshape(-1)

        if int(getattr(self, "global_rank", 0)) != 0:
            return

        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        idx_labels = list(range(self.num_classes))
        cm = confusion_matrix(labels_np, preds_np, labels=idx_labels)
        char_labels = [chr(ord("A") + i) for i in idx_labels]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=char_labels,
            yticklabels=char_labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Test Confusion Matrix")
        plt.yticks(rotation=0)
        fig.tight_layout()

        lg = self.trainer.logger
        log_dir = Path(getattr(lg, "log_dir", "."))
        log_dir.mkdir(parents=True, exist_ok=True)
        out_path = log_dir / "test_confusion_matrix.pdf"
        fig.savefig(out_path, dpi=350, format="pdf")
        plt.close(fig)

        report = classification_report(
            labels_np,
            preds_np,
            labels=idx_labels,
            target_names=char_labels,
            zero_division=0,
        )
        report_path = log_dir / "test_classification_report.txt"
        report_path.write_text(str(report), encoding="utf-8")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        no_decay = ["bias", "LayerNorm.weight"]

        params = list(self.named_parameters())
        decay_params = [
            p
            for n, p in params
            if p.requires_grad and not any(nd in n for nd in no_decay)
        ]
        nodecay_params = [
            p for n, p in params if p.requires_grad and any(nd in n for nd in no_decay)
        ]

        optimizer = AdamW(
            [
                {"params": decay_params},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )

        total_steps = max(
            1,
            int(getattr(self.trainer, "estimated_stepping_batches", 0)),
        )

        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
