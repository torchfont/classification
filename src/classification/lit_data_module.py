from collections.abc import Sequence
from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torchfont.datasets import GoogleFonts
from torchfont.transforms import LimitSequenceLength


def collate_fn(
    batch: Sequence[tuple[Tensor, Tensor, int, int]],
) -> tuple[Tensor, Tensor, Tensor]:
    types_list = [types for types, _, _, _ in batch]
    coords_list = [coords for _, coords, _, _ in batch]
    content_classes = [content for _, _, _, content in batch]

    types_tensor = pad_sequence(types_list, batch_first=True, padding_value=0)
    coords_tensor = pad_sequence(coords_list, batch_first=True, padding_value=0.0)
    content_tensor = torch.as_tensor(content_classes, dtype=torch.long)

    return types_tensor, coords_tensor, content_tensor


class LitGoogleFonts(LightningDataModule):
    def __init__(
        self,
        root: str = "data/google/fonts",
        ref: str = "main",
        max_len: int = 128,
        batch_size: int = 256,
        num_workers: int = 2,
        prefetch_factor: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.root = Path(root)
        self.ref = ref
        self.max_seq_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.val_ratio = 0.05
        self.test_ratio = 0.05
        self.seed = 37414078

        transform = LimitSequenceLength(max_len=self.max_seq_len)
        codepoints = [ord("A") + i for i in range(26)]
        dataset = GoogleFonts(
            root=Path(self.root),
            ref=self.ref,
            codepoint_filter=codepoints,
            transform=transform,
            download=True,
        )
        self.dataset = dataset
        self.commit_hash = dataset.commit_hash
        self.dataset_len = len(dataset)
        self.num_style_classes = len(dataset.style_classes)
        self.num_content_classes = len(dataset.content_classes)
        self.save_hyperparameters(
            {
                "commit_hash": self.commit_hash,
                "dataset_len": self.dataset_len,
                "num_style_classes": self.num_style_classes,
                "num_content_classes": self.num_content_classes,
            },
        )
        length = len(self.dataset)
        g = torch.Generator().manual_seed(self.seed)

        n_test = int(length * self.test_ratio)
        n_val = int(length * self.val_ratio)
        n_train = max(length - n_val - n_test, 0)

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [n_train, n_val, n_test],
            generator=g,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
            multiprocessing_context="fork",
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
            multiprocessing_context="fork",
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
            multiprocessing_context="fork",
        )
