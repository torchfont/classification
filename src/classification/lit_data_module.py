from collections.abc import Sequence
from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.transforms import load_glyph


def _make_transform(max_len: int):
    def transform(sample: GlyphSample) -> tuple[Tensor, Tensor, int, int]:
        types, coords = load_glyph(sample.ref)
        return (
            types[:max_len],
            coords[:max_len],
            sample.style_idx,
            sample.character_idx,
        )

    return transform


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
        max_len: int = 128,
        batch_size: int = 256,
        num_workers: int = 2,
        prefetch_factor: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.root = Path(root)
        self.max_seq_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.val_ratio = 0.05
        self.test_ratio = 0.05
        self.seed = 37414078

        codepoints = [ord("A") + i for i in range(26)]
        dataset = GlyphDataset(
            root=self.root,
            codepoints=codepoints,
            transform=_make_transform(self.max_seq_len),
        )
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.num_style_classes = len(dataset.style_classes)
        self.num_content_classes = len(dataset.character_classes)
        self.save_hyperparameters(
            {
                "dataset_len": self.dataset_len,
                "num_style_classes": self.num_style_classes,
                "num_content_classes": self.num_content_classes,
            },
        )
        g = torch.Generator().manual_seed(self.seed)

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [1 - self.val_ratio - self.test_ratio, self.val_ratio, self.test_ratio],
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
