# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import typing
from pathlib import Path

import datasets
from datasets import DatasetDict
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

from . import datasets


class WikiTextDataModule(LightningDataModule):
    def __init__(
        self, batch_size: int, data_dir: str | Path = "wikitext/wikitext-103-v1"
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.data_dir = Path(data_dir)

        self.prepare_data()

    def prepare_data(self):
        parent = str(self.data_dir.parent)
        child = str(self.data_dir.name)
        self.dataset = typing.cast(DatasetDict, datasets.load_dataset(parent, child))

    @property
    def train_dataset(self) -> Dataset:
        return datasets.wrap(self.dataset["train"])

    @property
    def test_dataset(self) -> Dataset:
        return datasets.wrap(self.dataset["test"])

    @property
    def validation_dataset(self) -> Dataset:
        return datasets.wrap(self.dataset["validation"])

    def setup(self, stage: str | None = None) -> None:
        return super().setup(stage)
