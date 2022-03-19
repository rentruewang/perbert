# pyright: reportPrivateImportUsage=false
from __future__ import annotations


import typing
from typing import Dict, NamedTuple

import datasets
from datasets import DatasetDict
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from bert_exp.constants import LightningStage
from bert_exp import constants
from torch.utils.data import Dataset
from .datasets import DatasetWrapper


class DataDir(NamedTuple):
    parent: str
    child: str

    @classmethod
    def create(cls, path: str | DataDir) -> DataDir:
        if isinstance(path, str):
            (parent, child) = path.split("/")
            return DataDir(parent, child)
        return path


class WikiTextDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        data_dir: str | DataDir = "wikitext/wikitext-103-v1",
        batch_size: int,
        num_workers: int = constants.NUM_CPUS
    ) -> None:
        super().__init__()

        self.data_dir = DataDir.create(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.datasets: Dict[str, Dataset] = {}

    def _prepare_data(self):
        (parent, child) = self.data_dir
        return typing.cast(DatasetDict, datasets.load_dataset(parent, child))

    def prepare_data(self) -> None:
        self._prepare_data()

    def setup(self, stage: str | None = None) -> None:
        dataset = self._prepare_data()

        if stage == LightningStage.FIT:
            self.datasets["train"] = DatasetWrapper(dataset["train"])
            self.datasets["validation"] = DatasetWrapper(dataset["validation"])

        if stage == LightningStage.TEST:
            self.datasets["test"] = DatasetWrapper(dataset["test"])

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
