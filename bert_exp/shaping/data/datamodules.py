# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import os
import typing
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple

import datasets
from datasets import DatasetDict
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from bert_exp.bert import AutoTokenizer, BatchEncoding, PreTrainedTokenizer
from bert_exp.constants import LightningStage

from .datasets import DatasetWrapper


class DataDir(NamedTuple):
    parent: str
    child: str

    def __str__(self) -> str:
        path = Path(self.parent) / Path(self.child)
        return str(path)

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
        path: str | DataDir = "wikitext/wikitext-103-v1",
        tokenizer: str | DataDir,
        use_fast: bool = True,
        max_length: int,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        self.path = DataDir.create(path)

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.max_length = max_length

        if num_workers > 0:
            self.num_workers = num_workers
        else:
            if (_cpus := os.cpu_count()) is not None:
                self.num_workers = _cpus
            else:
                self.num_workers = 1

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            tokenizer, use_fast=use_fast
        )

        self.datasets: Dict[str, Dataset] = {}

    def _prepare_data(self) -> DatasetDict:
        (parent, child) = self.path
        return typing.cast(DatasetDict, datasets.load_dataset(parent, child))

    def prepare_data(self) -> None:
        self._prepare_data()

    def setup(self, stage: str | LightningStage | None = None) -> None:
        dataset = self._prepare_data()

        if stage == LightningStage.FIT:
            self.datasets["train"] = DatasetWrapper(dataset["train"])
            self.datasets["validation"] = DatasetWrapper(dataset["validation"])

        if stage == LightningStage.TEST:
            self.datasets["test"] = DatasetWrapper(dataset["test"])

    def _tokenize_batch(self, batch: List[Dict[str, str]]) -> BatchEncoding:
        texts = [b["text"] for b in batch]
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    @property
    def _collate_fn(self) -> Callable[..., BatchEncoding]:
        return self._tokenize_batch

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    @classmethod
    def create(cls, cfg: DictConfig) -> WikiTextDataModule:
        dm_config = cfg["data"]
        return cls(**dm_config)
