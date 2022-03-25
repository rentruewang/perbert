# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import random
import typing
from enum import Enum
from pathlib import Path
from re import S
from select import select
from typing import Any, Callable, Dict, List, NamedTuple

import datasets as arrow_datasets
import loguru
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from numpy import ndarray
from numpy import random as np_random
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from bert_exp import (
    AutoTokenizer,
    BatchEncoding,
    Config,
    LightningStage,
    PreTrainedTokenizer,
    constants,
)

from . import datasets
from .datasets import DatasetWrapper


class MaskType(str, Enum):
    Token = "token"
    Attention = "attention"


class TextDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg
        data_cfg = self.cfg["data"]

        self.batch_size = data_cfg["batch_size"]
        self.pin_memory = data_cfg["pin_memory"]
        self.max_length = data_cfg["max_length"]

        if (num_workers := data_cfg["num_workers"]) > 0:
            self.num_workers = num_workers
        else:
            self.num_workers = constants.NUM_CPUS

        self.proc_batch = data_cfg["proc_batch"]
        self.line_by_line = data_cfg["line_by_line"]

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            data_cfg["tokenizer"], use_fast=data_cfg["use_fast"]
        )

        self.vocab_size: int = Config().vocab_size

        self.datasets = {}

    def _prepare_data(self) -> DatasetDict:
        return datasets.prepare(self.cfg)

    def prepare_data(self) -> None:
        self._prepare_data()

    def setup(self, stage: str | LightningStage | None = None) -> None:
        self.datasets = self._prepare_data()

        if stage == "fit":
            assert "train" in self.datasets.keys(), self.datasets
            assert "validation" in self.datasets.keys(), self.datasets

        if stage == "test":
            assert "test" in self.datasets.keys(), self.datasets

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.datasets["train"]
        loguru.logger.debug(f"Length of train dataset: {len(train_dataset)}")

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        test_dataset = self.datasets["test"]
        loguru.logger.debug(f"Length of test dataset: {len(test_dataset)}")

        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        val_dataset = self.datasets["validation"]
        loguru.logger.debug(f"Length of validation dataset: {len(val_dataset)}")

        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
