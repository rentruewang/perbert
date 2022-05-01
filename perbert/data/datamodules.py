# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from typing import Any

import loguru
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BertConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
)

from perbert import constants
from perbert.constants import CollatorType, LightningStage, Splits

from . import datasets
from .collators import Collator, DecayCollator, HuggingfaceCollator
from .datasets import DatasetDictWrapper


class TextDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self.batch_size = self.cfg["data"]["batch"]["data"]
        self.pin_memory = self.cfg["data"]["pin_memory"]
        self.max_length = self.cfg["data"]["max_length"]

        if (num_workers := self.cfg["data"]["workers"]["dataloader"]) > 0:
            self.num_workers = num_workers
        else:
            self.num_workers = constants.NUM_CPUS

        self.feature = self.cfg["data"]["dataset"]["feature"]

        self.collator: Collator = self._init_collator()

        self.vocab_size: int = BertConfig().vocab_size
        self.datasets = {}

    def _prepare_data(self) -> DatasetDictWrapper:
        loguru.logger.debug("Prepare data called.")
        return datasets.prepare(self.cfg)

    def prepare_data(self) -> None:
        self._prepare_data()

    def setup(self, stage: str | LightningStage) -> None:
        loguru.logger.debug("Setting up stage: {}.", stage)
        self.datasets = self._prepare_data()

        stage = LightningStage(stage)

        if stage == LightningStage.FIT:
            assert Splits.TRAIN in self.datasets.keys(), self.datasets
            assert Splits.VALIDATION in self.datasets.keys(), self.datasets

        if stage == LightningStage.TEST:
            assert Splits.TEST in self.datasets.keys(), self.datasets

    def _init_collator(self) -> Collator:

        tokenizer = AutoTokenizer.from_pretrained(self.cfg["data"]["tokenizer"])
        mask_prob = self.cfg["model"]["lm"]["mask_prob"]

        collator_type = CollatorType(self.cfg["model"]["lm"]["collator"])

        if collator_type == CollatorType.Token:
            collator_cls = DataCollatorForLanguageModeling
        elif collator_type == CollatorType.WholeWord:
            collator_cls = DataCollatorForWholeWordMask
        else:
            raise ValueError(f"Collator type: {collator_type} not supported.")

        decay = self.cfg["model"]["lm"]["decay"]

        if decay == 1:
            return HuggingfaceCollator(
                klass=collator_cls,
                tokenizer=tokenizer,
                mask_prob=mask_prob,
                max_length=self.max_length,
            )
        elif 0 < decay < 1:
            return DecayCollator(
                klass=collator_cls,
                tokenizer=tokenizer,
                mask_prob=mask_prob,
                max_length=self.max_length,
                decay=decay,
            )
        else:
            raise ValueError(f"Decay rate: {decay} is invalid.")

    def _collate_hook(self, inputs: Any) -> Any:
        loguru.logger.trace(inputs)
        return self.collator(inputs)

    def _dataloader(self, split: Splits) -> DataLoader:
        dataset = self.datasets[split]
        loguru.logger.debug("Length of {} dataset: {}", split, len(dataset))

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_hook,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(Splits.TRAIN)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(Splits.VALIDATION)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(Splits.TEST)
