# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from typing import Any

import loguru
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertConfig, DataCollatorForLanguageModeling

from perbert import constants
from perbert.constants import LightningStage, Splits

from . import datasets
from .datasets import DatasetDictWrapper


class TextDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg
        data_cfg = self.cfg["data"]

        self.batch_size = data_cfg["batch"]["data"]
        self.pin_memory = data_cfg["pin_memory"]
        self.max_length = data_cfg["max_length"]

        if (num_workers := data_cfg["workers"]["dataloader"]) > 0:
            self.num_workers = num_workers
        else:
            self.num_workers = constants.NUM_CPUS

        self.feature = data_cfg["dataset"]["feature"]

        tokenizer = AutoTokenizer.from_pretrained(data_cfg["tokenizer"])
        lm_cfg = cfg["model"]["lm"]
        mask_prob = lm_cfg["mask_prob"]
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mask_prob,
            pad_to_multiple_of=self.max_length,
        )

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

    def _collate_hook(self, inputs: Any) -> Any:
        loguru.logger.trace(inputs)
        return self.collator(inputs, return_tensors="pt")

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
