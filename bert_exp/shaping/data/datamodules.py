# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import random
import typing
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple

import datasets
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


class MaskType(str, Enum):
    Token = "token"
    Attention = "attention"


class TextDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        data_cfg = cfg["data"]
        lm_cfg = data_cfg["lm"]

        self.path = DataDir.create(data_cfg["path"])

        self.batch_size = data_cfg["batch_size"]
        self.pin_memory = data_cfg["pin_memory"]
        self.max_length = data_cfg["max_length"]

        if (num_workers := data_cfg["num_workers"]) > 0:
            self.num_workers = num_workers
        else:
            self.num_workers = constants.NUM_CPUS

        self.proc_batch = data_cfg["proc_batch"]
        self.line_by_line = data_cfg["line_by_line"]

        self.ignore = lm_cfg["ignore"]

        self.mask_prob = lm_cfg["mask"]
        self.random = lm_cfg["random"]
        self.unchanged = lm_cfg["unchanged"]
        assert 0 <= self.random + self.unchanged <= 1, [self.random, self.unchanged]

        self.mask_type = MaskType(lm_cfg["mask_type"])

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            data_cfg["tokenizer"], use_fast=data_cfg["use_fast"]
        )

        self.vocab_size: int = Config().vocab_size

        self.datasets: Dict[str, DatasetWrapper] = {}

    def _prepare_data(self) -> DatasetDict:
        (parent, child) = self.path
        return typing.cast(DatasetDict, datasets.load_dataset(parent, child))

    def prepare_data(self) -> None:
        self._prepare_data()

    def setup(self, stage: str | LightningStage | None = None) -> None:
        dataset = self._prepare_data()

        if stage == LightningStage.FIT:
            train_dataset = self.preprocess(dataset["train"])
            validation_dataset = self.preprocess(dataset["validation"])

            self.datasets["train"] = DatasetWrapper(train_dataset)
            self.datasets["validation"] = DatasetWrapper(validation_dataset)

        if stage == LightningStage.TEST:
            test_dataset = self.preprocess(dataset["test"])

            self.datasets["test"] = DatasetWrapper(test_dataset)

    def _filter_empty(self, entry: Dict[str, str]) -> bool:
        return len(entry["text"]) > 0

    def _line_by_line(self, entries: Dict[str, List[str]]) -> Dict[str, List[str]]:
        texts = entries["text"]
        selected_line = random.choice(texts)
        assert selected_line
        return {"text": [selected_line]}

    def _joined_lines(self, entries: Dict[str, List[str]]) -> Dict[str, List[str]]:
        texts = entries["text"]
        joined_line = " ".join(texts)
        assert joined_line
        return {"text": [joined_line]}

    def _encode(self, entry: Dict[str, str]) -> Dict[str, List[int]]:
        text = entry["text"]

        tokenized = self.tokenizer.encode(text)

        return {"text": tokenized}

    def _trunc_pad(self, entry: Dict[str, List[int]]) -> Dict[str, ndarray]:
        tokens = entry["text"]
        length = len(tokens)
        maximum = self.max_length
        result = np.array(tokens)

        if length > maximum:
            random_head = random.randrange(length - maximum)
            result = result[random_head : random_head + maximum]
        elif length < maximum:
            result = np.pad(
                result,
                pad_width=(0, maximum - len(tokens)),
                constant_values=self.ignore,
            )

        assert result.shape == (maximum,), result.shape

        return {"data": result}

    def _mask_tokens(self, entry: Dict[str, ndarray]) -> Dict[str, ndarray]:
        data = np.array(entry["data"])

        assert np.all(data < self.vocab_size), data > self.vocab_size

        # Here, 1 is masked, 0 is masked.
        mask = np_random.random(size=data.shape) < self.mask_prob
        random_assign = np_random.random(size=data.shape) < self.random
        randint = np_random.randint(low=0, high=self.vocab_size, size=data.shape)
        unchanged = np_random.random(size=data.shape) < self.unchanged

        ignore = np.full_like(data, self.ignore)
        labels = np.where(mask, ignore, data)

        # Unchanged nullifies masks.
        mask = mask & ~unchanged

        # If not random assign, random_num = 0.
        random_num = randint * random_assign.astype("int")

        if self.mask_type == MaskType.Token:
            # E(random_num) == E(random_num) % max for uniform distribution in the range [0, max).
            data = (data + random_num) % self.vocab_size
            mask_tokens = np.full_like(
                data, fill_value=self.tokenizer.mask_token_id, dtype=int
            )
            data = np.where(mask, mask_tokens, data)
            attn = np.zeros_like(data)
        elif self.mask_type == MaskType.Attention:
            # For attention, 1 is not masked, 0 is masked.
            attn = ~mask
        else:
            raise ValueError(f"Mask type: {self.mask_type} is not supported.")

        return {"input_ids": data, "attention_mask": attn, "labels": labels}

    def _enc_trunc_pad_mask(self, entry: Dict[str, str]) -> Dict[str, ndarray]:
        encoded = self._encode(entry)
        unified_length = self._trunc_pad(encoded)
        masked = self._mask_tokens(unified_length)
        return masked

    def preprocess(self, dataset: Dataset) -> Dataset:
        loguru.logger.info("Filtering empty sequences.")
        dataset = dataset.filter(self._filter_empty, batched=False)

        if self.line_by_line:
            loguru.logger.info("Tokenizing line by line.")
            function = self._line_by_line
        else:
            loguru.logger.info("Tokenizing by joining lines.")
            function = self._joined_lines
        dataset = dataset.map(
            function=function,
            batched=True,
            batch_size=self.proc_batch,
            num_proc=self.num_workers,
        )

        loguru.logger.info(
            "Encode, truncate / pad to desired legnth, and mask sequences."
        )
        dataset = dataset.map(
            function=self._enc_trunc_pad_mask, num_proc=self.num_workers
        )
        return dataset

    def _map_to_batch_enc(self, entries: List[Dict[str, List[int]]]) -> BatchEncoding:
        keys = {"input_ids", "attention_mask", "labels"}

        return BatchEncoding(
            {key: torch.tensor([entry[key] for entry in entries]) for key in keys}
        )

    @property
    def collate_fn(self) -> Callable[[Any], BatchEncoding]:
        return self._map_to_batch_enc

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.datasets["train"]
        loguru.logger.debug(f"Length of train dataset: {len(train_dataset)}")

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        test_dataset = self.datasets["test"]
        loguru.logger.debug(f"Length of test dataset: {len(test_dataset)}")

        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        val_dataset = self.datasets["validation"]
        loguru.logger.debug(f"Length of validation dataset: {len(val_dataset)}")

        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )
