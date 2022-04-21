# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import typing
from pathlib import Path

import datasets
import loguru
from bert_exp.constants import Splits
from datasets import DatasetDict
from omegaconf import DictConfig

from .mappers import TextMapper
from .wrappers import DatasetDictWrapper


def load(cfg: DictConfig) -> DatasetDict:
    loguru.logger.info("Fetching the dataset.")

    data_cfg = cfg["data"]

    args = data_cfg["dataset"]["args"]
    loguru.logger.debug("Arguments used: {}", args)
    dicts = typing.cast(DatasetDict, datasets.load_dataset(**args))

    if all(str(key) in dicts for key in Splits):
        return dicts

    assert "train" in dicts, dicts

    train_dataset = dicts["train"]

    splits_ratio = data_cfg["splits"]
    test_ratio = splits_ratio["test"]
    val_ratio = splits_ratio["validation"]
    train_ratio = 1 - test_ratio - val_ratio
    assert train_ratio > 0, [train_ratio, test_ratio, val_ratio]

    ds = train_dataset.train_test_split(test_size=test_ratio)
    train_ds = ds["train"]
    train_val = train_ds.train_test_split(
        test_size=val_ratio / (train_ratio + val_ratio)
    )

    return DatasetDict(
        {
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": ds["test"],
        }
    )


def process(cfg: DictConfig, data_dicts: DatasetDict) -> DatasetDict:
    loguru.logger.info("Preprocessing: tokenizing and truncating / padding the lines.")

    data_dicts = TextMapper.map(cfg, data_dicts)

    return data_dicts


def _location(cfg: DictConfig) -> str | None:
    return cfg["data"]["dataset"]["save_path"]


def _prepare(cfg: DictConfig) -> DatasetDict:
    if (location := _location(cfg)) and Path(location).exists():
        loguru.logger.info("Loading dataset from {}.", location)
        return typing.cast(DatasetDict, datasets.load_from_disk(location))
    else:
        loguru.logger.info("Preparing datasets.")
        data_dicts = load(cfg)
        return process(cfg, data_dicts)


def _save_if_path(cfg: DictConfig, data_dicts: DatasetDict) -> None:
    if (location := _location(cfg)) and not Path(location).exists():
        loguru.logger.info("Saving dataset to {}.", location)
        data_dicts.save_to_disk(location)


def prepare(cfg: DictConfig) -> DatasetDictWrapper:
    data_dicts = _prepare(cfg)
    _save_if_path(cfg, data_dicts)
    return DatasetDictWrapper(data_dicts)
