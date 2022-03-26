# pyright: reportPrivateImportUsage=false
import typing

import datasets
import loguru
from datasets import DatasetDict
from omegaconf import DictConfig

from bert_exp.constants import Splits

from .mappers import TextMapper
from .wrappers import DatasetDictWrapper


def get(cfg: DictConfig) -> DatasetDict:
    loguru.logger.info("Fetching the dataset.")

    data_cfg = cfg["data"]
    dicts = typing.cast(
        DatasetDict, datasets.load_dataset(**data_cfg["dataset"]["args"])
    )

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
    data_cfg = cfg["data"]

    loguru.logger.info("Preprocessing: tokenizing and truncating / padding the lines.")
    data_dicts = TextMapper.map(cfg, data_dicts)

    if location := data_cfg.get("save_to_disk", None):
        loguru.logger.info("Saving dataset to disk.")
        data_dicts.save_to_disk(location)

    return data_dicts


def prepare(cfg: DictConfig) -> DatasetDictWrapper:
    dataset_cfg = cfg["data"]["dataset"]
    if location := dataset_cfg.get("load_from_disk", None):
        loguru.logger.info("Loading dataset from disk.")
        data_dicts = typing.cast(DatasetDict, datasets.load_from_disk(location))
    else:
        loguru.logger.info("Preparing datasets")
        data_dicts = get(cfg)
        data_dicts = process(cfg, data_dicts)

    return DatasetDictWrapper(data_dicts)
