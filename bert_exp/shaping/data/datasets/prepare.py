# pyright: reportPrivateImportUsage=false
import typing

import datasets
from datasets import DatasetDict
from omegaconf import DictConfig

from bert_exp.constants import Splits

from .mappers import TextMapper


def get(cfg: DictConfig) -> DatasetDict:
    data_cfg = cfg["data"]
    dicts = typing.cast(DatasetDict, datasets.load_dataset(**data_cfg["dataset"]))

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


def prepare(cfg: DictConfig) -> DatasetDict:
    data_dicts = get(cfg)

    data_dicts = TextMapper.map(cfg, data_dicts)

    return data_dicts
