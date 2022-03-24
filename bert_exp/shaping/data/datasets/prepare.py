# pyright: reportPrivateImportUsage=false
import typing

import datasets
from datasets import DatasetDict
from omegaconf import DictConfig


def get(cfg: DictConfig) -> DatasetDict:
    dataset_cfg = cfg["data"]["dataset"]
    return typing.cast(DatasetDict, datasets.load_dataset(**dataset_cfg))
