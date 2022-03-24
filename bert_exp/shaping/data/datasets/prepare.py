# pyright: reportPrivateImportUsage=false
import datasets
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
import typing


def get(cfg: DictConfig) -> DatasetDict:
    dataset_cfg = cfg["data"]["dataset"]
    return typing.cast(DatasetDict, datasets.load_dataset(**dataset_cfg))
