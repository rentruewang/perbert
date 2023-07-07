from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Dict, List, Protocol, TypeVar

import loguru
import numpy as np
from numpy import ndarray
from omegaconf import DictConfig
from transformers import AutoTokenizer, BatchEncoding
from typing_extensions import Self


class Mappable(Protocol):
    @abstractmethod
    def filter(
        self,
        function: Callable[[Any], Any],
        *,
        batched: bool = False,
        batch_size: int = 1000,
        writer_batch_size: int = 1000,
        num_proc: int | None = None,
        load_from_cache_file: bool = False,
        desc: str | None = None,
    ) -> Self:
        ...

    @abstractmethod
    def map(
        self,
        function: Callable[[Any], Any],
        *,
        batched: bool = False,
        batch_size: int = 1000,
        writer_batch_size: int = 1000,
        num_proc: int | None = None,
        load_from_cache_file: bool = False,
        remove_columns: List[str] | None = None,
        desc: str | None = None,
    ) -> Self:
        ...

    @property
    @abstractmethod
    def column_names(self) -> List[str] | Dict[str, List[str]]:
        ...

    @abstractmethod
    def remove_columns(self, column_names: str | List[str]) -> Self:
        ...


def _flat_column_names(mapper: Mappable) -> List[str]:
    columns = mapper.column_names

    if isinstance(columns, List):
        return columns

    columns = sum(columns.values(), [])
    columns = set(columns)
    columns = list(columns)

    return columns


T = TypeVar("T", bound=Mappable)


class Mapper(Protocol[T]):
    @abstractmethod
    def __init__(self, cfg: DictConfig, mapper: T) -> None:
        ...

    @abstractmethod
    def apply(self) -> T:
        ...

    @classmethod
    def map(cls, cfg: DictConfig, mapper: T) -> T:
        return cls(cfg, mapper).apply()


class TextMapper(Mapper[T]):
    def __init__(self, cfg: DictConfig, mapper: T) -> None:
        self.mapper = mapper
        self.init_columns = _flat_column_names(mapper)

        data_cfg = cfg["data"]

        self.num_workers = data_cfg["workers"]["dataset"]
        self.line_by_line = data_cfg["line_by_line"]
        self.proc_batch = data_cfg["batch"]["proc"]

        self.padding = data_cfg["padding"]
        self.max_length = data_cfg["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(data_cfg["tokenizer"])

    def _filter_empty(self, entry: Dict[str, str]) -> bool:
        return len(entry["text"]) > 0

    def _line_by_line(self, entries: Dict[str, str]) -> BatchEncoding:
        return self.tokenizer(
            entries["text"],
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
            return_special_tokens_mask=True,
            return_tensors="np",
        )

    def _joined_lines(self, entries: Dict[str, List[str]]) -> Dict[str, List[ndarray]]:
        joined_line = " ".join(entries["text"])

        tokenized = self.tokenizer(
            joined_line,
            return_special_tokens_mask=True,
            return_tensors="np",
        )

        tokenized = {key: np.squeeze(value, 0) for (key, value) in tokenized.items()}
        length = self.max_length

        return {
            key: [
                value[idx : idx + length]
                for idx in range(0, len(value) - length, length)
            ]
            for (key, value) in tokenized.items()
        }

    def apply(self) -> T:
        mapper = self.mapper

        loguru.logger.info("Removing empty text sequences.")
        mapper = mapper.filter(
            self._filter_empty,
            batched=False,
            batch_size=1,
            writer_batch_size=self.proc_batch,
            num_proc=self.num_workers,
            load_from_cache_file=True,
            desc="Removing empty text sequences.",
        )
        loguru.logger.debug(mapper)

        if self.line_by_line:
            loguru.logger.info("Tokenizing texts line by line.")
            mapper = mapper.map(
                self._line_by_line,
                batched=True,
                batch_size=1,
                writer_batch_size=self.proc_batch,
                num_proc=self.num_workers,
                remove_columns=self.init_columns,
                load_from_cache_file=True,
                desc="Line by line tokenization.",
            )
        else:
            loguru.logger.info("Tokenizing texts by joining lines together.")
            mapper = mapper.map(
                self._joined_lines,
                batched=True,
                batch_size=self.proc_batch,
                writer_batch_size=self.proc_batch,
                num_proc=self.num_workers,
                remove_columns=self.init_columns,
                load_from_cache_file=True,
                desc="Joined lines tokenization.",
            )

        loguru.logger.debug(mapper)
        return mapper
