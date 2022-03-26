# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from abc import abstractclassmethod, abstractmethod
from ast import Call, Return
from asyncore import write
from lib2to3.pgen2.tokenize import tokenize
from typing import Any, Callable, Dict, Generic, List, Protocol, TypeVar

from numpy import ndarray
from omegaconf import DictConfig
from typing_extensions import Self

from bert_exp.bert import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    BatchEncoding,
)
import loguru


class Mappable(Protocol):
    @abstractmethod
    def filter(
        self,
        function: Callable[[Any], Any],
        *,
        batched: bool,
        batch_size: int,
        writer_batch_size: int,
        num_proc: int,
        desc: str,
    ) -> Self:
        ...

    @abstractmethod
    def map(
        self,
        function: Callable[[Any], Any],
        *,
        batched: bool,
        batch_size: int,
        writer_batch_size: int,
        num_proc: int,
        drop_last_batch: bool,
        desc: str,
    ) -> Self:
        ...


T = TypeVar("T", bound=Mappable)


class Mapper(Protocol[T]):
    @abstractmethod
    def __init__(self, cfg: DictConfig, mapper: T) -> None:
        ...

    @abstractmethod
    def __call__(self) -> T:
        ...

    @classmethod
    def map(cls, cfg: DictConfig, mapper: T) -> T:
        return cls(cfg, mapper)()


class TextMapper(Mapper[T]):
    def __init__(self, cfg: DictConfig, mapper: T) -> None:
        self.mapper = mapper

        data_cfg = cfg["data"]

        self.num_workers = data_cfg["workers"]["dataset"]
        self.line_by_line = data_cfg["line_by_line"]
        self.proc_batch = data_cfg["batch"]["proc"]

        self.padding = data_cfg["padding"]
        self.max_length = data_cfg["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(data_cfg["tokenizer"])

    def _filter_empty(self, entry: Dict[str, str]) -> bool:
        return len(entry["text"]) > 0

    def _tokenize(self, text: str, max_length: int | None = None) -> BatchEncoding:
        o = self.tokenizer(
            text,
            padding=self.padding,
            max_length=max_length,
            truncation=True,
            return_special_tokens_mask=True,
            return_tensors="np",
        )
        return o

    def _line_by_line(self, entries: Dict[str, str]) -> BatchEncoding:
        text = entries["text"]
        return self._tokenize(text, self.max_length)

    def _joined_lines(self, entries: Dict[str, List[str]]) -> Dict[str, List[ndarray]]:
        texts = entries["text"]
        joined_line = " ".join(texts)
        assert joined_line
        tokenized = self._tokenize(joined_line)
        length = self.max_length
        return {
            key: value[idx : idx + length]
            for (key, value) in tokenized.items()
            for idx in range(0, len(value), length)
        }

    def __call__(self) -> T:
        loguru.logger.info("Removing empty text sequences.")
        self.mapper = self.mapper.filter(
            self._filter_empty,
            batched=False,
            batch_size=1,
            writer_batch_size=self.proc_batch,
            num_proc=self.num_workers,
            desc="Removing empty text sequences.",
        )

        if self.line_by_line:
            loguru.logger.info("Tokenizing texts line by line.")
            self.mapper = self.mapper.map(
                self._line_by_line,
                batched=False,
                batch_size=1,
                writer_batch_size=self.proc_batch,
                num_proc=self.num_workers,
                drop_last_batch=False,
                desc="Line by line tokenization.",
            )
        else:
            loguru.logger.info("Tokenizing texts by joining lines together.")
            self.mapper = self.mapper.map(
                self._joined_lines,
                batched=True,
                batch_size=self.proc_batch,
                writer_batch_size=self.proc_batch,
                drop_last_batch=False,
                num_proc=self.num_workers,
                desc="Joined lines tokenization.",
            )

        return self.mapper
