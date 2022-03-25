# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Dict, Generic, List, Protocol, TypeVar

from numpy import ndarray
from omegaconf import DictConfig
from transformers import AutoTokenizer
from typing_extensions import Self


class Mappable(Protocol):
    @abstractmethod
    def map(
        self,
        function: Callable[..., Any],
        *,
        batched: bool,
        batch_size: int,
        num_proc: int,
        drop_last_batch: bool
    ) -> Self:
        ...


T = TypeVar("T", bound=Mappable)


class TextMapper(Generic[T]):
    def __init__(self, cfg: DictConfig, ds: T) -> None:
        data_cfg = cfg["data"]

        self.num_workers = data_cfg["workers"]["dataset"]
        self.line_by_line = data_cfg["line_by_line"]
        self.proc_batch = data_cfg["proc_batch"]

        self.padding = data_cfg["padding"]
        self.max_length = data_cfg["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(data_cfg["tokenizer"])

        self.ds = ds

    def _filter_empty(self, entry: Dict[str, str]) -> bool:
        return len(entry["text"]) > 0

    def _encode(self, entry: Dict[str, str]) -> Dict[str, List[int]]:
        text = entry["text"]

        tokenized = self.tokenizer.encode(text)

        return {"text": tokenized}

    def _tokenize(self, text: str) -> ndarray:
        return self.tokenizer(
            text,
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
            return_special_tokens_mask=True,
            return_tensors="np",
        )

    def _line_by_line(self, entries: Dict[str, str]) -> Dict[str, ndarray]:
        text = entries["text"]
        tokenized = self._tokenize(text)
        return {"text": tokenized}

    def _joined_lines(self, entries: Dict[str, List[str]]) -> Dict[str, List[ndarray]]:
        texts = entries["text"]
        joined_line = " ".join(texts)
        assert joined_line
        tokenized = self._tokenize(joined_line)
        length = self.max_length

        return {
            "text": [
                tokenized[idx : idx + length]
                for idx in range(0, len(tokenized), length)
            ]
        }

    def __call__(self) -> T:
        if self.line_by_line:
            return self.ds.map(
                self._line_by_line,
                batched=False,
                batch_size=1,
                num_proc=self.num_workers,
                drop_last_batch=False,
            )
        else:
            return self.ds.map(
                self._joined_lines,
                batched=True,
                batch_size=self.proc_batch,
                drop_last_batch=False,
                num_proc=self.num_workers,
            )

    @classmethod
    def map(cls, cfg: DictConfig, ds: T) -> T:
        return cls(cfg, ds)()
