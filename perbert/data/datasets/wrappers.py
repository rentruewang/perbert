from __future__ import annotations

import abc
from collections.abc import Sized
from typing import Generic, Protocol, TypeVar

import loguru
from datasets import DatasetDict
from torch.utils.data import Dataset

from perbert.constants import Splits

T = TypeVar("T")
"T is a invariant type."

K = TypeVar("K", contravariant=True)
"K is a contravariant type."

V = TypeVar("V", covariant=True)
"V is a covariant type."


class Indexible(Protocol[K, V]):
    "Indexible type implements both `__len__` and `__getitem__`"

    @abc.abstractmethod
    def __len__(self) -> int:
        "self.__len__() <==> len(self)"
        ...

    @abc.abstractmethod
    def __getitem__(self, key: K) -> V:
        "self.__getitem__(key) <==> self[key]"

        ...


class DatasetWrapper(Dataset, Sized, Generic[T]):
    "DatasetWrapper wraps creates a Dataset from an Indexible."

    def __init__(self, seq: Indexible[int, T]) -> None:
        """
        Initializes the DatasetWrapper with an Indexible whose key is of type integer.

        Parameters
        ----------

        seq: Indexible[int, T]
            The sequence object to wrap.
        """

        super().__init__()

        loguru.logger.debug("Wrapped: {}", seq)
        self._seq = seq

    def __len__(self) -> int:
        return len(self._seq)

    def __getitem__(self, key: int) -> T:
        return self._seq[key]


class DatasetDictWrapper(dict[str, T]):
    def __init__(self, dd: DatasetDict) -> None:
        super().__init__(dd)

        loguru.logger.debug("Wrapped: {}", dd)

        # Additional checks that all keys are provided.
        for split in Splits:
            if str(split) not in dd:
                raise KeyError("Key {} not found in DatasetDict.", split)

    def __getitem__(self, key: str | Splits) -> T:
        if isinstance(key, Splits):
            key = str(key)

        return super().__getitem__(key)
