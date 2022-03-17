from abc import abstractmethod
from typing import Generic, Protocol, TypeVar

from torch.utils.data import Dataset

# TODO: documentation

T = TypeVar("T")
K = TypeVar("K", contravariant=True)
V = TypeVar("V", covariant=True)


class Indexable(Protocol[K, V]):
    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, key: K) -> V:
        ...


class _GeneratedDataset(Dataset, Generic[T]):
    def __init__(self, seq: Indexable[int, T]) -> None:
        super().__init__()

        self._seq = seq

    def __len__(self) -> int:
        return len(self._seq)

    def __getitem__(self, key: int) -> T:
        return self._seq[key]


def wrap(seq: Indexable[int, T]) -> Dataset[T]:
    return _GeneratedDataset[T](seq)
