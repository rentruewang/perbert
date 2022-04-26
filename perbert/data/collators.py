from abc import abstractmethod
from typing import Any, Dict, List, Protocol


class Collator(Protocol):
    @abstractmethod
    def __call__(self, encodings: List[Dict[str, Any]]) -> Dict[str, Any]:
        ...


class WrappedCollator(Collator):
    def __init__(self, wrapped: Any) -> None:
        self.wrapped = wrapped

    def __call__(self, encodings: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.wrapped(encodings)
