from abc import abstractmethod
from typing import Any, Dict, List, Protocol, Sequence
from transformers import BatchEncoding


class Collator(Protocol):
    @abstractmethod
    def __call__(self, encodings: List[Dict[str, Any]]) -> Dict[str, Any]:
        ...
