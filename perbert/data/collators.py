from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Protocol, Type

from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class Collator(Protocol):
    @abstractmethod
    def __call__(self, encodings: List[Dict[str, Any]]) -> Dict[str, Any]:
        ...


class HuggingfaceCollator(Collator):
    def __init__(
        self,
        klass: Type[DataCollatorForLanguageModeling | DataCollatorForWholeWordMask],
        *,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        mask_prob: float,
        max_length: int,
    ) -> None:
        self.klass = klass
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.max_length = max_length

        if not 0 <= mask_prob <= 1:
            raise ValueError(f"MLM probability: {mask_prob} should be between 0 and 1")

    def _get_collator(
        self,
    ) -> DataCollatorForLanguageModeling | DataCollatorForWholeWordMask:
        return self.klass(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mask_prob,
            pad_to_multiple_of=self.max_length,
            return_tensors="pt",
        )

    def __call__(self, encodings: List[Dict[str, Any]]) -> Dict[str, Any]:
        collator = self._get_collator()
        return collator(encodings)


class DecayCollator(HuggingfaceCollator):
    def __init__(
        self,
        klass: Type[DataCollatorForLanguageModeling | DataCollatorForWholeWordMask],
        *,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        mask_prob: float,
        max_length: int,
        decay: float,
    ) -> None:
        super().__init__(
            klass=klass, tokenizer=tokenizer, mask_prob=mask_prob, max_length=max_length
        )

        if not 0 < decay < 1:
            raise ValueError(f"Decay base: {decay} should be between 0 and 1.")

        self.base = decay

        self._eventual_mask_prob = self.mask_prob
        self.mask_prob = 0

    def __call__(self, encodings: List[Dict[str, Any]]) -> Dict[str, Any]:
        prob_diff = self._eventual_mask_prob - self.mask_prob
        self.mask_prob = self._eventual_mask_prob - prob_diff * self.base
        return super().__call__(encodings=encodings)
