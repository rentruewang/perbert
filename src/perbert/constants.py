from __future__ import annotations

import os
from enum import Enum
from typing import Any


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        return str(self) == str(other)


class LightningStage(StrEnum):
    FIT = "fit"
    TEST = "test"


class Splits(StrEnum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


class CollatorType(StrEnum):
    TOKEN = "token"
    WHOLE_WORD = "wholeword"


class SchedulerAlgo(StrEnum):
    CONST = "constant"
    BERT = "bert"
    STEP = "step"


if (_cpus := os.cpu_count()) is not None:
    NUM_CPUS = _cpus
else:
    NUM_CPUS = 1
