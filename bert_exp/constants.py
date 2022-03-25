from __future__ import annotations

import os
from enum import Enum


class LightningStage(str, Enum):
    FIT = "fit"
    TEST = "test"

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: LightningStage | str | None) -> bool:
        return (other is None) or (str(self) == str(other))


class Splits(str, Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: Splits | str) -> bool:
        return str(self) == str(other)


if (_cpus := os.cpu_count()) is not None:
    NUM_CPUS = _cpus
else:
    NUM_CPUS = 1
