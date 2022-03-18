from __future__ import annotations

from enum import Enum


class LightningStage(str, Enum):
    FIT = "fit"
    TEST = "test"

    def __eq__(self, other: LightningStage | str | None) -> bool:
        return other is None or str(self) == str(other)
