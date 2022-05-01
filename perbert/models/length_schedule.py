from __future__ import annotations

import math


class LengthScheduler:
    def __init__(
        self,
        max_length: int,
        total_steps: int,
        min_length: int = 128,
        algo: str = "bert",
        step_interval: int = 10000,
        step_size: float = 2.0,
    ) -> None:
        self.steps = -1

        self.max_length = max_length
        self._eventual_length = self.max_length
        self.total_steps = total_steps
        self.min_length = min_length
        self.algo = algo
        self.interval = step_interval
        self.step_size = step_size

    def bert_schedule(self) -> None:
        if self.steps / self.total_steps < 0.9:
            self.max_length = self.min_length
        else:
            self.max_length = self._eventual_length

    def step_schedule(self) -> None:
        self.max_length = min(
            self._eventual_length,
            int(
                self.min_length
                * math.pow(self.step_size, math.floor(self.steps / self.interval))
            ),
        )

    def step(self) -> None:
        self.steps += 1
        if self.algo == "constant":
            pass
        elif self.algo == "bert":
            self.bert_schedule()
        elif self.algo == "step":
            self.step_schedule()
        else:
            raise NotImplementedError
        return self.max_length
