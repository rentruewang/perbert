from __future__ import annotations

from perbert.constants import SchedulerAlgo


class LengthScheduler:
    def __init__(
        self,
        max_length: int,
        total_steps: int,
        min_length: int = 128,
        algo: str | SchedulerAlgo = SchedulerAlgo.BERT,
        step_interval: int = 10000,
        step_size: float = 2.0,
    ) -> None:
        self.steps = -1

        self.max_length = max_length
        self.total_steps = total_steps
        self.min_length = min_length
        self.algo = SchedulerAlgo(algo)
        self.interval = step_interval
        self.step_size = step_size

    def bert_schedule(self) -> int:
        if self.steps / self.total_steps < 0.9:
            return self.min_length
        else:
            return self.max_length

    def step_schedule(self) -> int:
        return min(
            self.max_length,
            int(self.min_length * (self.step_size ** (self.steps // self.interval))),
        )

    def step(self) -> int:
        self.steps += 1

        if self.algo == SchedulerAlgo.CONST:
            return self.max_length
        elif self.algo == SchedulerAlgo.BERT:
            return self.bert_schedule()
        elif self.algo == SchedulerAlgo.STEP:
            return self.step_schedule()
        else:
            raise NotImplementedError
