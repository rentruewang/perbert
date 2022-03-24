# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from typing import List

import loguru
from omegaconf import DictConfig
from pl_bolts.callbacks import ORTCallback, PrintTableMetricsCallback
from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning.callbacks import (
    Callback,
    DeviceStatsMonitor,
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)


class Trainer(PLTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        trainer_cfg = self.cfg["trainer"]

        super().__init__(callbacks=self._callbacks, **trainer_cfg)

    @property
    def _callbacks(self) -> List[Callback]:
        callback_cfg = self.cfg["callbacks"]

        callbacks = []

        if callback_cfg["early_stopping"]:
            callbacks.append(EarlyStopping())

        if callback_cfg["device_stats"]:
            callbacks.append(DeviceStatsMonitor())

        if ckpt := callback_cfg["checkpoint"]:
            callbacks.append(ModelCheckpoint(ckpt))

        if callback_cfg["onnx"]:
            callbacks.append(ORTCallback())

        if callback_cfg["rich"]:
            callbacks.extend([RichModelSummary(), RichProgressBar()])

        if callback_cfg["table_metrics"]:
            callbacks.append(PrintTableMetricsCallback())

        return callbacks
