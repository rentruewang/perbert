from __future__ import annotations

import typing
from collections.abc import Mapping
from typing import Any

from aim.pytorch_lightning import AimLogger
from omegaconf import DictConfig
from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning.callbacks import (
    Callback,
    DeviceStatsMonitor,
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger


class Trainer(PLTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        super().__init__(
            callbacks=self.__callbacks,
            logger=self.__loggers,
            **self.cfg["trainer"],
        )

    @property
    def __callbacks(self) -> list[Callback]:
        callback_cfg = self.cfg["callbacks"]

        callbacks = []

        if monitor := callback_cfg["early_stopping"]:
            callbacks.append(EarlyStopping(monitor=monitor))

        if callback_cfg["device_stats"]:
            callbacks.append(DeviceStatsMonitor())

        if ckpt := callback_cfg["checkpoint"]:
            callbacks.append(ModelCheckpoint(ckpt))

        if callback_cfg["rich"]:
            callbacks.extend([RichModelSummary(), RichProgressBar()])

        return callbacks

    @property
    def __loggers(self) -> list[Logger]:
        logger_cfg = self.cfg["loggers"]

        loggers = []

        if path := logger_cfg["tensorboard"]:
            if isinstance(path, str):
                tb_logger = TensorBoardLogger(save_dir=path)
            elif isinstance(path, DictConfig):
                path = typing.cast(Mapping[str, Any], path)
                tb_logger = TensorBoardLogger(**path)
            else:
                tb_logger = TensorBoardLogger(save_dir="tensorboard")
            loggers.append(tb_logger)

        if path := logger_cfg["wandb"]:
            if isinstance(path, str):
                wandb_logger = WandbLogger(name=path)
            elif isinstance(path, DictConfig):
                path = typing.cast(Mapping[str, Any], path)
                wandb_logger = WandbLogger(**path)
            else:
                wandb_logger = WandbLogger()
            loggers.append(wandb_logger)

        if path := logger_cfg["aim"]:
            if isinstance(path, DictConfig):
                path = typing.cast(Mapping[str, Any], path)
                aim_logger = AimLogger(**path)
            else:
                aim_logger = AimLogger()
            loggers.append(aim_logger)

        return loggers
