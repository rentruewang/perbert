# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from omegaconf import DictConfig
from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar


class Trainer(PLTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        trainer_cfg = cfg["trainer"]
        callback_cfg = cfg["callbacks"]

        callbacks = []
        if callback_cfg["rich"]:
            callbacks += [RichModelSummary(), RichProgressBar()]

        super().__init__(callbacks=callbacks, **trainer_cfg)
