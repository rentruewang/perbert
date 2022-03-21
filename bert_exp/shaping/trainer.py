# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from typing import Any

from omegaconf import DictConfig
from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar


class Trainer(PLTrainer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @classmethod
    def create(cls, cfg: DictConfig) -> Trainer:
        """
        A class method to create a Trainer from a dictionary of config created by hydra.

        Parameters
        ----------

        cfg: DictConfig
            A dictionary config object. In this case, cfg["trainer"] is the kwargs to the Trainer class.
        """

        trainer_cfg = cfg["trainer"]
        callback_cfg = cfg["callbacks"]

        callbacks = []
        if callback_cfg["rich"]:
            callbacks += [RichModelSummary, RichProgressBar]

        trainer = cls(**trainer_cfg)

        return trainer
