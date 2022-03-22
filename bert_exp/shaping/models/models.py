# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from enum import Enum
from typing import Any, Dict

import loguru
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor, no_grad
from torch.optim import Adam, AdamW, Optimizer

from bert_exp import BatchEncoding, Config, ForMaskedLM, Output

from .init import bert_init


class OptimizerType(str, Enum):
    Adam = "adam"
    AdamW = "adamw"


class Model(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        model_cfg = cfg["model"]
        model_name = model_cfg["path"]

        if model_name is not None:
            lm = ForMaskedLM.from_pretrained(model_name)
        else:
            bert_config = Config()
            lm = ForMaskedLM(bert_config)
            lm.apply(bert_init(bert_config))

        if not isinstance(lm, ForMaskedLM):
            raise ValueError(f"Model name: {model_name} is invalid.")

        self.lm = lm
        self.config = cfg

    @property
    def bert_config(self) -> Config:
        return self.lm.bert.config

    def forward(self, **kwargs: Any) -> Output:
        return self.lm(**kwargs)

    def training_step(self, batch: BatchEncoding, batch_idx: int) -> Dict[str, Tensor]:
        loguru.logger.trace(f"Training step batch: {batch_idx}")

        output: Output = self(**batch)

        assert isinstance(output.loss, Tensor)
        return {"loss": output.loss}

    @no_grad()
    def validation_step(
        self, batch: BatchEncoding, batch_idx: int
    ) -> Dict[str, Tensor]:
        loguru.logger.trace(f"Validation step batch: {batch_idx}")

        output: Output = self(**batch)

        assert isinstance(output.loss, Tensor)
        return {"val_loss": output.loss}

    @no_grad()
    def test_step(self, batch: BatchEncoding, batch_idx: int) -> Dict[str, Tensor]:
        loguru.logger.trace(f"Test step batch: {batch_idx}")

        output: Output = self(**batch)

        assert isinstance(output.loss, Tensor)
        return {"test_loss": output.loss}

    def configure_optimizers(self) -> Optimizer:
        model_cfg = self.config["model"]
        optim_type = OptimizerType(model_cfg["optimizer"])
        lr = model_cfg["lr"]

        if optim_type == OptimizerType.Adam:
            optim_cls = Adam
        elif optim_type == OptimizerType.AdamW:
            optim_cls = AdamW
        else:
            raise ValueError(f"Optimizer type: {optim_type} not supported.")

        return optim_cls(params=self.parameters(), lr=lr)
