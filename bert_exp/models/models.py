# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from enum import Enum
from typing import Any

import loguru
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor, no_grad
from torch.optim import Adam, AdamW, Optimizer

from bert_exp.bert import BatchEncoding, Config, ForMaskedLM, Output

from . import init


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
            lm.apply(init.bert_init(bert_config))

        if not isinstance(lm, ForMaskedLM):
            raise ValueError("Model name: {} is invalid.", model_name)

        self.lm = lm
        self.config = cfg

        loguru.logger.debug("Model used: {}", self.lm)

    @property
    def bert_config(self) -> Config:
        return self.lm.bert.config

    def forward(self, **kwargs: Any) -> Output:
        return self.lm(**kwargs)

    def _step(self, batch: BatchEncoding, batch_idx: int, name: str) -> Tensor:
        loguru.logger.trace("{} step batch: {}", name, batch_idx)

        output: Output = self(**batch)

        loss = output.loss
        assert isinstance(loss, Tensor)
        self.log(name=f"{name}/loss", value=loss.item())
        return loss

    def training_step(self, batch: BatchEncoding, batch_idx: int) -> Tensor:
        return self._step(batch, batch_idx=batch_idx, name="train")

    @no_grad()
    def test_step(self, batch: BatchEncoding, batch_idx: int) -> Tensor:
        return self._step(batch, batch_idx=batch_idx, name="test")

    @no_grad()
    def validation_step(self, batch: BatchEncoding, batch_idx: int) -> Tensor:
        return self._step(batch, batch_idx=batch_idx, name="validation")

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

        optimizer = optim_cls(params=self.parameters(), lr=lr)
        loguru.logger.info("Optimizer: {}", optimizer)
        return optimizer
