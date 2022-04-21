# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import typing
from enum import Enum
from typing import Any, Type

import loguru
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor, no_grad
from torch.nn import Module
from torch.optim import Adam, AdamW, Optimizer
from transformers import BatchEncoding, BertConfig, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertOutput, BertPreTrainedModel

from . import init


class OptimizerType(str, Enum):
    Adam = "adam"
    AdamW = "adamw"


class Model(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        model_cfg = cfg["model"]
        model_name = model_cfg["path"]

        mlm_cls = typing.cast(Type[BertPreTrainedModel], BertForMaskedLM)
        if model_name is not None:
            lm: Any = mlm_cls.from_pretrained(model_name)
        else:
            bert_config = BertConfig()
            lm: Any = mlm_cls(bert_config)

        assert lm is not None

        lm = typing.cast(Module, lm)

        if model_cfg["init"]:
            lm.apply(init.bert_init(lm.config))

        self.lm = lm
        self.config = cfg

        loguru.logger.debug("Model used: {}", self.lm)

    @property
    def bert_config(self) -> BertConfig:
        return self.lm.bert.config

    def forward(self, **kwargs: Any) -> BertOutput:
        return self.lm(**kwargs)

    def _step(self, batch: BatchEncoding, batch_idx: int, name: str) -> Tensor:
        loguru.logger.trace("{} step batch: {}", name, batch_idx)

        output: BertOutput = self(**batch)

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
