# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import typing
from enum import Enum
from operator import truediv
from typing import Any, Dict, List, Type

import loguru
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor, no_grad
from torch.nn import Module
from torch.optim import Adam, AdamW, Optimizer
from torchmetrics import Accuracy, Metric
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
        self.cfg = cfg

        self.metrics = self.configure_metrics()

        loguru.logger.debug("Model used: {}", self.lm)

    @property
    def bert_config(self) -> BertConfig:
        return self.lm.bert.config

    def forward(self, **kwargs: Any) -> BertOutput:
        return self.lm(**kwargs)

    def _step(self, batch: BatchEncoding, batch_idx: int, name: str) -> Tensor:
        loguru.logger.trace("{} step batch: {}", name, batch_idx)

        output: BertOutput = self(**batch)
        loss = typing.cast(Tensor, output.loss)
        self.log(name=f"{name}/loss", value=loss.item(), on_step=True)

        logits = output.logits
        input_ids = batch.input_ids

        for (metric, func) in self.metrics.items():
            result = func(logits, input_ids)
            self.log(name=f"{name}/{metric}", value=result, on_step=True)

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
        model_cfg = self.cfg["model"]
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

    def configure_metrics(self) -> Dict[str, Metric]:
        metrics = {}

        met_cfg = self.cfg["model"]["metrics"]

        if met_cfg["accuracy"]:
            loguru.logger.info("Accuracy metric is used.")
            self.metrics["acc"] = Accuracy()

        return metrics
