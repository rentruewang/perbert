# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import typing
from enum import Enum
from typing import Any, Dict, List, Tuple, Type

import loguru
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor, no_grad
from torch.nn import Module
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import Accuracy, Metric
from transformers import BatchEncoding, BertConfig, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertOutput, BertPreTrainedModel
from transformers import optimization

from . import init
from .length_schedulers import LengthScheduler


class OptimizerType(str, Enum):
    Adam = "adam"
    AdamW = "adamw"


class Model(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        model_cfg = cfg["model"]
        model_name = model_cfg["path"]
        self.max_length = cfg["data"]["max_length"]

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
        assert isinstance(logits, Tensor)
        size = logits.size(0) * logits.size(1)
        logits = logits.reshape(size, logits.size(2))
        labels = batch.labels.view(-1)

        for (metric, func) in self.metrics.items():
            result = func(logits, labels)
            self.log(name=f"{name}/{metric}", value=result, on_step=True)

        return loss

    def training_step(self, batch: BatchEncoding, batch_idx: int) -> Tensor:
        max_length = self.length_scheduler.step()
        if max_length < self.max_length:
            for k, v in batch.data.items():
                batch.data[k] = v[:, :max_length].contiguous()
                del v
            torch.cuda.empty_cache()
        return self._step(batch, batch_idx=batch_idx, name="train")

    @no_grad()
    def test_step(self, batch: BatchEncoding, batch_idx: int) -> Tensor:
        return self._step(batch, batch_idx=batch_idx, name="test")

    @no_grad()
    def validation_step(self, batch: BatchEncoding, batch_idx: int) -> Tensor:
        return self._step(batch, batch_idx=batch_idx, name="validation")

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LambdaLR]]:
        model_cfg = self.cfg["model"]

        optim_type = OptimizerType(model_cfg["optimizer"])
        if optim_type == OptimizerType.Adam:
            optim_cls = Adam
        elif optim_type == OptimizerType.AdamW:
            optim_cls = AdamW
        else:
            raise ValueError(f"Optimizer type: {optim_type} not supported.")

        weight_decay = model_cfg["weight_decay"]
        lr = model_cfg["lr"]
        optimizer = optim_cls(
            params=self.parameters(), lr=lr, weight_decay=weight_decay
        )

        assert self.trainer is not None
        total_steps = int(self.trainer.estimated_stepping_batches)
        lr_scheduler_type = model_cfg["lr_scheduler_type"]
        warmup_steps = (
            int(total_steps * model_cfg["warmup_ratio"])
            if model_cfg["warmup_ratio"]
            else model_cfg["warmup_steps"]
        )

        self.length_scheduler = LengthScheduler(
            max_length=self.cfg["data"]["max_length"],
            total_steps=total_steps,
            **self.cfg["model"]["length_scheduler"],
        )
        scheduler = optimization.get_scheduler(
            lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        loguru.logger.info("Optimizer: {}", optimizer)
        return ([optimizer], [scheduler])

    def configure_metrics(self) -> Dict[str, Metric]:
        metrics = {}

        met_cfg = self.cfg["model"]["metrics"]

        if met_cfg["accuracy"]:
            loguru.logger.info("Accuracy metric is used.")
            self.acc = Accuracy(ignore_index=-100)
            metrics["acc"] = self.acc

        return metrics
