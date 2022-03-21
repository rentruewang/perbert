from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Sequence

from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor, no_grad
from torch.nn import functional as F
from torch.optim import Adam, AdamW, Optimizer

from bert_exp.bert import Config, ForMaskedLM, Output, BatchEncoding


class OptimizerType(str, Enum):
    Adam = "adam"
    AdamW = "adamw"


class Model(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        model_cfg = cfg["model"]
        model_name = model_cfg["path"]

        lm = ForMaskedLM.from_pretrained(model_name)
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
        del batch_idx

        output: Output = self(**batch)

        # FIXME: loss here is None because labels are not provided.
        return output.loss

    @no_grad()
    def validation_step(
        self, batch: BatchEncoding, batch_idx: int
    ) -> Dict[str, Tensor]:
        return self.training_step(batch, batch_idx=batch_idx)

    @no_grad()
    def test_step(self, batch: BatchEncoding, batch_idx: int) -> Dict[str, Tensor]:
        return self.training_step(batch, batch_idx=batch_idx)

    def configure_optimizers(self) -> Optimizer:
        model_cfg: str = self.config["model"]
        optim_type = OptimizerType(model_cfg["optimizer"])
        lr = model_cfg["lr"]

        if optim_type == OptimizerType.Adam:
            return Adam(self.parameters(), lr=lr)
        elif optim_type == OptimizerType.AdamW:
            return AdamW(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Optimizer type: {optim_type} not supported.")

    @classmethod
    def create(cls, cfg: DictConfig) -> Model:
        return cls(cfg)
