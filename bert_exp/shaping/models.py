from typing import Any, Sequence

from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor

from bert_exp.bert import Config, ForMaskedLM, Output


class Bert(LightningModule):
    def __init__(self, model_name: str, cfg: DictConfig) -> None:
        super().__init__()
        lm = ForMaskedLM.from_pretrained(model_name)
        if not isinstance(lm, ForMaskedLM):
            raise ValueError(f"Model name: {model_name} is invalid.")

        self.lm = lm
        self.config = cfg

    @property
    def bert_config(self) -> Config:
        return self.lm.bert.config

    def forward(self, *args: Any, **kwargs: Any) -> Output:
        return self.lm(*args, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        raise NotImplementedError

    def training_step_end(self, batch_parts: Sequence[Tensor]) -> Tensor:
        raise NotImplementedError

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        raise NotImplementedError

    def validation_step_end(self, batch_parts: Sequence[Tensor]) -> Tensor:
        raise NotImplementedError

    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        raise NotImplementedError

    def test_step_end(self, batch_parts: Sequence[Tensor]) -> Tensor:
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
