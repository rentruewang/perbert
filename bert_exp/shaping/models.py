from turtle import forward
from typing import Any, Tuple

from pytorch_lightning import LightningModule

from bert_exp.bert import Config, ForMaskedLM, Model, Output


class Bert(LightningModule):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.bert = ForMaskedLM.from_pretrained(model_name)

    def forward(self, *args: Any, **kwargs: Any) -> Output:
        return self.bert(*args, **kwargs)
