# pyright: reportPrivateImportUsage=false
from typing import Callable

import loguru
from torch.nn import Embedding, LayerNorm, Linear, Module, init
from transformers import BertConfig


def bert_init(cfg: BertConfig) -> Callable[[Module], None]:
    loguru.logger.info("Creating the initialization function.")

    def init(layer: Module) -> None:
        if isinstance(layer, Linear):
            linear_init(layer, cfg)

        if isinstance(layer, Embedding):
            emb_init(layer, cfg)

        if isinstance(layer, LayerNorm):
            layernorm_init(layer, cfg)

    return init


def linear_init(layer: Linear, cfg: BertConfig) -> None:
    loguru.logger.trace("Calling linear_init on {}", layer)

    init.normal_(layer.weight, mean=0.0, std=cfg.initializer_range)

    if (bias := layer.bias) is not None:
        init.zeros_(bias)


def layernorm_init(layer: LayerNorm, cfg: BertConfig) -> None:
    del cfg
    loguru.logger.trace("Calling linearnorm_init on {}", layer)

    init.ones_(layer.weight)
    init.zeros_(layer.bias)


def emb_init(layer: Embedding, cfg: BertConfig) -> None:
    loguru.logger.trace("Calling emb_init on {}", layer)

    init.normal_(layer.weight, mean=0.0, std=cfg.initializer_range)
