from typing import Callable

import loguru
from torch.nn import Embedding, LayerNorm, Linear, Module

from bert_exp import Config


def bert_init(cfg: Config) -> Callable[[Module], None]:
    loguru.logger.info("Creating the initialization function.")

    def init(layer: Module) -> None:
        if isinstance(layer, Linear):
            linear_init(layer, cfg)

        if isinstance(layer, Embedding):
            emb_init(layer, cfg)

        if isinstance(layer, LayerNorm):
            layernorm_init(layer, cfg)

    return init


def linear_init(layer: Linear, cfg: Config) -> None:
    loguru.logger.debug(f"Calling linear_init on {layer}")

    layer.weight.normal_(mean=0.0, std=cfg.initializer_range)

    if (bias := layer.bias) is not None:
        bias.zero_()


def layernorm_init(layer: LayerNorm, cfg: Config) -> None:
    del cfg
    loguru.logger.debug(f"Calling linearnorm_init on {layer}")

    layer.weight.fill_(1.0)
    layer.bias.fill_(0.0)


def emb_init(layer: Embedding, cfg: Config) -> None:
    loguru.logger.debug(f"Calling emb_init on {layer}")

    layer.weight.normal_(mean=0.0, std=cfg.initializer_range)
