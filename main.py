import sys

import hydra
import loguru
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from perbert.data import TextDataModule
from perbert.models import Model
from perbert.trainer import Trainer


@hydra.main(config_path="conf", config_name="main")
def app(cfg: DictConfig) -> None:
    # Always seed everything with the given seed.
    pl.seed_everything(cfg["seed"], workers=True)

    logger_cfg = cfg["loggers"]
    if level := logger_cfg["level"]:
        loguru.logger.info("Logger level: {}", level)
        loguru.logger.remove()
        loguru.logger.add(sys.stderr, level=level)

    loguru.logger.info("Config used: {}", OmegaConf.to_yaml(cfg))

    trainer = Trainer(cfg)
    model = Model(cfg)
    datamodule = TextDataModule(cfg)

    # Tuning to fine the best size.
    if cfg["tune"]:
        trainer.tune(model=model, datamodule=datamodule)

    stage_cfg = cfg["stages"]
    ckpt = cfg["ckpt"]

    if stage_cfg["eval_only"]:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt)
        return

    if stage_cfg["pretrain"]:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)

    if stage_cfg["finetune"]:
        raise NotImplementedError


if __name__ == "__main__":
    app()
