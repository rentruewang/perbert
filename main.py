import sys

import loguru
import pytorch_lightning as pl
from hydra import main
from omegaconf import DictConfig, OmegaConf

from perbert.data import TextDataModule
from perbert.models import Model
from perbert.trainer import Trainer


@main(config_path="conf", config_name="main")
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

    # Tuning to fine the best size.
    if cfg["tune"]:
        trainer.tune(model=model, datamodule=datamodule)

    stage_cfg = cfg["stages"]

    if stage_cfg["eval_only"]:
        datamodule = TextDataModule(cfg)
        trainer.test(model=model, datamodule=datamodule)
        return

    if stage_cfg["pretrain"]:
        datamodule = TextDataModule(cfg)
        trainer.fit(model=model, datamodule=datamodule)

    if stage_cfg["finetune"]:
        raise NotImplementedError


if __name__ == "__main__":
    app()
