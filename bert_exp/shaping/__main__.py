import pytorch_lightning as pl
from hydra import main
from omegaconf import DictConfig

from .data import WikiTextDataModule
from .models import Model
from .trainer import Trainer


@main(config_path="conf", config_name="main")
def app(cfg: DictConfig):
    # Always seed everything with the given seed.
    pl.seed_everything(cfg["seed"], workers=True)

    trainer = Trainer.create(cfg)

    model = Model.create(cfg)

    datamodule = WikiTextDataModule.create(cfg)

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    app()
