from omegaconf import DictConfig
from pytorch_lightning import Trainer


def prepare(cfg: DictConfig) -> Trainer:
    return Trainer()
