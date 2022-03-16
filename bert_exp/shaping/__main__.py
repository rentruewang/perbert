from hydra import main
from omegaconf import DictConfig, OmegaConf


@main(config_path="conf", config_name="main")
def app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    app()
