from reconstruction.ResAE.dataloaders.experiments import *
from reconstruction.ResAE.configs import Config


def get_dataloader(cfg: Config):
    return globals().get(cfg.experiment.name)