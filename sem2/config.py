""" Default configuration. """
# TODO: What sort of config to use? Dataclass? Frozen dataclass? Pydantic? How should the user specify it?
from dataclasses import dataclass
from typing import Literal
import torch

Optimisers = Literal["AdamW", "Adam", "SGD"]

@dataclass(frozen=True)
class Config:
    """ Configuration for the model and training process. """
    device: str | torch.device = "cpu"

    # Optimiser parameters
    optim: Optimisers = "AdamW"
    lr: float = 10e-4
    weight_decay: float = 0.
    momentum = 0.9                  # Only for SGD optimiser

    # Training parameters
    batch_size: int = 128
    epochs: int = 100
    rng_seed: int = 42
    bs: int = 256               # Batch size used in training for sampling distributions.

    # Metrics to use in loss function
    # Energy score
    use_energy_score: bool = True
    energy_score_weight: float = 1.
    energy_score_beta: float = 1.
    # ATE
    use_ate: bool = True
    ate_weight: float = 0.01
    T_node: str = "D"
    Y_node: str = "Y"

    # MMD   # TODO
    use_mmd: bool = False
    mmd_weight: float = 1.


def get_config(extra_args: dict=None) -> Config:
    """ Generate config. Extra arguments override default config. """
    if extra_args is None:
        extra_args = {}

    # Dynamically set device if not given
    if "device" not in extra_args:
        extra_args["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    print(extra_args)
    return Config(**extra_args)
