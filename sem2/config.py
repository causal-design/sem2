from pydantic import BaseModel, ConfigDict
from typing import Literal, Union, Any
import torch

Optimisers = Literal["AdamW", "Adam", "SGD"]

class Config(BaseModel):
    """ Configuration for the model and training process. """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    device: Union[str, Any] = "cpu"

    # Optimiser parameters
    optim: Optimisers = "AdamW"
    lr: float = 10e-4
    weight_decay: float = 0.
    momentum: float = 0.9                  # Only for SGD optimiser

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
