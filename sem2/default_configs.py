import warnings
from dataclasses import dataclass, field
from typing import Literal, Union, Any, Optional

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ══════════════════════════════════════════════════════════════════════════════
# Node configs  (public, user-facing)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LinearNodeConfig:
    """Public configuration for linear SEM nodes.

    input_dim is not exposed as it is inferred from the graph structure.
    init_weight_val and init_bias_val are not exposed since fixed coefficients
    are specified inline in the formula syntax (e.g. '1*x', '-0.5*x'),
    keeping the model formula self-contained and readable.

    Parameters
    ----------
    output_dim : int, default=1
        Number of real-valued output coordinates, not category levels.
    use_intercept : bool, default=True
        Estimate an intercept when the formula omits a constant. An explicit
        formula constant, including zero, takes precedence.
    """

    # Number of real-valued output coordinates. Discrete category counts belong
    # to d(...; levels=N), not to a linear node's output dimension.
    output_dim: int = 1

    # Estimate an intercept when the formula omits a constant. Explicit formula
    # constants always win; otherwise False forces the equation through the origin.
    use_intercept: bool = True


@dataclass
class MLPConfig:
    """Network architecture; discrete output_dim is set by the formula's levels.

    Parameters
    ----------
    input_dim : int, optional
        Inferred from predecessor widths during node construction.
    hidden_dims : list[int], optional
        Hidden-layer widths; defaults to [512, 512, 512]. Copied at construction.
    output_dim : int, default=1
        Continuous output width; discrete nodes use the formula's level count.
    dropout : float, default=0.0
        Dropout probability after each hidden activation.
    activation : str, default="relu"
        Hidden activation: linear, relu, tanh, sigmoid, leaky_relu, elu, gelu or
        selu. The output layer is linear.
    """

    input_dim: int = None                                                   # Inferred from graph
    hidden_dims: list = field(default_factory=lambda: [512, 512, 512])      # Hidden layer widths
    output_dim: int = 1                                                     # Output dimension
    dropout: float = 0.0                                                    # 0 = no dropout
    activation: str = "relu"                                                # relu | tanh | sigmoid | ... https://docs.pytorch.org/docs/2.12/nn.html

    def __post_init__(self) -> None:
        """Copy layer widths so this config does not share the caller's mutable list."""
        self.hidden_dims = list(self.hidden_dims)


# ══════════════════════════════════════════════════════════════════════════════
# Training config  (public, user-facing)
# ══════════════════════════════════════════════════════════════════════════════

Optimisers = Literal["AdamW", "Adam", "SGD"]


class TrainingConfig(BaseModel):
    """Configuration for a SEM2 training run.

    Covers device selection, optimiser, training loop and general operator settings.
    Objectives are supplied separately; node architecture is configured via
    LinearNodeConfig and MLPConfig.

    Parameters
    ----------
    device : str or torch.device, optional
        Defaults to CUDA, then MPS, then CPU according to availability.
        Construct the model on the same device before fitting.
    optim : {"AdamW", "Adam", "SGD"}, default="AdamW"
        Optimizer family.
    lr : float, default=0.0005
        Positive learning rate.
    weight_decay : float, default=0.0
        Nonnegative optimizer weight decay.
    momentum : float, default=0.9
        Nonnegative SGD momentum; unused by Adam and AdamW.
    batch_size : int, default=1024
        Positive number of observations per training minibatch.
    epochs : int, default=12
        Positive number of passes through the data.
    rng_seed : int, optional
        Explicit training seed; None does not reseed. Queries have independent seeds.
    sample_bs : int, default=512
        Default operator Monte Carlo count, separate from data batch size.
    energy_score_beta : float, default=1.0
        Distance exponent, strictly between zero and two.
    energy_score_unbiased : bool, default=True
        Exclude same-draw pairs in energy losses. False uses all pairs.

    Examples
    --------
    Configuration with automatic device selection or an explicit override::

        cfg = TrainingConfig()                        # auto-selects best device
        cfg = TrainingConfig(device="cpu", epochs=50) # explicit override
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid", allow_inf_nan=False)

    # Device: leave as None to auto-select CUDA > MPS > CPU.
    # Construct the model on the same device before fitting.
    device: Union[str, Any] = None

    # Optimiser
    optim: Optimisers = "AdamW"
    lr: float = Field(default=5e-4, gt=0)
    weight_decay: float = Field(default=0., ge=0)
    momentum: float = Field(default=0.9, ge=0)       # SGD only

    # Training loop
    batch_size: int = Field(default=1024, gt=0)
    epochs: int = Field(default=12, gt=0)
    rng_seed: int | None = None
    sample_bs: int = 512                           # Sample batch size for distribution sampling

    # Energy score
    energy_score_beta: float = Field(default=1., gt=0, lt=2)
    energy_score_unbiased: bool = True

    @model_validator(mode="after")
    def _resolve_device(self) -> "TrainingConfig":
        """Auto-select device when none is specified."""
        if self.device is None:
            # model_config is frozen so we use object.__setattr__ to set
            # the resolved device without triggering the immutability guard.
            object.__setattr__(self, "device", _get_device())
        return self


# ══════════════════════════════════════════════════════════════════════════════
# Device (private)
# ══════════════════════════════════════════════════════════════════════════════

def _get_device() -> str:
    """Returns the best available device string: CUDA > MPS > CPU.
    Private users can set device via TrainingConfig(device="cuda") to override,
    or leave it unset to trigger auto-detection.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[device] GPU detected: {gpu_name} -- using CUDA")
        return "cuda"
    if torch.backends.mps.is_available():
        warnings.warn(
            "[device] MPS detected -- using Apple GPU. "
            "Note: MPS has limited op support; fall back to CPU with device='cpu' if you see errors.",
            stacklevel=2,
        )
        return "mps"
    warnings.warn(
        "[device] No GPU detected -- falling back to CPU. "
        "Training may be significantly slower.",
        stacklevel=2,
    )
    return "cpu"