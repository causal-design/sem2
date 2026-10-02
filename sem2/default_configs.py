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
    """

    # Number of real-valued output coordinates. Discrete category counts belong
    # to d(...; levels=N), not to a linear node's output dimension.
    output_dim: int = 1

    # Estimate an intercept when the formula omits a constant. Explicit formula
    # constants always win; otherwise False forces the equation through the origin.
    use_intercept: bool = True


@dataclass
class MLPConfig:
    """Network architecture; discrete output_dim is set by the formula's levels."""

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

    Covers device selection, optimiser and training loop settings.
    Objectives are supplied separately; node architecture is configured via
    LinearNodeConfig and MLPConfig.

    Usage:
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

    # Energy score loss
    use_energy_score: bool = True
    energy_score_weight: float = 1.
    energy_score_beta: float = 1.

    # ATE loss
    use_ate: bool = False
    ate_weight: float = -0.05
    T_node: str = "X"
    Y_node: str = "Y"

    # MMD loss  # TODO
    use_mmd: bool = False
    mmd_weight: float = 1.

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