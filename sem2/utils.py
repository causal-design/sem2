"""Shared runtime utilities."""
import torch


def set_seed(seed: int | None = None, device: torch.device | str = "cpu") -> None:
    """Seed Torch on CPU and the selected CUDA/MPS device only when requested.

    None is a no-op. Explicit seeds change the default generators in place;
    subsequent sampling advances them normally, without state restoration.

    Parameters
    ----------
    seed : int, optional
        Seed for the default Torch generators; ``None`` leaves them unchanged.
    device : str or torch.device, default="cpu"
        Also seed this CUDA device or the MPS generator, in addition to CPU.
        NumPy and Python's random generators are not affected.
    """
    if seed is None:
        return
    device = torch.device(device)
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        # Initialize the backend before accessing its default generators.
        torch.cuda.init()
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.default_generators[index].manual_seed(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)
