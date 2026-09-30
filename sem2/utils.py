"""Shared runtime utilities."""
from collections.abc import Iterator
from contextlib import contextmanager
import torch


@contextmanager
def seeded_rng(seed: int | None = 0, device: torch.device | str = "cpu") -> Iterator[None]:
    """Temporarily seed Torch on CPU and the selected CUDA/MPS device.

    None means zero. Use as a context manager or decorator; surrounding RNG
    states are restored even if the seeded operation raises an exception.
    """
    seed = 0 if seed is None else seed
    device = torch.device(device)
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    elif device.type == "mps":
        devices = [0]

    # Save CPU and the selected accelerator only. Seeding their generators
    # directly avoids changing unrelated devices through torch.manual_seed().
    with torch.random.fork_rng(devices=devices, device_type="mps" if device.type == "mps" else "cuda"):
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.default_generators[devices[0]].manual_seed(seed)
        elif device.type == "mps":
            torch.mps.manual_seed(seed)
        yield
