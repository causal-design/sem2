"""Prepare a DataFrame for tensor-backed minibatch loading."""
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..data import VariableInfo, prepare_data
from ..model.models import SEM2


def prepare_dataloader(model: SEM2, data: pd.DataFrame, batch_size: int,
                       *, variables: tuple[VariableInfo, ...] = (),
                       device: torch.device | str | None = None):
    """Return a shuffled loader, shared variable metadata and preparation report."""
    batch, report = prepare_data(model, data, variables, device=device)
    loader = DataLoader(TensorDataset(batch.values), batch_size=batch_size, shuffle=True)
    return loader, batch.variables, report
