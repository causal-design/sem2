"""Finite probability laws computed by models, not observed data or graph nodes."""
from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
import torch
from ..data.records import DataBatch


@dataclass
class DiscreteJointDistribution:
    """Finite named states and probabilities, not realized observations.

    probabilities has shape [exogenous draws, states]. Each row is a conditional
    distribution for one Monte Carlo sample of the exogenous inputs. Averaging
    rows estimates the marginal law.
    This record represents either a joint distribution or a selected marginal.
    """
    states: DataBatch
    probabilities: torch.Tensor

    def __post_init__(self) -> None:
        """Check that probability columns index the state table on the same device."""
        if not isinstance(self.states, DataBatch):
            raise TypeError("states must be a DataBatch.")
        if self.probabilities.ndim != 2 or self.probabilities.shape[1] != len(self.states):
            raise ValueError("Probabilities must have shape [draws, states].")
        if self.probabilities.device != self.states.values.device:
            raise ValueError("States and probabilities must be on the same device.")

    def to(self, *args, **kwargs) -> DiscreteJointDistribution:
        probabilities = self.probabilities.to(*args, **kwargs)
        return DiscreteJointDistribution(self.states.to(device=probabilities.device), probabilities)

    def marginalize(self, names: str | Iterable[str]) -> DiscreteJointDistribution:
        """Combine equal projected states; retain the exogenous-draw axis."""
        states = self.states.subset(names)
        if states.values.shape[1] == 0:
            values = states.values.new_empty((1, 0))
            probabilities = self.probabilities.sum(-1, keepdim=True)
        else:
            values, inverse = torch.unique(states.values, dim=0, return_inverse=True)
            probabilities = self.probabilities.new_zeros((self.probabilities.shape[0], len(values)))
            probabilities = probabilities.index_add(1, inverse, self.probabilities)
        return DiscreteJointDistribution(DataBatch(values, states.variables), probabilities)
