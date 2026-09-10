"""Small internal data records shared by loaders and operators.

These are not a replacement for the planned DataFrame preparation layer. Each
variable occupies a contiguous block of columns. Metadata describes the stored
representation, not the random variable's attainable support. Treat shared
encodings as read-only; encoding category IDs does not assign numerical scores.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(frozen=True)
class VariableInfo:
    """Column identity and declared semantics; None means unknown.

    ``encoding`` maps external labels to stored integer state codes. It is
    optional: model-generated state IDs need not have external labels.
    Categorical variables are discrete, but numerical discrete variables need
    not be categorical. Neither flag is inferred from a tensor's dtype.
    """
    name: str
    categorical: bool | None = None
    discrete: bool | None = None
    encoding: Mapping[object, int] | None = None
    width: int = 1

    def __post_init__(self):
        if not isinstance(self.width, int) or self.width < 1:
            raise ValueError("Variable width must be a positive integer.")


@dataclass(frozen=True)
class NamedBatch:
    """A [rows, total width] tensor and shared variable metadata.

    ``select`` explicitly unwraps ordered columns for numerical kernels;
    ``subset`` retains their metadata. Tensor operations preserve autograd.
    """
    values: Tensor
    variables: tuple[VariableInfo, ...]
    _index: dict[str, slice] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        variables = tuple(self.variables)
        object.__setattr__(self, "variables", variables)
        if not isinstance(self.values, Tensor) or self.values.ndim != 2:
            raise ValueError("NamedBatch values must be a tensor with shape [rows, total width].")
        if self.values.shape[1] != sum(v.width for v in variables):
            raise ValueError("Tensor width must match the sum of named variable widths.")
        names = tuple(v.name for v in variables)
        if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
            raise ValueError("Variable names must be nonempty, unique strings.")
        index, start = {}, 0
        for variable in variables:
            index[variable.name] = slice(start, start + variable.width)
            start += variable.width
        object.__setattr__(self, "_index", index)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._index)

    def __len__(self) -> int:
        return self.values.shape[0]

    def select(self, *names: str) -> Tensor:
        missing = [name for name in names if name not in self._index]
        if missing:
            raise ValueError(f"Named batch is missing variables: {missing}")
        if not names:
            return self.values[:, :0]
        return torch.cat([self.values[:, self._index[name]] for name in names], dim=1)

    def subset(self, *names: str) -> NamedBatch:
        values = self.select(*names)
        info = {v.name: v for v in self.variables}
        return NamedBatch(values, tuple(info[n] for n in names))

    def rows(self, index) -> NamedBatch:
        values = self.values[index]
        if values.ndim == 1:
            values = values.unsqueeze(0)
        return NamedBatch(values, self.variables)

    def to(self, *args, **kwargs) -> NamedBatch:
        return NamedBatch(self.values.to(*args, **kwargs), self.variables)


@dataclass(frozen=True)
class FiniteDistribution:
    """Named states and probabilities[..., state], never observation rows.

    For SEM2Discrete the leading axis indexes sampled exogenous inputs. Marginal
    projection sums over omitted state coordinates but does not average over
    these draws. Call probabilities.mean(0) explicitly when integrating them.
    """
    states: NamedBatch
    probabilities: Tensor

    def __post_init__(self):
        if not isinstance(self.states, NamedBatch) or not isinstance(self.probabilities, Tensor):
            raise TypeError("FiniteDistribution requires named states and a probability tensor.")
        if self.probabilities.ndim < 1 or self.probabilities.shape[-1] != len(self.states):
            raise ValueError("The final probability axis must match the number of states.")
        if self.probabilities.device != self.states.values.device:
            raise ValueError("States and probabilities must be on the same device.")

    def marginal(self, *names: str) -> FiniteDistribution:
        """Select named variables, coalescing equal rows and summing their mass."""
        if not names:
            raise ValueError("A marginal requires at least one variable.")
        selected = self.states.subset(*names)
        states, inverse = torch.unique(selected.values, dim=0, return_inverse=True)
        shape = (*self.probabilities.shape[:-1], states.shape[0])
        probabilities = self.probabilities.new_zeros(shape).index_add(-1, inverse, self.probabilities)
        return FiniteDistribution(NamedBatch(states, selected.variables), probabilities)
