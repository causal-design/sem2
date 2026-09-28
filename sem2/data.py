"""Small records shared by loaders, samplers and operators."""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class VariableInfo:
    """Column meaning, separate from node support.

    encoding maps external labels to stored codes; scores maps those labels
    (or stored values without an encoding) to explicitly supplied numeric scores.
    width preserves existing low-level vector nodes, not vector DataFrame cells.
    """
    name: str
    width: int = 1
    categorical: bool | None = None
    encoding: dict | None = None
    scores: dict | None = None


@dataclass
class DataBatch:
    """Rows of values with shared metadata in column order."""
    values: torch.Tensor
    variables: tuple[VariableInfo, ...]

    def __post_init__(self):
        """Keep ordered metadata and check alignment so named selection cannot mislabel columns."""
        self.variables = tuple(self.variables)
        if not isinstance(self.values, torch.Tensor) or self.values.ndim != 2:
            raise ValueError("DataBatch values must be a two-dimensional tensor.")
        if any(not isinstance(v.name, str) or not v.name or not isinstance(v.width, int) or v.width < 1
               for v in self.variables):
            raise ValueError("Variables require nonempty names and positive widths.")
        if len(set(self.names)) != len(self.variables):
            raise ValueError("Variable names must be unique.")
        if self.values.shape[1] != sum(v.width for v in self.variables):
            raise ValueError("Data columns must match variable widths.")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.variables)

    def __len__(self):
        return self.values.shape[0]

    def select(self, names) -> torch.Tensor:
        """Select whole variable blocks in the requested order."""
        names = (names,) if isinstance(names, str) else tuple(names)
        if len(set(names)) != len(names):
            raise ValueError("Selected variable names must be unique.")
        columns = {}
        start = 0
        for variable in self.variables:
            columns[variable.name] = list(range(start, start + variable.width))
            start += variable.width
        missing = set(names) - columns.keys()
        if missing:
            raise ValueError(f"Unknown variables: {sorted(missing)}")
        return self.values[:, [i for name in names for i in columns[name]]]

    def subset(self, names) -> DataBatch:
        names = (names,) if isinstance(names, str) else tuple(names)
        variables = dict(zip(self.names, self.variables))
        values = self.select(names)
        return DataBatch(values, tuple(variables[name] for name in names))

    def __getitem__(self, rows) -> DataBatch:
        """Slice rows, retaining the batch dimension for a single row."""
        if isinstance(rows, tuple):
            raise TypeError("Use subset() to select named columns; indexing only slices rows.")
        values = self.values[rows]
        if values.ndim == 1:
            values = values.unsqueeze(0)
        return DataBatch(values, self.variables)

    def to(self, *args, **kwargs) -> DataBatch:
        return DataBatch(self.values.to(*args, **kwargs), self.variables)


@dataclass
class FiniteDistribution:
    """Named states and probabilities [exogenous draws, states]."""
    states: DataBatch
    probabilities: torch.Tensor

    def __post_init__(self):
        """Check that probability columns index the state table on the same device."""
        if not isinstance(self.states, DataBatch):
            raise TypeError("states must be a DataBatch.")
        if self.probabilities.ndim != 2 or self.probabilities.shape[1] != len(self.states):
            raise ValueError("Probabilities must have shape [draws, states].")
        if self.probabilities.device != self.states.values.device:
            raise ValueError("States and probabilities must be on the same device.")

    def to(self, *args, **kwargs) -> FiniteDistribution:
        probabilities = self.probabilities.to(*args, **kwargs)
        return FiniteDistribution(self.states.to(device=probabilities.device), probabilities)

    def marginalize(self, names) -> FiniteDistribution:
        """Combine equal projected states; retain the exogenous-draw axis."""
        states = self.states.subset(names)
        if states.values.shape[1] == 0:
            values = states.values.new_empty((1, 0))
            probabilities = self.probabilities.sum(-1, keepdim=True)
        else:
            values, inverse = torch.unique(states.values, dim=0, return_inverse=True)
            probabilities = self.probabilities.new_zeros((self.probabilities.shape[0], len(values)))
            probabilities = probabilities.index_add(1, inverse, self.probabilities)
        return FiniteDistribution(DataBatch(values, states.variables), probabilities)


@dataclass
class ScalarResult:
    """A named scalar quantity; value retains its autograd graph."""
    value: torch.Tensor
    name: str
    variables: tuple[VariableInfo, ...] = ()

    def __post_init__(self):
        """Require one tensor-valued quantity without detaching its gradient graph."""
        if not isinstance(self.value, torch.Tensor) or self.value.ndim != 0:
            raise ValueError("ScalarResult value must be a scalar tensor.")
        self.variables = tuple(self.variables)
