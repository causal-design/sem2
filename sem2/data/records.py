"""Small named-value records shared by preparation, samplers and operators."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Iterable
import torch


@dataclass
class VariableInfo:
    """Column meaning, separate from node support.

    encoding maps external labels to stored codes; numeric_mapping maps those labels
    (or stored values without an encoding) to explicitly supplied numeric values.
    dim preserves existing low-level vector nodes, not vector DataFrame cells.
    """
    name: str
    dim: int = 1
    categorical: bool | None = None
    encoding: dict | None = None
    numeric_mapping: dict | None = None


@dataclass
class DataBatch:
    """Rows with ordered metadata; strings create scalar VariableInfo defaults."""
    values: torch.Tensor
    variables: tuple[VariableInfo | str, ...] | list[VariableInfo | str]

    def __post_init__(self) -> None:
        """Keep ordered metadata and check alignment so named selection cannot mislabel columns."""
        self.variables = tuple(VariableInfo(v) if isinstance(v, str) else v for v in self.variables)
        if not isinstance(self.values, torch.Tensor) or self.values.ndim != 2:
            raise ValueError("DataBatch values must be a two-dimensional tensor.")
        if any(not isinstance(v.name, str) or not v.name or not isinstance(v.dim, int) or v.dim < 1
               for v in self.variables):
            raise ValueError("Variables require nonempty names and positive dimensions.")
        if len(set(self.names)) != len(self.variables):
            raise ValueError("Variable names must be unique.")
        if self.values.shape[1] != sum(v.dim for v in self.variables):
            raise ValueError("Data columns must match variable dimensions.")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.variables)

    def __len__(self) -> int:
        return self.values.shape[0]

    def select(self, names: str | Iterable[str]) -> torch.Tensor:
        """Select whole variable blocks in the requested order."""
        names = (names,) if isinstance(names, str) else tuple(names)
        if len(set(names)) != len(names):
            raise ValueError("Selected variable names must be unique.")
        columns = {}
        start = 0
        for variable in self.variables:
            columns[variable.name] = list(range(start, start + variable.dim))
            start += variable.dim
        missing = set(names) - columns.keys()
        if missing:
            raise ValueError(f"Unknown variables: {sorted(missing)}")
        return self.values[:, [i for name in names for i in columns[name]]]

    def subset(self, names: str | Iterable[str]) -> DataBatch:
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
class ScalarResult:
    """A named scalar quantity; value retains its autograd graph."""
    value: torch.Tensor
    name: str
    variables: tuple[VariableInfo, ...] = ()

    def __post_init__(self) -> None:
        """Require one tensor-valued quantity without detaching its gradient graph."""
        if not isinstance(self.value, torch.Tensor) or self.value.ndim != 0:
            raise ValueError("ScalarResult value must be a scalar tensor.")
        self.variables = tuple(self.variables)
