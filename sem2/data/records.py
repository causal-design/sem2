"""Small named-value records shared by preparation, samplers and operators."""
from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Iterable
import torch


@dataclass
class VariableInfo:
    """Meaning of a named variable, separate from its model support.

    Parameters
    ----------
    name : str
        Model variable name, after column-name normalization.
    dim : int, default 1
        Number of tensor component columns, not the number of category levels.
        Vector-valued low-level nodes are supported; DataFrame fitting is scalar.
    categorical : bool or None
        Whether stored values denote categories. Preparation infers None.
    encoding : dict or None
        Label-to-code mapping derived from declared pandas categories for the
        ordinary categorical interface. Users need not supply integer IDs.
        Unobserved declared levels are retained; codes are not analysis scores.
    numeric_mapping : dict or None
        Explicit label-to-score mapping, or value-to-score mapping without an
        encoding. Preparation preserves this separate from stored identities.
    ordered : bool, default False
        Declared pandas category ordering; does not imply numeric spacing.
    """
    name: str
    dim: int = 1
    categorical: bool | None = None
    encoding: dict | None = None
    numeric_mapping: dict | None = None
    ordered: bool = False


@dataclass
class DataBatch:
    """Two-dimensional named values with aligned variable metadata.

    Parameters
    ----------
    values : torch.Tensor
        Shape [rows, sum(variable.dim)]. Rows can be observations, model draws,
        intervention assignments or enumerated states; the caller gives context.
    variables : list[VariableInfo]
        Ordered metadata for consecutive column blocks. Use VariableInfo(name)
        for scalar defaults, not bare string shorthand.

    Notes
    -----
    Selection and slicing preserve metadata and gradients. This is a thin
    tensor record, not a DataFrame, probability distribution or scalar quantity.
    """
    values: torch.Tensor
    variables: list[VariableInfo]

    def __post_init__(self) -> None:
        """Keep ordered metadata and check alignment so named selection cannot mislabel columns."""
        self.variables = list(self.variables)
        if any(not isinstance(v, VariableInfo) for v in self.variables):
            raise TypeError("variables must contain VariableInfo objects, not names.")
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
        return DataBatch(values, [variables[name] for name in names])

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
    """A named zero-dimensional tensor retaining its autograd graph.

    Associated variables describe the quantity, not observation columns.
    Consequently this record is separate from the two-dimensional DataBatch;
    it already has a name and need not inherit row/column slicing.
    """
    value: torch.Tensor
    name: str
    variables: list[VariableInfo] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Require one tensor-valued quantity without detaching its gradient graph."""
        if not isinstance(self.value, torch.Tensor) or self.value.ndim != 0:
            raise ValueError("ScalarResult value must be a scalar tensor.")
        self.variables = list(self.variables)
