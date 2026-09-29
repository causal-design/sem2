"""Small records shared by loaders, samplers and operators."""
from __future__ import annotations
from dataclasses import dataclass, replace
import keyword
from numbers import Integral
from typing import TYPE_CHECKING
import pandas as pd
import torch

if TYPE_CHECKING:
    from ..model.components.nodes import RandomVariableSupport
    from ..model.models.base import SEM2


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


def _column_names(columns: pd.Index) -> list[str]:
    """Normalize labels, rejecting ambiguous names rather than guessing by order."""
    used, names = set(), []
    for label in columns:
        if isinstance(label, str) and label.isidentifier() and not keyword.iskeyword(label):
            name = label
        else:
            name = "".join(char if ("_" + char).isidentifier() else "_" for char in str(label))
            if not name or not name.isidentifier():
                name = "_" + name
            if keyword.iskeyword(name):
                name += "_"
        if name in used:
            raise ValueError(f"Column names collide after normalization: {name!r}. Rename them explicitly.")
        names.append(name)
        used.add(name)
    return names


def _encode_column(series: pd.Series, variable: VariableInfo) -> tuple[pd.Series, VariableInfo]:
    """Encode declared labels; numeric columns already contain stored values."""
    categorical = isinstance(series.dtype, pd.CategoricalDtype)
    encoding = variable.encoding
    if categorical or not pd.api.types.is_numeric_dtype(series.dtype):
        if variable.categorical is False:
            raise ValueError(f"Label column {variable.name!r} cannot be declared quantitative.")
        if any(not pd.api.types.is_scalar(value) for value in series):
            raise ValueError(f"Only scalar observations are supported: {variable.name!r}.")
        if encoding is None:
            if not categorical:
                raise TypeError(f"Use pandas categorical dtype or supply an encoding for {variable.name!r}.")
            encoding = dict(zip(series.cat.categories, range(len(series.cat.categories))))
        labels = series.cat.categories.tolist() if categorical else series.unique().tolist()
        missing = set(labels) - encoding.keys()
        if missing:
            raise ValueError(f"Encoding is missing labels for {variable.name!r}: {sorted(missing, key=repr)}")
        series = series.astype(object).map(encoding)
        variable = replace(variable, categorical=True, encoding=encoding)
    if encoding is not None:
        codes = list(encoding.values())
        if any(not isinstance(code, Integral) for code in codes) or len(set(codes)) != len(codes):
            raise ValueError(f"Encoding for {variable.name!r} must use unique integer codes.")
        unknown = set(series.unique().tolist()) - set(codes)
        if unknown:
            raise ValueError(f"Unknown encoded values for {variable.name!r}: {sorted(unknown, key=repr)}")
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_complex_dtype(series.dtype):
        raise TypeError(f"Variable {variable.name!r} must contain real scalar values.")
    return series, variable


def _validate_column(
    series: pd.Series,
    value: torch.Tensor,
    variable: VariableInfo,
    support: RandomVariableSupport,
) -> None:
    """Check original numeric values, encoding codes and score coverage before dtype conversion."""
    name = variable.name
    if not torch.isfinite(value).all() or (
        support.constraint is not None and not support.constraint.check(value).all()
    ):
        raise ValueError(f"Values must be finite and inside the declared domain of {name!r}.")
    if variable.encoding is not None and support.constraint is not None:
        codes = value.new_tensor(list(variable.encoding.values()))
        if not support.constraint.check(codes).all():
            raise ValueError(f"Encoding codes outside the declared domain of {name!r}.")
    if variable.scores is None:
        return

    labels = variable.encoding if variable.encoding is not None else series.unique().tolist()
    missing = set(labels) - variable.scores.keys()
    if missing:
        raise ValueError(f"Scores are missing labels for {name!r}: {sorted(missing, key=repr)}")
    scores = value.new_tensor(list(variable.scores.values()))
    if not torch.isfinite(scores).all():
        raise ValueError(f"Scores for {name!r} must be finite.")
    if support.values is not None:
        scored_codes = {variable.encoding[label] if variable.encoding is not None else label
                        for label in variable.scores}
        missing = set(support.values.tolist()) - scored_codes
        if missing:
            raise ValueError(f"Scores are missing declared states of {name!r}: {sorted(missing, key=repr)}")


def prepare_data(
    model: SEM2,
    frame: pd.DataFrame,
    variables: tuple[VariableInfo, ...] = (),
    device: torch.device | str | None = None,
) -> tuple[DataBatch, dict[str, object]]:
    """Prepare a DataFrame once, returning (DataBatch, report) without mutation.

    Metadata uses normalized model names. Numeric columns contain stored values;
    pandas categorical columns contain labels, with declared category order used
    for default encoding. Other label columns require an explicit encoding.
    Supplied scores are preserved, not applied or inferred here.

    Rows missing any supplied variable are dropped and counted in the report.
    Observation status belongs to this batch, not the structural graph.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame; convert the input to a DataFrame first.")
    if not frame.columns.is_unique or not len(frame.columns):
        raise ValueError("DataFrame columns must be nonempty and unique.")
    if any(name in model.node_specs and model.node_specs[name].is_helper for name in frame.columns):
        raise ValueError("Data cannot reference internal helper nodes.")
    names = _column_names(frame.columns)
    unknown = set(names) - model.input_node
    if unknown:
        raise ValueError(f"Unknown model columns after normalization: {sorted(unknown)}")
    supplied = {v.name: v for v in variables}
    if len(supplied) != len(variables) or set(supplied) - set(names):
        raise ValueError("Metadata must have unique names matching supplied columns.")

    # Work on a copy and select complete cases before allocating model tensors.
    data = frame.copy()
    data.columns = names
    data = data.dropna()
    if data.empty:
        raise ValueError("No complete rows remain.")
    parameter = next(model.parameters(), None)
    dtype = parameter.dtype if parameter is not None else torch.get_default_dtype()
    device = device if device is not None else (parameter.device if parameter is not None else model.device)
    observed = [name for name in model.topological_order if name in names]
    values, metadata = [], []
    for name in observed:
        node = model.node_dict[name]
        variable = supplied.get(name, model._variable_info(name))
        if variable.width != 1 or node.out_dim != 1 or (
            node.dist is not None and (node.dist.batch_shape or node.dist.event_shape)
        ):
            raise ValueError(f"Only scalar fitting variables are supported: {name!r}.")
        if variable.categorical is None:
            variable = replace(variable, categorical=(model._variable_info(name).categorical
                               or isinstance(data[name].dtype, pd.CategoricalDtype)
                               or variable.encoding is not None))
        if name not in supplied and isinstance(data[name].dtype, pd.CategoricalDtype):
            variable = replace(variable, categorical=True)
        series, variable = _encode_column(data[name], variable)
        # Check original numeric values before narrowing to the model's dtype.
        value = torch.as_tensor(series.to_numpy(dtype="float64", copy=True), device=device)
        _validate_column(series, value, variable, node.support)
        # Finite source values can still overflow when converted to the model dtype.
        value = value.to(dtype=dtype)
        if not torch.isfinite(value).all():
            raise ValueError(f"Values of {name!r} overflow the model dtype.")
        values.append(value)
        metadata.append(variable)

    batch = DataBatch(torch.stack(values, dim=-1), tuple(metadata))
    report = {
        "column_names": dict(zip(frame.columns, names)),
        "observed": batch.names,
        "unobserved": tuple(name for name in model.topological_order
                            if name in model.input_node and name not in names),
        "input_rows": len(frame), "retained_rows": len(data), "dropped_rows": len(frame) - len(data),
    }
    return batch, report
