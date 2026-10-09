"""Prepare pandas observations separately from shared named-value records."""
from __future__ import annotations
from dataclasses import replace
import keyword
from numbers import Integral
from typing import TYPE_CHECKING
import pandas as pd
import torch
from .records import DataBatch, VariableInfo

if TYPE_CHECKING:
    from ..model.components.nodes import RandomVariableSupport
    from ..model.models.base import SEM2


def _normalize_column_names(columns: pd.Index) -> list[str]:
    """Return model-safe names without changing the original DataFrame labels.

    Replace invalid identifier characters, prefix leading digits, suffix Python
    keywords, and reject normalization collisions rather than guess by order.
    """
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
        ordered = series.cat.ordered if categorical else variable.ordered
        series = series.astype(object).map(encoding)
        variable = replace(variable, categorical=True, encoding=encoding, ordered=ordered)
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
    """Check original values, encoding codes and numeric mapping coverage before conversion."""
    name = variable.name
    if not torch.isfinite(value).all() or (
        support.constraint is not None and not support.constraint.check(value).all()
    ):
        raise ValueError(f"Values must be finite and inside the declared domain of {name!r}.")
    if variable.encoding is not None and support.constraint is not None:
        codes = value.new_tensor(list(variable.encoding.values()))
        if not support.constraint.check(codes).all():
            raise ValueError(f"Encoding codes outside the declared domain of {name!r}.")
    if variable.numeric_mapping is None:
        return

    labels = variable.encoding if variable.encoding is not None else series.unique().tolist()
    missing = set(labels) - variable.numeric_mapping.keys()
    if missing:
        raise ValueError(f"Numeric mapping is missing labels for {name!r}: {sorted(missing, key=repr)}")
    mapped = value.new_tensor(list(variable.numeric_mapping.values()))
    if not torch.isfinite(mapped).all():
        raise ValueError(f"Numeric mapping for {name!r} must be finite.")
    if support.values is not None:
        mapped_codes = {variable.encoding[label] if variable.encoding is not None else label
                        for label in variable.numeric_mapping}
        missing = set(support.values.tolist()) - mapped_codes
        if missing:
            raise ValueError(f"Numeric mapping is missing declared states of {name!r}: {sorted(missing, key=repr)}")


def prepare_data(
    model: SEM2,
    frame: pd.DataFrame,
    variables: list[VariableInfo] | None = None,
    device: torch.device | str | None = None,
) -> tuple[DataBatch, dict[str, object]]:
    """Clean and align observations once without mutating the supplied frame.

    Parameters
    ----------
    model : SEM2
        Supplies names, topological order and declared domains, not observation
        status. Fitting columns must be scalar user-declared variables.
    frame : pandas.DataFrame
        Original named observations. Invalid column identifiers are normalized;
        collisions, unknown variables and internal helper names are rejected.
        Rows missing any supplied variable are dropped in a working copy.
        Declare label columns with pandas.Categorical, including unused levels
        and optional ordering; default codes follow the declared category order.
        Other label columns require an explicit encoding. Numeric columns hold
        stored values, not labels awaiting automatic recoding.
    variables : list[VariableInfo] or None
        Optional metadata keyed by normalized names. Explicit numeric mappings
        are validated and preserved, never applied to or substituted for IDs.
    device : torch.device, str or None
        Target device; defaults to the model's device. Values use model dtype.

    Returns
    -------
    batch : DataBatch
        Complete rows with columns and metadata in model topological order.
    report : dict
        Original-to-normalized column names, observed/unobserved model names,
        and input/retained/dropped row counts. No loader or fitting is created.

    Notes
    -----
    Domains, encoding and mapping coverage are checked before dtype narrowing;
    overflow is checked afterwards. Observation status belongs to this data
    pairing, not the structural graph. Category ordering does not supply scores.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame; convert the input to a DataFrame first.")
    if not frame.columns.is_unique or not len(frame.columns):
        raise ValueError("DataFrame columns must be nonempty and unique.")
    if any(name in model.node_specs and model.node_specs[name].is_helper for name in frame.columns):
        raise ValueError("Data cannot reference internal helper nodes.")
    names = _normalize_column_names(frame.columns)
    unknown = set(names) - model.input_node
    if unknown:
        raise ValueError(f"Unknown model columns after normalization: {sorted(unknown)}")
    variables = [] if variables is None else variables
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
        if variable.dim != 1 or node.out_dim != 1 or (
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

    batch = DataBatch(torch.stack(values, dim=-1), metadata)
    report = {
        "column_names": dict(zip(frame.columns, names)),
        "observed": batch.names,
        "unobserved": tuple(name for name in model.topological_order
                            if name in model.input_node and name not in names),
        "input_rows": len(frame), "retained_rows": len(data), "dropped_rows": len(frame) - len(data),
    }
    return batch, report
