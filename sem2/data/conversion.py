"""Explicit numerical views of named values without changing stored identities."""
import warnings
import torch
from .records import DataBatch


def databatch_to_tensor(batch: DataBatch) -> torch.Tensor:
    """Apply numeric mappings without replacing stored category identities.

    Mappings are assumed complete; unmapped values propagate as NaN.
    The no-mapping path warns and uses codes numerically. That fallback is not a
    category-aware metric; declaring ordered categories does not justify it.
    Supply explicit scores for meaningful means
    or distances. Noncategorical finite values may also use an explicit mapping.

    Parameters
    ----------
    batch : DataBatch
        Named values and optional label-to-score mappings. Mapped variables must
        be scalar; encodings translate labels to the stored identities.

    Returns
    -------
    torch.Tensor
        Numerical values with the same shape and device. Integer blocks become
        floating point. The original values and metadata are unchanged.
    """
    columns = []
    blocks = batch.values.split([v.dim for v in batch.variables], dim=-1)
    for variable, values in zip(batch.variables, blocks):
        if not values.is_floating_point():
            values = values.float()
        if variable.numeric_mapping is not None:
            if variable.dim != 1:
                raise ValueError("Numeric mappings require scalar variables.")
            mapped = torch.full_like(values, float("nan"))
            for label, value in variable.numeric_mapping.items():
                code = variable.encoding[label] if variable.encoding is not None else label
                mask = values == code
                mapped = torch.where(mask, value, mapped)

            values = mapped
        elif variable.categorical:
            warnings.warn(
                f"Using stored category codes numerically for {variable.name!r}; "
                "supply a numeric mapping or use category-probability contrasts.",
                stacklevel=3,
            )
        columns.append(values)
    return torch.cat(columns, dim=-1)
