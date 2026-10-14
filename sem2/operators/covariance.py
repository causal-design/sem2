"""Model covariance over all user-declared variables."""
from __future__ import annotations
import torch
from .operators_base import OperatorInput, OperatorOutput, SEM2Operator
from ..data.records import DataBatch
from ..model.models.base import SEM2
from ..model.models.linear import SEM2Linear
from ..default_configs import TrainingConfig


class Covariance(SEM2Operator):
    """Exact linear covariance, otherwise estimated from Monte Carlo samples.

    Both matrix axes follow variables, with vector variables occupying blocks.
    Finite-state models integrate discrete states and sample exogenous inputs;
    the conditional-covariance decomposition retains probability gradients.
    Assumes finite second moments. Category codes retain their model-defined
    numeric representation.
    sample_bs overrides cfg.sample_bs; evaluate(sample_bs=...) affects one call.

    Parameters
    ----------
    model : SEM2
        Model whose user-declared variables define the matrix axes.
    sample_bs : int, optional
        Exogenous/model draw count. Non-linear models require at least two.
    cfg : TrainingConfig, optional
        Supplies ``sample_bs`` when no explicit count is given.
    """
    input_type = OperatorInput.SELF
    output_type = OperatorOutput.MATRIX

    def __init__(self, model: SEM2, sample_bs: int | None = None,
                 *, cfg: TrainingConfig | None = None) -> None:
        cfg = cfg if cfg is not None else TrainingConfig(device=model.device)
        self.model = model
        self.sample_bs = cfg.sample_bs if sample_bs is None else sample_bs
        self.names = tuple(n for n in model.topological_order if n in model.input_node)
        self.variables = [model._variable_info(n) for n in self.names]
        self.columns = {}
        start = 0
        for variable in self.variables:
            self.columns[variable.name] = list(range(start, start + variable.dim))
            start += variable.dim
        if not isinstance(model, SEM2Linear) and self.sample_bs < 2:
            raise ValueError("Covariance requires at least two Monte Carlo samples.")

    def evaluate(self, *, sample_bs: int | None = None) -> tuple[torch.Tensor, dict]:
        """Return the full covariance matrix and its ordered variable metadata.

        Parameters
        ----------
        sample_bs : int, optional
            Draw-count override, ignored for analytical linear covariance.

        Returns
        -------
        covariance : torch.Tensor
            Shape (total_width, total_width), retaining parameter gradients.
        details : dict
            ``variables`` describes consecutive row and column blocks.
        """
        if isinstance(self.model, SEM2Linear):
            covariance = self.model.cov_matrix(list(self.names))
        else:
            sample_bs = self.sample_bs if sample_bs is None else sample_bs
            if sample_bs < 2:
                raise ValueError("Covariance requires at least two Monte Carlo samples.")
            prediction, _ = self.model.sample(sample_bs)
            if isinstance(prediction, DataBatch):
                covariance = torch.cov(prediction.select(self.names).T)
            else:
                probabilities = prediction.probabilities
                states = prediction.states.values.to(probabilities)
                conditional = probabilities @ states
                blocks = {n: conditional[:, i:i + 1] for i, n in enumerate(prediction.states.names)}
                means = torch.cat([blocks[n] if n in blocks else self.model.node_dict[n].get_value().to(probabilities)
                                   for n in self.names], dim=-1)
                covariance = torch.cov(means.T).reshape(means.shape[1], means.shape[1])
                # Total covariance = covariance of conditional means + mean conditional covariance.
                within = states.T @ (probabilities.mean(0)[:, None] * states)
                within = within - conditional.T @ conditional / len(conditional)
                indices = [i for n in prediction.states.names for i in self.columns[n]]
                covariance[torch.tensor(indices, device=states.device)[:, None], indices] += within
        width = sum(v.dim for v in self.variables)
        return covariance.reshape(width, width), {"variables": self.variables}
