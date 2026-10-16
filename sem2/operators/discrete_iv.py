"""Energy fitting for discrete instrumental-variable problems."""
from __future__ import annotations
import torch
from .divergence import EnergyLoss
from ..data.records import DataBatch, ScalarResult
from ..default_configs import TrainingConfig
from ..model.components.nodes import NodeExogenous
from ..model.models.discrete import SEM2Discrete


class DiscreteIVEnergyLoss(EnergyLoss):
    """Provisional placeholder for the deferred, reviewed IV fitting interface.

    The existing executable calculation is retained for compatibility, not as
    completed Step-7 functionality. It compares conditional outcome distributions
    for a scalar exogenous discrete instrument using empirical group frequencies.

    Deferred: an IV model API/convenience initializer, estimator/weight review,
    valid minibatch policy and independent IV fitting validation. No stratified
    batching is implemented. Each evaluated batch currently needs at least two
    observations at every declared instrument level; ordinary shuffled batches
    need not satisfy this and may raise in the energy kernel.

    Callers supply a model encoding the IV assumptions. This class checks neither
    those causal assumptions nor identification, and does not certify effect or
    bound recovery. The deferred work above requires separate implementation
    and validation before this can be treated as a supported IV fitting API.
    """

    def __init__(self, cfg: TrainingConfig, model: SEM2Discrete, instrument: str, *,
                 sample_bs: int | None = None, beta: float | None = None) -> None:
        super().__init__(cfg, model, sample_bs=sample_bs, beta=beta)
        if not isinstance(model, SEM2Discrete):
            raise TypeError("DiscreteIVEnergyLoss requires SEM2Discrete.")
        node = model.node_dict[instrument] if instrument in model.input_node else None
        if not isinstance(node, NodeExogenous) or node.out_dim != 1:
            raise ValueError("The instrument must be a scalar exogenous discrete node.")
        support = node.support
        if support.constraint is None or not support.constraint.is_discrete or support.values is None:
            raise ValueError("The instrument requires enumerated discrete support.")
        self.instrument = instrument
        self.levels = support.values.reshape(-1)
        if len(self.levels) < 2:
            raise ValueError("The instrument requires at least two declared levels.")

    def evaluate(self, observations: DataBatch, *, sample_bs: int | None = None,
                 beta: float | None = None) -> tuple[ScalarResult, dict]:
        sample_bs = self.sample_bs if sample_bs is None else sample_bs
        beta = self.beta if beta is None else beta
        observations = observations.to(self.device)
        unknown = set(observations.names) - self.model.input_node
        if unknown:
            raise ValueError(f"Unknown or internal observed variables: {sorted(unknown)}")
        names = [n for n in observations.names if n in self.model.state_node_names]
        outcomes = observations.subset(names)
        instrument = observations.subset(self.instrument)
        z = instrument.values[:, 0]
        total = observations.values.new_zeros((), dtype=torch.float32)
        metrics = {}
        # Keep group-size checks in the energy kernel, not in the sampling loop.
        for level in self.levels:
            group = outcomes[z == level]
            intervention = DataBatch(level.expand(sample_bs, 1), instrument.variables)
            prediction, _ = self.model.sample(batch_size=sample_bs, set_nodes=intervention)
            result, details = self.discrete_loss.energy_dist(group, prediction, beta=beta)
            weight = len(group) / len(observations)
            total = total + weight * result.value
            for name, value in details.items():
                metrics[name] = metrics.get(name, 0) + weight * value
        return ScalarResult(total, "energy_distance", outcomes.variables), metrics

    def __repr__(self) -> str:
        return "DiscreteIVEnergyLoss"
