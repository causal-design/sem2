"""Energy fitting for discrete instrumental-variable problems."""
from __future__ import annotations
import torch
from .divergence import EnergyLoss
from ..data.records import DataBatch, ScalarResult
from ..default_configs import TrainingConfig
from ..model.components.nodes import NodeExogenous
from ..model.models.discrete import SEM2Discrete


class DiscreteIVEnergyLoss(EnergyLoss):
    """Fit conditional outcome distributions at each declared instrument level.

    Conditional half-energy distances use empirical instrument frequencies.
    Fitting uses the complete prepared dataset in one batch, with at least two
    observations per declared level. No levels are dropped or given pseudo-counts.

    Callers supply a model encoding the IV assumptions. This class checks neither
    those causal assumptions nor identification, and does not certify effect or
    bound recovery.

    Parameters
    ----------
    cfg : TrainingConfig
        Device and default sample count and energy exponent.
    model : SEM2Discrete
        Model encoding the supplied IV assumptions.
    instrument : str
        Scalar exogenous variable with enumerated support of at least two levels.
    sample_bs : int, optional
        Exogenous draws per instrument level; overrides the config count.
    beta : float, optional
        Distance exponent; overrides the config value.
    unbiased : bool, optional
        Pair correction; overrides ``cfg.energy_score_unbiased`` at construction.
    """

    def __init__(self, cfg: TrainingConfig, model: SEM2Discrete, instrument: str, *,
                 sample_bs: int | None = None, beta: float | None = None,
                 unbiased: bool | None = None) -> None:
        super().__init__(cfg, model, sample_bs=sample_bs, beta=beta, unbiased=unbiased)
        if not isinstance(model, SEM2Discrete):
            raise TypeError("DiscreteIVEnergyLoss requires SEM2Discrete.")
        node = model.node_dict[instrument] if instrument in model.input_node else None
        if not isinstance(node, NodeExogenous) or node.out_dim != 1:
            raise ValueError("The instrument must be a scalar exogenous discrete node.")
        support = node.support
        if support.constraint is None or not support.constraint.is_discrete or support.values is None:
            raise ValueError("The instrument requires enumerated discrete support.")
        self.instrument = instrument
        self.levels = support.values.reshape(-1).tolist()
        if len(self.levels) < 2:
            raise ValueError("The instrument requires at least two declared levels.")

    def _group_observations(self, observations: DataBatch) -> list[DataBatch]:
        """Select conditional margins and report ineligible levels before sampling."""
        names = [n for n in observations.names if n in self.model.state_node_names]
        outcomes = observations.subset(names)
        z = observations.select(self.instrument)[:, 0]
        groups = [outcomes[z == level] for level in self.levels]
        counts = dict(zip(self.levels, map(len, groups)))
        if min(counts.values()) < 2:
            raise ValueError(f"Instrument {self.instrument!r} requires at least two observations "
                             f"per declared level; counts={counts}.")
        return groups

    def evaluate(self, observations: DataBatch, *, sample_bs: int | None = None,
                 beta: float | None = None) -> tuple[ScalarResult, dict]:
        """Average conditional discrepancies using empirical instrument weights.

        Parameters
        ----------
        observations : DataBatch
            Instrument and observed endogenous variables. Each declared level
            needs at least two observations; no stratification is performed.
        sample_bs : int, optional
            Model draws per level for this call; corrected pairs need two.
        beta : float, optional
            Distance exponent for this call.

        Returns
        -------
        result : ScalarResult
            Group-frequency-weighted half-energy distance, retaining gradients.
        diagnostics : dict
            Weighted energy components and total variation; ``max_cell_error``
            is the largest conditional cell error across levels. Keys prefixed
            with ``instrument=level/`` retain each level's counts and diagnostics.
        """
        sample_bs = self.sample_bs if sample_bs is None else sample_bs
        beta = self.beta if beta is None else beta
        observations = observations.to(self.device)
        unknown = set(observations.names) - self.model.input_node
        if unknown:
            raise ValueError(f"Unknown or internal observed variables: {sorted(unknown)}")
        groups = self._group_observations(observations)
        loss = self._prepare_discrete_loss(groups[0])
        instrument = observations.subset(self.instrument)
        total = 0
        metrics = {}
        for level, group in zip(self.levels, groups):
            intervention = DataBatch(instrument.values.new_full((sample_bs, 1), level), instrument.variables)
            prediction, _ = self.model.sample(batch_size=sample_bs, set_nodes=intervention)
            result, details = loss.energy_dist(group, prediction, beta=beta)
            weight = len(group) / len(observations)
            total = total + weight * result.value
            prefix = f"{self.instrument}={level}/"
            metrics[prefix + "count"] = len(group)
            for name, value in details.items():
                metrics[prefix + name] = value
                if name != "max_cell_error":
                    metrics[name] = metrics.get(name, 0) + weight * value
        metrics["max_cell_error"] = torch.stack([
            metrics[f"{self.instrument}={level}/max_cell_error"] for level in self.levels
        ]).max()
        return ScalarResult(total, "energy_distance", groups[0].variables), metrics

    def __repr__(self) -> str:
        return "DiscreteIVEnergyLoss"
