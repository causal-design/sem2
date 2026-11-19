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
    Fitting uses the complete prepared dataset in one batch. Each declared
    level needs two observations for Ustat or one for Vstat.

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
    pair_estimator : {"Ustat", "Vstat"}, optional
        Pair correction; overrides ``cfg.energy_pair_estimator`` at construction.
    """

    def __init__(self, cfg: TrainingConfig, model: SEM2Discrete, instrument: str, *,
                 sample_bs: int | None = None, beta: float | None = None,
                 pair_estimator: str | None = None) -> None:
        super().__init__(cfg, model, sample_bs=sample_bs, beta=beta, pair_estimator=pair_estimator)
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

    def validate_fit_data(self, observations: DataBatch, batch_size: int) -> None:
        if batch_size < len(observations):
            raise ValueError(f"IV fitting requires a full-data batch; set batch_size >= {len(observations)}.")
        counts = dict(zip(self.levels, map(len, self._split_by_instrument(observations))))
        minimum = 2 if self.cfg.energy_pair_estimator == "Ustat" else 1
        if min(counts.values()) < minimum:
            raise ValueError(f"{self.cfg.energy_pair_estimator} requires at least {minimum} "
                             f"observation(s) per {self.instrument} level; counts={counts}.")

    def _split_by_instrument(self, observations: DataBatch) -> list[DataBatch]:
        """Return one batch of endogenous observations per instrument level.

        Group rows by their observed instrument value in ``self.levels`` order;
        exclude the instrument column from the variables whose fit is scored.
        """
        names = [n for n in observations.names if n in self.model.state_node_names]
        outcomes = observations.subset(names)
        z = observations.select(self.instrument)[:, 0]
        return [outcomes[z == level] for level in self.levels]

    def evaluate(self, observations: DataBatch, *, sample_bs: int | None = None) -> tuple[ScalarResult, dict]:
        """Average conditional discrepancies using empirical instrument weights.

        Parameters
        ----------
        observations : DataBatch
            Instrument and observed endogenous variables. Each declared level
            needs at least two observations for Ustat or one for Vstat.
        sample_bs : int, optional
            Model draws per level for this call; corrected pairs need two.

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
        observations = observations.to(self.device)
        unknown = set(observations.names) - self.model.input_node
        if unknown:
            raise ValueError(f"Unknown or internal observed variables: {sorted(unknown)}")
        groups = self._split_by_instrument(observations)
        loss = self._get_discrete_energy_loss(groups[0])
        instrument = observations.subset(self.instrument)
        total = 0
        metrics = {}
        for level, group in zip(self.levels, groups):
            intervention = DataBatch(instrument.values.new_full((sample_bs, 1), level), instrument.variables)
            prediction, _ = self.model.sample(batch_size=sample_bs, do_intervention=intervention)
            result, details = loss.energy_dist(group, prediction)
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
