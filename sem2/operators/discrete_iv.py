"""Energy fitting for discrete instrumental-variable problems."""
from __future__ import annotations
import torch
import networkx as nx
from .divergence import EnergyLoss
from ..data.records import DataBatch, ScalarResult
from ..default_configs import TrainingConfig
from ..model.distributions import DiscreteJointDistribution
from ..model.models.discrete import SEM2Discrete


class DiscreteIVEnergyLoss(EnergyLoss):
    """Fit conditional outcome distributions at each declared instrument level.

    Conditional half-energy distances use empirical instrument frequencies.
    Fitting uses complete cases in one batch. Every instrument level needs one
    row per observed stratum (two for distinct-pair estimation).

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
        Observed scalar variable with enumerated support of at least two levels.
    covariates : sequence of str, optional
        Observed discrete baseline variables. Fit within each stratum and
        average using the empirical marginal frequencies of the strata.
    sample_bs : int, optional
        Exogenous draws per instrument level; overrides the config count.
    beta : float, optional
        Distance exponent; overrides the config value.
    unbiased : bool, optional
        Pair correction; overrides ``cfg.energy_score_unbiased`` at construction.
    """

    def __init__(self, cfg: TrainingConfig, model: SEM2Discrete, instrument: str, *,
                 sample_bs: int | None = None, beta: float | None = None,
                 unbiased: bool | None = None, covariates: tuple[str, ...] = ()) -> None:
        super().__init__(cfg, model, sample_bs=sample_bs, beta=beta, unbiased=unbiased)
        if not isinstance(model, SEM2Discrete):
            raise TypeError("DiscreteIVEnergyLoss requires SEM2Discrete.")
        self.covariates = tuple(covariates)
        if len(set((instrument, *self.covariates))) != 1 + len(self.covariates):
            raise ValueError("Instrument and baseline covariates must be distinct.")
        for name in (instrument, *self.covariates):
            node = model.node_dict[name] if name in model.input_node else None
            if node is None or node.out_dim != 1 or node.support.values is None:
                raise ValueError(f"{name!r} requires scalar enumerated support.")
        self.instrument = instrument
        self.levels = model.node_dict[instrument].support.values.reshape(-1).tolist()
        if len(self.levels) < 2:
            raise ValueError("The instrument requires at least two declared levels.")

        g = nx.DiGraph(model.admg.directed)
        g.add_nodes_from(model.input_node)
        if set(self.covariates) & nx.descendants(g, instrument):
            raise ValueError("Baseline covariates cannot descend from the instrument.")

    def check_iv_graph(self, treatment: str, outcome: str) -> None:
        """Check exclusion and outcome exchangeability in the graph given baseline C.

        Raises ValueError on violation. Does not test strength or establish the
        stronger no-unmeasured-Z/treatment-confounding assumption of the loss.
        """
        z, c = self.instrument, set(self.covariates)
        if ({treatment, outcome} - self.model.input_node or
                len({z, treatment, outcome, *c}) != 3 + len(c)):
            raise ValueError("Specify distinct known treatment, outcome, instrument and covariates.")
        graph = self.model.admg
        g = nx.DiGraph(graph.directed)
        g.add_nodes_from(graph.endogenous | graph.exogenous)
        if c & (nx.descendants(g, treatment) | nx.descendants(g, outcome)):
            raise ValueError("Conditioning variables must be baseline covariates.")
        bypass = g.copy()
        bypass.remove_node(treatment)
        if nx.has_path(bypass, z, outcome):
            raise ValueError("Exclusion fails: a directed path bypasses treatment.")
        # A latent fork per bidirected edge preserves m-separation. Fixing X
        # removes its incoming arrows, including any shared-source arrows.
        for edge in graph.bidirected:
            g.add_edges_from((edge, vertex) for vertex in edge)
        g.remove_edges_from(list(g.in_edges(treatment)))
        if not nx.is_d_separator(g, {z}, {outcome}, c):
            raise ValueError("Instrument and outcome are not separated after fixing treatment.")

    def _groups(self, observations):
        names = [n for n in observations.names if n in self.model.state_node_names
                 and n not in (self.instrument, *self.covariates)]
        if not names:
            raise ValueError("No observed endogenous variables to score.")
        outcomes = observations.subset(names)
        z = observations.select(self.instrument)[:, 0]
        c = observations.select(self.covariates) if self.covariates else z.new_empty((len(z), 0))
        strata = torch.unique(c, dim=0) if self.covariates else [c.new_empty(0)]
        for row in strata:
            mask = (c == row).all(1)
            for level in self.levels:
                group = outcomes[mask & (z == level)]
                if len(group) < (2 if self.unbiased else 1):
                    raise ValueError(f"Instrument {level}, stratum {row.tolist()}: counts={len(group)}; "
                                     "each level needs one row (at least two rows for U-statistics).")
                yield dict(zip(self.covariates, row.tolist())), level, group

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
        total = 0
        metrics = {}
        maxima = []
        for stratum, level, group in self._groups(observations):
            loss = self._prepare_discrete_loss(group)
            # Isolated exogenous C are independent roots, so conditioning fixes
            # their draws. For endogenous C restrict the joint state table instead.
            fixed = {self.instrument: level, **{n:v for n,v in stratum.items()
                                                if n in self.model.admg.exogenous}}
            intervention = DataBatch(observations.values.new_tensor(list(fixed.values())).expand(sample_bs,-1),
                                     list(fixed))
            prediction, _ = self.model.sample(batch_size=sample_bs, do_intervention=intervention)
            mask = torch.ones(len(prediction.states), dtype=torch.bool, device=self.device)
            weights = None
            for name, value in stratum.items():
                if name not in fixed:
                    mask &= prediction.states.select(name)[:, 0] == value
            if len(fixed) < 1 + len(stratum):
                p = prediction.probabilities[:, mask]
                weights = p.sum(1)
                if not weights.sum() > 0:
                    raise ValueError(f"Zero predicted probability for stratum {stratum}.")
                prediction = DiscreteJointDistribution(prediction.states[mask],
                    p / torch.where(weights > 0, weights, 1)[:, None])
            result, details = loss.energy_dist(group, prediction, beta=beta, draw_weights=weights)
            weight = len(group) / len(observations)
            total = total + weight * result.value
            prefix = ''.join(f'{n}={v}/' for n,v in stratum.items()) + f"{self.instrument}={level}/"
            maxima.append(details['max_cell_error'])
            metrics[prefix + "count"] = len(group)
            for name, value in details.items():
                metrics[prefix + name] = value
                if name != "max_cell_error":
                    metrics[name] = metrics.get(name, 0) + weight * value
        metrics["max_cell_error"] = torch.stack(maxima).max()
        return ScalarResult(total, "energy_distance", group.variables), metrics

    def __repr__(self) -> str:
        return "DiscreteIVEnergyLoss"
