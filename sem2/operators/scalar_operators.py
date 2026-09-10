# Keep TYPE_CHECKING-only annotations usable on Python < 3.14.
from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from .operators_base import OperatorInput, OperatorOutput, SEM2Operator
from .vector_operators import PotentialOutcome
from .discrete_energy_loss import DiscreteEnergyLoss
from ..data import NamedBatch, FiniteDistribution
from ..model.construction import SEMType

if TYPE_CHECKING:
    from ..model.models import SEM2
    from ..default_configs import TrainingConfig


class EnergyLoss(SEM2Operator):
    """Energy fitting with explicit variable alignment.

    With instrument_node=None, fit the observed joint distribution (marginalizing
    unobserved endogenous states in a discrete model). Setting instrument_node
    explicitly opts into the existing IV-specific discrete objective: compare
    groups with draws under do(instrument=level). This assumes an exogenous
    instrument in the supplied causal model; it is not general conditioning.
    Distances still use stored numerical values, not inferred category scores.
    """
    input_type = OperatorInput.DATA
    output_type = OperatorOutput.SCALAR

    def __init__(self, cfg: TrainingConfig, model: SEM2):
        self.model = model
        self.bs = cfg.sample_bs
        self.beta = cfg.energy_score_beta
        self.instrument_node = cfg.instrument_node
        if self.bs < 2:
            raise ValueError("Unbiased energy fitting needs sample_bs >= 2.")
        if not (SEMType.is_continuous(model.sem_type) or SEMType.is_discrete(model.sem_type)):
            raise NotImplementedError(f"Energy score is not implemented for {model.sem_type}.")
        if self.instrument_node is not None:
            node = model.node_dict[self.instrument_node] if self.instrument_node in model.node_dict else None
            if (not SEMType.is_discrete(model.sem_type) or node is None
                    or node.dist is None or not node.dist.support.is_discrete):
                raise ValueError("instrument_node requires an exogenous discrete node in a discrete model.")

    def evaluate(self, observations: NamedBatch, **kwargs) -> tuple[torch.Tensor, dict]:
        if not isinstance(observations, NamedBatch):
            raise TypeError("EnergyLoss expects a NamedBatch; provide explicit variable metadata.")
        if not len(observations) or not observations.names:
            raise ValueError("Energy fitting requires nonempty observations and variables.")
        observations = observations.to(self.model.device)
        if SEMType.is_discrete(self.model.sem_type):
            return self._evaluate_discrete(observations)
        return self._evaluate_continuous(observations)

    def _discrete_group(self, observations: NamedBatch, set_nodes=None):
        if not observations.names:
            raise ValueError("No fitting variables remain after selecting the instrument.")
        result, _ = self.model.sample(batch_size=self.bs, set_nodes=set_nodes)
        result = result.marginal(*observations.names)
        return DiscreteEnergyLoss(self.beta).energy_dist(observations, result)

    def _evaluate_discrete(self, observations: NamedBatch):
        if self.instrument_node is None:
            return self._discrete_group(observations)
        instrument = observations.select(self.instrument_node).squeeze(-1)
        instrument_batch = self.model._interventions(len(observations), observations.subset(self.instrument_node))
        fit_names = tuple(n for n in observations.names if n != self.instrument_node)
        fitting = observations.subset(*fit_names)
        loss = observations.values.new_zeros((), dtype=torch.float32)
        metrics = {}
        # Only observed levels: absent levels need not be fabricated, and level
        # counts are not assumed binary. Each unbiased group needs >= 2 rows.
        for level in torch.unique(instrument):
            mask = instrument == level
            group = fitting.rows(mask)
            if len(group) < 2:
                raise ValueError("Unbiased IV energy distance needs at least two observations per instrument level.")
            set_nodes = NamedBatch(level.expand(self.bs, 1), instrument_batch.variables)
            value, extra = self._discrete_group(group, set_nodes)
            weight = mask.float().mean()
            loss = loss + weight * value
            for name, statistic in extra.items():
                metrics[name] = metrics.get(name, 0) + weight * statistic
        return loss, metrics

    def _evaluate_continuous(self, observations: NamedBatch):
        unknown = set(observations.names) - set(self.model.node_dict)
        if unknown:
            raise ValueError(f"Observations contain unknown model variables: {sorted(unknown)}")
        self.model._variable_info(observations.names, observations.variables)
        all_preds, _ = self.model.sample(batch_size=self.bs)
        preds = all_preds.subset(*observations.names)
        return self._energy_score(observations.values.to(preds.values), preds.values)

    def _energy_score(self, targets, preds, **kwargs):
        # Independent observation and model sample counts; all cross-pairs.
        m = preds.shape[0]
        cross = (torch.cdist(targets, preds) ** self.beta).mean()
        pair = torch.cdist(preds, preds) ** self.beta
        within = (pair.sum() - pair.diagonal().sum()) / (m * (m - 1))
        score = cross - within / 2
        return score, {"energy_score": score, "estimated-true distance": cross,
                       "estimated-estimated distance": within}

    def __repr__(self):
        return "EnergyLoss"


class MeanPotentialOutcome(SEM2Operator):
    """E[target | do(intervene=value)], using named outcomes.

    As before, discrete means use the numerical state values. This does not
    assign meaningful scores to arbitrary categorical labels; F28 is separate.
    """
    input_type = OperatorInput.DATA
    output_type = OperatorOutput.SCALAR

    def __init__(self, target_node: str, intervene_node: str, cfg: TrainingConfig, model: SEM2):
        self.model = model
        self.target_node = target_node
        self.intervene_node = intervene_node
        if intervene_node not in model.node_dict:
            raise ValueError(f"Unknown intervention node: {intervene_node!r}")
        self.sampler = PotentialOutcome(model, target_node)
        if model.node_dict[target_node].out_dim != 1 or model.node_dict[intervene_node].out_dim != 1:
            raise ValueError("MeanPotentialOutcome requires scalar target and intervention nodes.")

    def evaluate(self, set_value: NamedBatch, **kwargs) -> tuple[torch.Tensor, dict]:
        if not isinstance(set_value, NamedBatch):
            raise TypeError("MeanPotentialOutcome expects a NamedBatch.")
        if set_value.names != (self.intervene_node,):
            raise ValueError(f"MeanPotentialOutcome requires only the intervention variable {self.intervene_node!r}.")
        outcome, _ = self.sampler.evaluate(
            batch_size=len(set_value), set_nodes=set_value,
        )
        if isinstance(outcome, FiniteDistribution):
            probabilities = outcome.probabilities.mean(0)  # integrate exogenous draws explicitly
            values = outcome.states.select(self.target_node).squeeze(-1).to(probabilities)
            return (probabilities * values).sum(), {}
        values = outcome.select(self.target_node)
        if not values.is_floating_point():
            values = values.float()
        return values.mean(), {}


class ATE(SEM2Operator):
    """E[Y | do(T=1)] - E[Y | do(T=0)], with roles selected from config."""
    input_type = OperatorInput.SELF
    output_type = OperatorOutput.SCALAR

    def __init__(self, cfg: TrainingConfig, model: SEM2):
        self.device = model.device
        self.bs = cfg.sample_bs
        self.model = model
        self.treat_pot_out = MeanPotentialOutcome(cfg.Y_node, cfg.T_node, cfg, model)

    def evaluate(self, *args, **kwargs) -> tuple[torch.Tensor, dict]:
        # Float values also work for discrete nodes, after exact validation, and
        # avoid feeding integer inputs into continuous linear/MLP mechanisms.
        variables = self.model._variable_info((self.treat_pot_out.intervene_node,))
        t0 = NamedBatch(torch.zeros([self.bs, 1], device=self.device), variables)
        t1 = NamedBatch(torch.ones([self.bs, 1], device=self.device), variables)
        pot_out_t0, _ = self.treat_pot_out.evaluate(t0)
        pot_out_t1, _ = self.treat_pot_out.evaluate(t1)
        return pot_out_t1 - pot_out_t0, {}

    def __repr__(self):
        return "ATE"


class SEM2Loss(SEM2Operator):
    """Weighted scalar objectives over a named batch, plus model-only queries."""
    input_type = OperatorInput.DATA
    output_type = OperatorOutput.SCALAR

    def __init__(self, cfg: TrainingConfig, model: SEM2):
        self.device = model.device
        components = []
        if cfg.use_energy_score:
            components.append((EnergyLoss(cfg, model), cfg.energy_score_weight))
        if cfg.use_ate:
            components.append((ATE(cfg, model), cfg.ate_weight))
        if cfg.use_mmd:
            raise NotImplementedError("MMD loss not implemented yet.")
        if not components:
            raise ValueError("Enable at least one training objective.")
        self.loss_components = components

    def evaluate(self, observations: NamedBatch, **kwargs) -> tuple[torch.Tensor, dict]:
        if not isinstance(observations, NamedBatch):
            raise TypeError("SEM2Loss expects a NamedBatch.")
        total_loss = torch.tensor(0.0, device=self.device)
        metrics = {}
        for component, weight in self.loss_components:
            args = (observations,) if component.input_type == OperatorInput.DATA else ()
            value, sub_metrics = component.evaluate(*args, **kwargs)
            total_loss = total_loss + weight * value
            metrics.update(sub_metrics)
            metrics[str(component)] = value.detach()
        return total_loss, metrics
