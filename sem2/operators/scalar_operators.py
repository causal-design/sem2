"""Scalar operators consuming named data and returning ScalarResult."""
from __future__ import annotations
import torch
from .operators_base import OperatorInput, OperatorOutput, SEM2Operator, _apply_numeric_mapping
from .vector_operators import PotentialOutcomeSample
from .discrete_energy_loss import DiscreteEnergyLoss
from ..model.construction.sem_types import SEMType
from ..data import DataBatch, ScalarResult, VariableInfo


class EnergyLoss(SEM2Operator):
    """Compare named observations with model draws or finite probabilities."""
    input_type = OperatorInput.DATA
    output_type = OperatorOutput.SCALAR

    def __init__(self, cfg, model):
        self.model = model
        self.device = cfg.device
        self.bs = cfg.sample_bs
        self.beta = cfg.energy_score_beta
        self.discrete_loss = DiscreteEnergyLoss(self.beta)

    def evaluate(self, observations: DataBatch, **kwargs) -> tuple[ScalarResult, dict]:
        if not isinstance(observations, DataBatch):
            raise TypeError("observations must be a DataBatch.")
        observations = observations.to(self.device)
        unknown = set(observations.names) - self.model.input_node
        if unknown:
            raise ValueError(f"Unknown or internal observed variables: {sorted(unknown)}")
        if SEMType.is_continuous(self.model.sem_type):
            prediction, _ = self.model.sample(batch_size=self.bs)
            return self._energy_score(observations, prediction)
        names = [name for name in observations.names if name in self.model.state_node_names
                 and not self.model.node_specs[name].is_helper]
        outcomes = observations.subset(names)
        prediction, _ = self.model.sample(batch_size=self.bs)
        return self.discrete_loss.energy_dist(outcomes, prediction)

    def _energy_score(self, observations: DataBatch, prediction: DataBatch) -> tuple[ScalarResult, dict]:
        """Compare observations [N, D] with named draws [M, D], allowing N != M."""
        targets = _apply_numeric_mapping(observations)
        draws = _apply_numeric_mapping(DataBatch(prediction.select(observations.names), observations.variables)).to(targets)
        n = draws.shape[0]
        if not len(observations) or n < 2:
            raise ValueError("Energy score requires observations and at least two model draws.")
        cross = (torch.cdist(targets, draws) ** self.beta).mean()
        distances = torch.cdist(draws, draws) ** self.beta
        pair = (distances.sum() - distances.diagonal().sum()) / (n * (n - 1))
        value = cross - pair / 2
        return ScalarResult(value, "energy_score", observations.variables), {
            "energy_score": value, "estimated-true distance": cross, "estimated-estimated distance": pair
        }

    def __repr__(self):
        return "EnergyLoss"


class MeanPotentialOutcome(SEM2Operator):
    """Mean of a scalar outcome under a named intervention.

    Supply outcome metadata to interpret categories using an explicit numeric mapping.
    Otherwise the existing code-valued mean is retained with a warning.
    """
    input_type = OperatorInput.SELF
    output_type = OperatorOutput.SCALAR

    def __init__(self, target_node: str, intervene_node: str, cfg, model, variable: VariableInfo | None = None):
        self.model = model
        self.target_node = target_node
        self.intervene_node = intervene_node
        if target_node not in model.input_node or intervene_node not in model.input_node:
            raise ValueError("Query nodes must be user-declared model variables.")
        self.variable = variable if variable is not None else model._variable_info(target_node)
        if self.variable.name != target_node or self.variable.width != 1 or model.node_dict[target_node].out_dim != 1:
            raise ValueError("MeanPotentialOutcome requires scalar target metadata matching the model.")
        if model.node_dict[intervene_node].out_dim != 1:
            raise ValueError("MeanPotentialOutcome requires a scalar intervention node.")
        if SEMType.is_continuous(model.sem_type):
            self.sampler = PotentialOutcomeSample(model, target_node)

    def evaluate(self, set_value: DataBatch, **kwargs) -> tuple[ScalarResult, dict]:
        if not isinstance(set_value, DataBatch) or set_value.names != (self.intervene_node,):
            raise TypeError("set_value must be a DataBatch containing the intervention variable.")
        if SEMType.is_continuous(self.model.sem_type):
            draws, details = self.sampler.evaluate(batch_size=len(set_value), set_nodes=set_value)
            value = _apply_numeric_mapping(DataBatch(draws.values, (self.variable,))).mean()
        else:
            prediction, details = self.model.sample(batch_size=len(set_value), set_nodes=set_value)
            prediction = prediction.marginalize([self.target_node])
            states = DataBatch(prediction.states.values, (self.variable,))
            numeric = _apply_numeric_mapping(states)[:, 0].to(prediction.probabilities)
            value = (prediction.probabilities @ numeric).mean()
        return ScalarResult(value, "mean_potential_outcome", (self.variable,)), details


class ATE(SEM2Operator):
    input_type = OperatorInput.SELF
    output_type = OperatorOutput.SCALAR

    def __init__(self, cfg, model, variable: VariableInfo | None = None):
        self.device = cfg.device
        self.bs = cfg.sample_bs
        self.model = model
        self.intervene_node = cfg.T_node
        self.treat_pot_out = MeanPotentialOutcome(cfg.Y_node, cfg.T_node, cfg, model, variable=variable)

    def evaluate(self, **kwargs) -> tuple[ScalarResult, dict]:
        metadata = (self.model._variable_info(self.intervene_node),)
        t0 = DataBatch(torch.zeros((self.bs, 1), device=self.device), metadata)
        t1 = DataBatch(torch.ones((self.bs, 1), device=self.device), metadata)
        y0, _ = self.treat_pot_out.evaluate(t0)
        y1, _ = self.treat_pot_out.evaluate(t1)
        return ScalarResult(y1.value - y0.value, "ATE", y1.variables), {"control": t0, "treatment": t1}

    def __repr__(self):
        return "ATE"


class SEM2Loss(SEM2Operator):
    """Weighted sum of explicit (scalar operator, weight) pairs.

    Component conventions are unchanged: a negative log-likelihood weight gives
    a minimization objective. Diagnostics use index:class prefixes to avoid collisions.
    """
    input_type = OperatorInput.DATA
    output_type = OperatorOutput.SCALAR

    def __init__(self, components: list[tuple[SEM2Operator, float]]) -> None:
        self.loss_components = list(components)
        if not self.loss_components:
            raise ValueError("At least one loss component is required.")
        for component, _ in self.loss_components:
            if component.output_type != OperatorOutput.SCALAR:
                raise TypeError("Loss components must return ScalarResult.")

    def evaluate(self, observations: DataBatch, **kwargs) -> tuple[ScalarResult, dict]:
        if not isinstance(observations, DataBatch):
            raise TypeError("observations must be a DataBatch.")
        # Start with a Python zero so the components determine tensor dtype/device.
        total = 0
        metrics, variables = {}, {}
        for index, (component, weight) in enumerate(self.loss_components):
            if component.input_type == OperatorInput.DATA:
                result, details = component.evaluate(observations, **kwargs)
            else:
                result, details = component.evaluate(**kwargs)
            total = total + weight * result.value
            variables.update((v.name, v) for v in result.variables)
            label = f"{index}:{type(component).__name__}"
            metrics.update((f"{label}/{name}", value) for name, value in details.items())
            metrics[label] = result.value.detach()
        return ScalarResult(total, "loss", tuple(variables.values())), metrics
