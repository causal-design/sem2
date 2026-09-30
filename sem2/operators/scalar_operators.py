"""Scalar operators consuming named data and returning ScalarResult."""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from .operators_base import OperatorInput, OperatorOutput, SEM2Operator, _apply_numeric_mapping
from .vector_operators import PotentialOutcomeSample
from .discrete_energy_loss import DiscreteEnergyLoss
from ..model.construction.sem_types import SEMType
from ..data import DataBatch, ScalarResult, VariableInfo
from ..utils import seeded_rng

if TYPE_CHECKING:
    from ..model.models.base import SEM2


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
        # Cross term: average every observation/model pair, even when N != M.
        cross = (torch.cdist(targets, draws) ** self.beta).mean()
        distances = torch.cdist(draws, draws) ** self.beta
        # Model pair term: omit self-pairs for the unbiased finite-sample estimate.
        pair = (distances.sum() - distances.diagonal().sum()) / (n * (n - 1))
        value = cross - pair / 2
        return ScalarResult(value, "energy_score", observations.variables), {
            "energy_score": value, "estimated-true distance": cross, "estimated-estimated distance": pair
        }

    def __repr__(self):
        return "EnergyLoss"


class MeanPotentialOutcome(SEM2Operator):
    """Mean of a scalar outcome under constant named interventions.

    n_samples counts Monte Carlo samples; discrete outcomes are integrated over
    their states. seed=None means seed=0. Evaluation restores the surrounding
    RNG state, so repeated queries are reproducible without reseeding training.
    Supply variable metadata for an explicit category-to-number mapping.
    """
    input_type = OperatorInput.SELF
    output_type = OperatorOutput.SCALAR

    def __init__(self, model: SEM2, *, outcome: str, intervention: dict[str, float],
                 n_samples: int = 1000, seed: int | None = 0,
                 variable: VariableInfo | None = None) -> None:
        self.model = model
        self.outcome = outcome
        self.intervention = dict(intervention)
        self.n_samples = n_samples
        self.seed = seed
        if outcome not in model.input_node or set(intervention) - model.input_node:
            raise ValueError("Query nodes must be user-declared model variables.")
        self.variable = variable if variable is not None else model._variable_info(outcome)
        if self.variable.name != outcome or self.variable.width != 1 or model.node_dict[outcome].out_dim != 1:
            raise ValueError("MeanPotentialOutcome requires scalar outcome metadata matching the model.")
        if any(model.node_dict[name].out_dim != 1 for name in intervention):
            raise ValueError("MeanPotentialOutcome requires scalar intervention nodes.")
        if SEMType.is_continuous(model.sem_type):
            self.sampler = PotentialOutcomeSample(model, outcome)

    def evaluate(self, **kwargs) -> tuple[ScalarResult, dict]:
        """Evaluate the configured intervention without changing global RNG state."""
        parameter = next(self.model.parameters(), None)
        device = parameter.device if parameter is not None else torch.device(self.model.device)
        dtype = parameter.dtype if parameter is not None else torch.get_default_dtype()
        # Broadcast constant assignments once; samplers still consume named batches.
        values = [torch.as_tensor(value, device=device, dtype=dtype).expand(self.n_samples, 1)
                  for value in self.intervention.values()]
        set_nodes = DataBatch(torch.cat(values, dim=1),
                              tuple(self.model._variable_info(n) for n in self.intervention)) if values else None

        with seeded_rng(self.seed, device=device):
            if SEMType.is_continuous(self.model.sem_type):
                draws, details = self.sampler.evaluate(batch_size=self.n_samples, set_nodes=set_nodes)
                value = _apply_numeric_mapping(DataBatch(draws.values, (self.variable,))).mean()
            else:
                prediction, details = self.model.sample(batch_size=self.n_samples, set_nodes=set_nodes)
                prediction = prediction.marginalize([self.outcome])
                states = DataBatch(prediction.states.values, (self.variable,))
                numeric = _apply_numeric_mapping(states)[:, 0].to(prediction.probabilities)
                value = (prediction.probabilities @ numeric).mean()
        return ScalarResult(value, "mean_potential_outcome", (self.variable,)), details


class ATE(SEM2Operator):
    input_type = OperatorInput.SELF
    output_type = OperatorOutput.SCALAR

    def __init__(self, cfg, model, variable: VariableInfo | None = None):
        self.model = model
        # Fixed 1-versus-0 treatment contrast, not a general multilevel effect.
        # Both queries use the same seed for paired exogenous Monte Carlo samples.
        self.control = MeanPotentialOutcome(model, outcome=cfg.Y_node, intervention={cfg.T_node: 0},
                                            n_samples=cfg.sample_bs, seed=cfg.rng_seed, variable=variable)
        self.treatment = MeanPotentialOutcome(model, outcome=cfg.Y_node, intervention={cfg.T_node: 1},
                                              n_samples=cfg.sample_bs, seed=cfg.rng_seed, variable=variable)

    def evaluate(self, **kwargs) -> tuple[ScalarResult, dict]:
        y0, control = self.control.evaluate()
        y1, treatment = self.treatment.evaluate()
        return ScalarResult(y1.value - y0.value, "ATE", y1.variables), {
            "control": control["interventions"], "treatment": treatment["interventions"]
        }

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
