"""Population means and contrasts under fixed model-implied interventions."""
from __future__ import annotations
from copy import deepcopy
from math import isfinite
from typing import TYPE_CHECKING
import warnings
import torch
from .operators_base import OperatorInput, OperatorOutput, SEM2Operator
from ..model.construction.sem_types import SEMType
from ..data.records import DataBatch, ScalarResult, VariableInfo
from ..data.conversion import databatch_to_tensor
from ..utils import set_seed

if TYPE_CHECKING:
    from ..model.models.base import SEM2


def _fixed_intervention(model: SEM2, assignments: dict[str, float]) -> DataBatch | None:
    """Validate scalar assignments once, before casting to the sampler dtype."""
    variables = []
    for name, value in assignments.items():
        if name not in model.input_node:
            raise ValueError(f"Unknown or internal intervention node: {name!r}")
        node = model.node_dict[name]
        if node.out_dim != 1 or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError(f"Intervention on {name!r} requires a finite scalar number.")
        support = node.support
        if support.values is not None:
            # Compare Python values before float32 rounding or integer conversion.
            if support.values.detach().cpu().flatten().tolist().count(value) != 1:
                raise ValueError(f"Intervention {name}={value} is outside its declared states.")
        elif support.constraint is not None:
            if support.constraint.is_discrete and value != int(value):
                raise ValueError(f"Intervention {name}={value} requires an integer state.")
            candidate = torch.tensor(value, dtype=torch.float64, device=model.device)
            if not support.constraint.check(candidate).all():
                raise ValueError(f"Intervention {name}={value} is outside its declared support.")
        variables.append(model._variable_info(name))
    if not variables:
        return None
    parameter = next(model.parameters(), None)
    dtype = parameter.dtype if parameter is not None else torch.get_default_dtype()
    values = torch.tensor([list(assignments.values())], device=model.device, dtype=dtype)
    if not torch.isfinite(values).all():
        raise ValueError("Interventions must be finite in the model's floating dtype.")
    return DataBatch(values, variables)


class PotentialOutcomeSample(SEM2Operator):
    """Sample a continuous model, stopping as soon as the target is computed.

    Parameters
    ----------
    model : SEM2
        Continuous model to sample, using its current train/eval mode.
    target_node : str
        User-declared target variable; internal helpers cannot be selected.
    """
    input_type = OperatorInput.SELF
    output_type = OperatorOutput.DATA_BATCH

    def __init__(self, model: SEM2, target_node: str) -> None:
        if not SEMType.is_continuous(model.sem_type):
            raise NotImplementedError("PotentialOutcomeSample requires a continuous model.")
        if target_node not in model.input_node:
            raise ValueError(f"Unknown or internal target node: {target_node!r}")
        self.model = model
        self.target_node = target_node

    def evaluate(self, sample_bs: int = 1,
                 do_intervention: DataBatch | None = None) -> tuple[DataBatch, dict]:
        """Return target draws under optional, already-valid interventions.

        Parameters
        ----------
        sample_bs : int, default=1
            Positive number of draws.
        do_intervention : DataBatch, optional
            Valid stored intervention values, one row per draw, on the model device.

        Returns
        -------
        draws : DataBatch
            Target values of shape (sample_bs, target_width).
        details : dict
            ``interventions`` records the supplied intervention batch.
        """
        return self.model._sample(sample_bs, do_intervention, target=self.target_node)


class MeanPotentialOutcome(SEM2Operator):
    """Population mean of a scalar outcome under fixed named interventions.

    Parameters
    ----------
    model : SEM2
        Model supplying the intervention law, not an identification guarantee.
    outcome : str
        User-declared scalar outcome.
    intervention : dict[str, float]
        Fixed stored values, not numerical outcome scores. Omitted nodes follow
        their equations; an empty dictionary is the natural regime.
    sample_bs : int, default=512
        Monte Carlo draw count, independent of training configuration.
    seed : int, optional
        Explicitly reseed evaluation; None continues the current RNG stream.
    variable : VariableInfo, optional
        Outcome metadata. Categorical outcomes require a complete numeric_mapping;
        without one, construction warns and raises ValueError. An explicit 0/1
        event score gives a probability rather than a mean of arbitrary codes.

    Notes
    -----
    Evaluation temporarily uses model.eval() but retains autograd and restores
    every module's previous training flag. Parameters are unchanged; cached node
    values and RNG state may change. Discrete outcome states are probability
    weighted, with Monte Carlo integration over remaining exogenous draws.
    """
    input_type = OperatorInput.SELF
    output_type = OperatorOutput.SCALAR

    def __init__(self, model: SEM2, *, outcome: str, intervention: dict[str, float],
                 sample_bs: int = 512, seed: int | None = None,
                 variable: VariableInfo | None = None) -> None:
        if outcome not in model.input_node:
            raise ValueError(f"Unknown or internal outcome: {outcome!r}")
        self.model = model
        self.outcome = outcome
        self.intervention = dict(intervention)
        self.sample_bs = sample_bs
        self.seed = seed
        declared = model._variable_info(outcome)
        self.variable = deepcopy(variable if variable is not None else declared)
        if self.variable.name != outcome or self.variable.dim != 1 or declared.dim != 1:
            raise ValueError("MeanPotentialOutcome requires scalar outcome metadata matching the model.")
        if declared.categorical or self.variable.categorical:
            self.variable.categorical = True
            if self.variable.numeric_mapping is None:
                message = (f"Categorical outcome {outcome!r} requires explicit numerical scores; "
                           "supply a numeric_mapping, including an event-indicator score for probabilities.")
                warnings.warn(message, stacklevel=2)
                raise ValueError(message)
        if self.variable.numeric_mapping is not None:
            scores, encoding = self.variable.numeric_mapping, self.variable.encoding
            support = model.node_dict[outcome].support.values
            if encoding is not None and set(scores) != set(encoding):
                raise ValueError("Outcome scores must cover every encoded label.")
            codes = [encoding[label] if encoding is not None else label for label in scores]
            if (support is None or len(set(codes)) != len(codes)
                    or set(codes) != set(support.detach().cpu().flatten().tolist())
                    or not all(isfinite(score) for score in scores.values())):
                raise ValueError("Outcome scores must be finite and cover each declared state exactly once.")
        self._intervention = _fixed_intervention(model, self.intervention)
        if SEMType.is_continuous(model.sem_type):
            self.sampler = PotentialOutcomeSample(model, outcome)
            self._evaluate_model = self._evaluate_continuous
            self.method = "monte_carlo"
        else:
            self._evaluate_model = self._evaluate_discrete
            self.method = ("probability_weighted" if outcome in model.state_node_names else "monte_carlo")

    def evaluate(self, *, sample_bs: int | None = None, seed: int | None = None,
                 beta: float | None = None) -> tuple[ScalarResult, dict]:
        """Evaluate without fitting or changing stored query settings.

        Parameters
        ----------
        sample_bs : int, optional
            Draw count for this call; None uses the constructor default.
        seed : int, optional
            Seed for this call; None uses the constructor default. Explicit
            seeds reset the stream without restoring its previous state.
        beta : float, optional
            Accepted for scalar-objective composition; unused.

        Returns
        -------
        result : ScalarResult
            Mean outcome with gradients, labelled ``mean_potential_outcome``.
        details : dict
            Outcome, fixed intervention, integration method, sample_bs, seed
            and evaluation_mode. No uncertainty or identification claim.
        """
        sample_bs = self.sample_bs if sample_bs is None else sample_bs
        seed = self.seed if seed is None else seed
        intervention = None
        if self._intervention is not None:
            intervention = DataBatch(self._intervention.values.expand(sample_bs, -1),
                                     self._intervention.variables)
        modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            set_seed(seed, device=self.model.device)
            value = self._evaluate_model(sample_bs, intervention)
        finally:
            for module, training in modes:
                module.training = training
        return ScalarResult(value, "mean_potential_outcome", [self.variable]), {
            "outcome": self.outcome, "intervention": dict(self.intervention),
            "method": self.method, "sample_bs": sample_bs, "seed": seed, "evaluation_mode": "eval",
        }

    def _evaluate_continuous(self, sample_bs: int, intervention: DataBatch | None) -> torch.Tensor:
        """Average target-only draws, retaining shared sampler early stopping."""
        draws, _ = self.sampler.evaluate(sample_bs=sample_bs, do_intervention=intervention)
        return databatch_to_tensor(DataBatch(draws.values, [self.variable])).mean()

    def _evaluate_discrete(self, sample_bs: int, intervention: DataBatch | None) -> torch.Tensor:
        """Integrate outcome states, or average draws of an exogenous outcome."""
        prediction, _ = self.model.sample(batch_size=sample_bs, do_intervention=intervention)
        if self.outcome not in self.model.state_node_names:
            draws = DataBatch(self.model.node_dict[self.outcome].get_value(), [self.variable])
            return databatch_to_tensor(draws).mean()
        prediction = prediction.marginalize([self.outcome])
        states = prediction.states.values.to(prediction.probabilities)
        numeric = databatch_to_tensor(DataBatch(states, [self.variable]))[:, 0]
        return (prediction.probabilities @ numeric).mean()


class ATE(SEM2Operator):
    """Alternative-minus-reference population mean contrast.

    Parameters
    ----------
    model : SEM2
        Model supplying both intervention laws.
    outcome : str
        User-declared scalar outcome.
    alternative, reference : dict[str, float]
        Complete fixed regimes under the omission rule: missing nodes follow
        their equations, not the other regime's assignments. Supports joint and
        nonbinary interventions. An empty dictionary denotes the natural regime.
    sample_bs : int, default=512
        Monte Carlo count per regime.
    seed : int, optional
        Reseed once before both regimes. Draws consume the stream sequentially;
        they are not guaranteed paired across regimes.
    variable : VariableInfo, optional
        Scalar outcome metadata and explicit categorical scores, as for
        MeanPotentialOutcome.

    Notes
    -----
    This is a population contrast under the supplied model, not ATT or an
    identification check. Evaluation uses eval mode, restores previous module
    modes, and retains gradients for composition with a fitting objective.
    """
    input_type = OperatorInput.SELF
    output_type = OperatorOutput.SCALAR

    def __init__(self, model: SEM2, *, outcome: str, alternative: dict[str, float],
                 reference: dict[str, float], sample_bs: int = 512,
                 seed: int | None = None, variable: VariableInfo | None = None) -> None:
        self.model = model
        self.seed = seed
        self.reference = MeanPotentialOutcome(model, outcome=outcome, intervention=reference,
                                             sample_bs=sample_bs, variable=variable)
        self.alternative = MeanPotentialOutcome(model, outcome=outcome, intervention=alternative,
                                               sample_bs=sample_bs, variable=variable)

    def evaluate(self, *, sample_bs: int | None = None, seed: int | None = None,
                 beta: float | None = None) -> tuple[ScalarResult, dict]:
        """Return the alternative mean minus the reference mean.

        Parameters
        ----------
        sample_bs : int, optional
            Draw count per regime for this call.
        seed : int, optional
            One-time seed override; None uses the constructor default.
        beta : float, optional
            Accepted for scalar-objective composition; unused.

        Returns
        -------
        result : ScalarResult
            Difference of means, labelled ``ATE``, retaining gradients.
        details : dict
            Outcome, both assignments, detached component means, integration
            methods, sample_bs, seed and evaluation_mode.
        """
        seed = self.seed if seed is None else seed
        set_seed(seed, device=self.model.device)
        y0, reference = self.reference.evaluate(sample_bs=sample_bs)
        y1, alternative = self.alternative.evaluate(sample_bs=sample_bs)
        return ScalarResult(y1.value - y0.value, "ATE", y1.variables), {
            "outcome": alternative["outcome"],
            "alternative": alternative["intervention"], "reference": reference["intervention"],
            "alternative_mean": y1.value.detach(), "reference_mean": y0.value.detach(),
            "method": alternative["method"], "sample_bs": alternative["sample_bs"],
            "seed": seed, "evaluation_mode": "eval",
        }

    def __repr__(self) -> str:
        return "ATE"
