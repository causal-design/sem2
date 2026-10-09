"""Sampling and scalar queries under model-implied interventions."""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from .operators_base import OperatorInput, OperatorOutput, SEM2Operator
from ..model.construction.sem_types import SEMType
from ..data.records import DataBatch, ScalarResult, VariableInfo
from ..data.numeric import numeric_values
from ..utils import set_seed
from ..default_configs import TrainingConfig

if TYPE_CHECKING:
    from ..model.models.base import SEM2


class PotentialOutcomeSample(SEM2Operator):
    """Sample a continuous model, stopping as soon as the target is computed."""
    input_type = OperatorInput.SELF
    output_type = OperatorOutput.DATA_BATCH

    def __init__(self, model: SEM2, target_node: str) -> None:
        if not SEMType.is_continuous(model.sem_type):
            raise NotImplementedError("PotentialOutcomeSample requires a continuous model.")
        if target_node not in model.input_node:
            raise ValueError(f"Unknown or internal target node: {target_node!r}")
        self.model = model
        self.target_node = target_node

    def evaluate(self, sample_bs: int = 1, set_nodes: DataBatch | None = None) -> tuple[DataBatch, dict]:
        return self.model._sample(sample_bs, set_nodes, target=self.target_node)


class MeanPotentialOutcome(SEM2Operator):
    """Mean of a scalar outcome under constant named interventions.

    sample_bs counts Monte Carlo samples; discrete outcomes are integrated over
    their states. By default, sampling uses the existing RNG stream. An explicit
    seed reseeds each evaluation; the previous RNG state is not restored.
    Supply variable metadata for an explicit category-to-number mapping.
    sample_bs overrides cfg.sample_bs; evaluate(sample_bs=...) overrides one call.
    beta is accepted for scalar-objective composition but does not affect this query.
    """
    input_type = OperatorInput.SELF
    output_type = OperatorOutput.SCALAR

    def __init__(self, model: SEM2, *, outcome: str, intervention: dict[str, float],
                 sample_bs: int | None = None, seed: int | None = None,
                 variable: VariableInfo | None = None, cfg: TrainingConfig | None = None) -> None:
        cfg = cfg if cfg is not None else TrainingConfig(device=model.device)
        self.model = model
        self.outcome = outcome
        self.intervention = dict(intervention)
        self.sample_bs = cfg.sample_bs if sample_bs is None else sample_bs
        self.seed = seed
        if outcome not in model.input_node or set(intervention) - model.input_node:
            raise ValueError("Query nodes must be user-declared model variables.")
        self.variable = variable if variable is not None else model._variable_info(outcome)
        if self.variable.name != outcome or self.variable.dim != 1 or model.node_dict[outcome].out_dim != 1:
            raise ValueError("MeanPotentialOutcome requires scalar outcome metadata matching the model.")
        if any(model.node_dict[name].out_dim != 1 for name in intervention):
            raise ValueError("MeanPotentialOutcome requires scalar intervention nodes.")
        if SEMType.is_continuous(model.sem_type):
            self.sampler = PotentialOutcomeSample(model, outcome)

    def evaluate(self, *, sample_bs: int | None = None,
                 beta: float | None = None) -> tuple[ScalarResult, dict]:
        """Evaluate the configured intervention, reseeding only if requested."""
        sample_bs = self.sample_bs if sample_bs is None else sample_bs
        parameter = next(self.model.parameters(), None)
        device = parameter.device if parameter is not None else torch.device(self.model.device)
        dtype = parameter.dtype if parameter is not None else torch.get_default_dtype()
        # Broadcast constant assignments once; samplers still consume named batches.
        values = [torch.as_tensor(value, device=device, dtype=dtype).expand(sample_bs, 1)
                  for value in self.intervention.values()]
        set_nodes = DataBatch(torch.cat(values, dim=1),
                              tuple(self.model._variable_info(n) for n in self.intervention)) if values else None

        set_seed(self.seed, device=device)
        if SEMType.is_continuous(self.model.sem_type):
            draws, details = self.sampler.evaluate(sample_bs=sample_bs, set_nodes=set_nodes)
            value = numeric_values(DataBatch(draws.values, (self.variable,))).mean()
        else:
            prediction, details = self.model.sample(batch_size=sample_bs, set_nodes=set_nodes)
            prediction = prediction.marginalize([self.outcome])
            states = DataBatch(prediction.states.values, (self.variable,))
            numeric = numeric_values(states)[:, 0].to(prediction.probabilities)
            value = (prediction.probabilities @ numeric).mean()
        return ScalarResult(value, "mean_potential_outcome", (self.variable,)), details


class ATE(SEM2Operator):
    """One-versus-zero treatment contrast with explicit graph roles.

    sample_bs overrides cfg.sample_bs; evaluate(sample_bs=...) affects one call.
    beta is accepted for scalar-objective composition but does not affect this contrast.
    """
    input_type = OperatorInput.SELF
    output_type = OperatorOutput.SCALAR

    def __init__(self, cfg: TrainingConfig, model: SEM2, variable: VariableInfo | None = None,
                 *, outcome: str, treatment: str, sample_bs: int | None = None) -> None:
        self.model = model
        # Fixed 1-versus-0 treatment contrast, not a general multilevel effect.
        # An explicit seed pairs the exogenous draws; otherwise use the current stream.
        self.control = MeanPotentialOutcome(model, outcome=outcome, intervention={treatment: 0},
                                            sample_bs=sample_bs, seed=cfg.rng_seed, variable=variable, cfg=cfg)
        self.treatment = MeanPotentialOutcome(model, outcome=outcome, intervention={treatment: 1},
                                              sample_bs=sample_bs, seed=cfg.rng_seed, variable=variable, cfg=cfg)

    def evaluate(self, *, sample_bs: int | None = None,
                 beta: float | None = None) -> tuple[ScalarResult, dict]:
        y0, control = self.control.evaluate(sample_bs=sample_bs)
        y1, treatment = self.treatment.evaluate(sample_bs=sample_bs)
        return ScalarResult(y1.value - y0.value, "ATE", y1.variables), {
            "control": control["interventions"], "treatment": treatment["interventions"]
        }

    def __repr__(self) -> str:
        return "ATE"
