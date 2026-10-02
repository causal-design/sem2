"""Operators returning named draws."""
from __future__ import annotations
from .operators_base import SEM2Operator, OperatorInput, OperatorOutput
from ..model.construction import SEMType
from ..dataloader.data import DataBatch
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

    def evaluate(self, batch_size: int = 1, set_nodes: DataBatch | None = None) -> tuple[DataBatch, dict]:
        return self.model._sample(batch_size, set_nodes, target=self.target_node)
