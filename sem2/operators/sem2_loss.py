"""Composition of scalar objectives, independent of their statistical purpose."""
from .operators_base import OperatorInput, OperatorOutput, SEM2Operator
from ..data.records import DataBatch, ScalarResult


class SEM2Loss(SEM2Operator):
    """Weighted sum of explicit (scalar operator, weight) pairs.

    Component conventions are unchanged: a negative log-likelihood weight gives
    a minimization objective. Diagnostics use index:class prefixes to avoid collisions.
    sample_bs overrides Monte Carlo counts and beta overrides energy exponents;
    components not using a setting leave their calculation unchanged.

    Parameters
    ----------
    components : list[tuple[SEM2Operator, float]]
        Nonempty sequence of scalar operators and weights. Data-taking and
        model-only components can be combined; weights are not normalized.
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

    def evaluate(self, observations: DataBatch, *, sample_bs: int | None = None,
                 beta: float | None = None) -> tuple[ScalarResult, dict]:
        """Evaluate the weighted sum without changing component conventions.

        Parameters
        ----------
        observations : DataBatch
            Batch for data-taking components; required even if all components
            are model-only.
        sample_bs : int, optional
            Monte Carlo override forwarded to each component.
        beta : float, optional
            Energy-exponent override forwarded to each component.

        Returns
        -------
        result : ScalarResult
            Weighted total with gradients and combined variable metadata.
        diagnostics : dict
            Unweighted detached component values and prefixed component details.
            Details retain each component's own gradient convention.
        """
        # Start with a Python zero so the components determine tensor dtype/device.
        total = 0
        metrics, variables = {}, {}
        for index, (component, weight) in enumerate(self.loss_components):
            if component.input_type == OperatorInput.DATA:
                result, details = component.evaluate(observations, sample_bs=sample_bs, beta=beta)
            else:
                result, details = component.evaluate(sample_bs=sample_bs, beta=beta)
            total = total + weight * result.value
            variables.update((v.name, v) for v in result.variables)
            label = f"{index}:{type(component).__name__}"
            metrics.update((f"{label}/{name}", value) for name, value in details.items())
            metrics[label] = result.value.detach()
        return ScalarResult(total, "loss", list(variables.values())), metrics
