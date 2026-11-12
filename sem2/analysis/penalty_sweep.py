"""Fresh-model fits across user-supplied signed target penalties."""
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite

import pandas as pd
import torch

from sem2.data.preparation import prepare_data
from sem2.data.records import VariableInfo
from sem2.default_configs import TrainingConfig
from sem2.model.models.base import SEM2
from sem2.operators.operators_base import OperatorInput, SEM2Operator
from sem2.operators.sem2_loss import SEM2Loss
from sem2.trainer.trainer import fit


@dataclass
class SweepResult:
    """One fitted model and its final, full-data evaluation in eval mode.

    ``objective`` is ``fit_loss + weight * target``. ``max_abs_gradient`` is
    the maximum absolute parameter gradient of that same final evaluation,
    excluding optimizer weight decay. It is not a convergence certificate.
    ``diagnostics`` retains SEM2Loss's component-prefixed details (top-level
    tensors detached), plus the data preparation report under ``preparation``.
    """

    weight: float
    fit_loss: float
    target: float
    objective: float
    max_abs_gradient: float
    diagnostics: dict
    model: SEM2


def penalty_sweep(
    factory: Callable[[], tuple[SEM2, SEM2Operator, SEM2Operator]],
    data: pd.DataFrame,
    weights: Sequence[float],
    config: TrainingConfig,
    *,
    variables: list[VariableInfo] | None = None,
) -> list[SweepResult]:
    """Minimize fit loss plus a signed target penalty for each supplied weight.

    Parameters
    ----------
    factory : callable
        Return a fresh model, a data-taking scalar fit loss to minimize, and a
        model-only scalar target. Both operators must use that model. The factory
        controls initialization; returning a previously fitted model is rejected.
    data : pandas.DataFrame
        Named observations, prepared with the existing complete-case policy.
    weights : sequence of float
        Finite signed weights, processed in the supplied order. Positive weights
        favor smaller targets, negative weights favor larger targets; zero fits
        observations alone. No weights, restarts or warm starts are added.
    config : TrainingConfig
        Existing training settings. rng_seed seeds training, not construction;
        seed explicitly in or before the factory to control initialization.
    variables : list[VariableInfo], optional
        Named metadata, including categorical encodings and numerical scores.

    Returns
    -------
    list[SweepResult]
        Final component values, diagnostics and fitted models in eval mode.
        Components are evaluated once together after fitting, with sample_bs
        from config. Monte Carlo diagnostics can therefore vary between runs.

    Notes
    -----
    Calls the existing fit function with a fresh optimizer for every model.
    Failures propagate rather than being silently skipped. Returned models are
    retained in memory. Numerical extrema are not certified identified bounds:
    assess observational fit, optimization and model assumptions separately.
    """
    weights = [float(weight) for weight in weights]
    if not weights or not all(isfinite(weight) for weight in weights):
        raise ValueError("weights must be a nonempty sequence of finite numbers.")
    results: list[SweepResult] = []
    for weight in weights:
        model, fit_loss, target = factory()
        if any(model is previous.model for previous in results):
            raise ValueError("factory must return a fresh model for each weight.")
        if fit_loss.input_type != OperatorInput.DATA or target.input_type != OperatorInput.SELF:
            raise TypeError("Expected a data-taking fit loss and a model-only target.")
        objective = SEM2Loss([(fit_loss, 1.), (target, weight)])
        fit(model, data, objective, config, variables=variables)

        model.eval()
        batch, preparation = prepare_data(model, data, variables=variables, device=config.device)
        final, details = objective.evaluate(batch, sample_bs=config.sample_bs)
        parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        gradients = torch.autograd.grad(final.value, parameters, allow_unused=True)
        maxima = [gradient.detach().abs().max() for gradient in gradients if gradient is not None]
        max_gradient = torch.stack(maxima).max().item() if maxima else 0.
        diagnostics = {key: value.detach() if isinstance(value, torch.Tensor) else value
                       for key, value in details.items()}
        diagnostics['preparation'] = preparation
        results.append(SweepResult(
            weight=weight,
            fit_loss=details[f'0:{type(fit_loss).__name__}'].item(),
            target=details[f'1:{type(target).__name__}'].item(),
            objective=final.value.detach().item(),
            max_abs_gradient=max_gradient,
            diagnostics=diagnostics,
            model=model,
        ))
    return results
