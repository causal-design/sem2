"""Fit a separate Gaussian model to observations from a known true model.

Run ``python -m experiments.trainer`` from the repository root after installing
SEM2. The true equations are x = u and y = 1 + 2*x + v, with independent standard
normal u and v. Only x and y are observed. The fitted model knows the graph and
unit noise variances, but estimates the intercept and x-to-y slope from scratch.
Its exact Gaussian maximum-likelihood solution is ordinary least squares (OLS).

The printed diagnostics describe this finite-sample example, not a general
convergence guarantee or IV validation. The previous fixture-based IV smoke
example is retained in ``experiments/iv_trainer.py``.
"""
import numpy as np
import pandas as pd
import torch

from sem2.data.records import DataBatch
from sem2.default_configs import TrainingConfig
from sem2.model.models.linear import SEM2Linear
from sem2.operators.linear import LogLikelihood
from sem2.operators.sem2_loss import SEM2Loss
from sem2.trainer.trainer import SEM2Trainer
from sem2.utils import set_seed


def _coefficients(model: SEM2Linear) -> pd.Series:
    """Read the intercept and slope by name, not by assumed parent order."""
    node = model.node_dict["y"]
    return pd.Series({
        "intercept": node.intercept().detach().item(),
        "x -> y": node.effective_weight()[0, node.predecessors.index("x")].detach().item(),
    })


def _moments(model: SEM2Linear) -> pd.Series:
    """Return analytic population moments for the observed variables x and y."""
    means = model.mean_vector().detach()
    covariance = model.cov_matrix(["x", "y"]).detach()
    return pd.Series({
        "mean(x)": means[model.node_order.index("x")].item(),
        "mean(y)": means[model.node_order.index("y")].item(),
        "var(x)": covariance[0, 0].item(),
        "cov(x,y)": covariance[0, 1].item(),
        "var(y)": covariance[1, 1].item(),
    })


def run_example(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Generate 512 observations, fit, and return comparison tables and diagnostics."""
    set_seed(seed)
    truth = SEM2Linear("u,v ~ Normal(0,1); x <- 1*u + 0; y <- 2*x + 1*v + 1")
    with torch.no_grad():
        draws, _ = truth.sample(512)
    observations = draws.subset(["x", "y"])
    data = pd.DataFrame(observations.values.numpy(), columns=observations.names)

    # No parameters or latent draws are transferred from truth to the fitted model.
    model = SEM2Linear("u,v ~ Normal(0,1); x <- 1*u + 0; y <- x + 1*v")
    coefficients = pd.DataFrame({"true": _coefficients(truth), "initial": _coefficients(model)})
    moments = pd.DataFrame({"true": _moments(truth), "initial": _moments(model)})
    objective = SEM2Loss([(LogLikelihood(model), -1.)])
    config = TrainingConfig(
        device="cpu", optim="SGD", lr=0.1, momentum=0., weight_decay=0.,
        epochs=200, batch_size=len(data), rng_seed=seed,
    )
    trainer = SEM2Trainer(config, model, data, objective)
    batch = DataBatch(trainer.loader.dataset.tensors[0], trainer.variables)
    initial_loss = objective.evaluate(batch)[0].value.detach().item()
    trainer.train()

    coefficients["fitted"] = _coefficients(model)
    moments["fitted"] = _moments(model)
    # Independent reference: the likelihood reduces to least squares in y given x.
    design = np.column_stack([np.ones(len(data)), data["x"].to_numpy(dtype=float)])
    coefficients["OLS"] = np.linalg.lstsq(design, data["y"].to_numpy(dtype=float), rcond=None)[0]

    # Evaluate after the final update on all observations, not the last batch log.
    final_loss = objective.evaluate(batch)[0].value
    gradients = torch.autograd.grad(final_loss, tuple(model.parameters()), allow_unused=True)
    diagnostics = {
        "initial mean negative log-likelihood": initial_loss,
        "fitted mean negative log-likelihood": final_loss.detach().item(),
        "maximum absolute final gradient": max(
            gradient.abs().max().item() for gradient in gradients if gradient is not None
        ),
        "maximum absolute fitted-minus-OLS coefficient": (
            coefficients["fitted"] - coefficients["OLS"]
        ).abs().max().item(),
    }
    return coefficients, moments, diagnostics


def main() -> None:
    coefficients, moments, diagnostics = run_example()
    print("Gaussian fit: 512 observations, seed=0, CPU, 200 full-batch SGD updates.")
    print("Independent standard-normal noise; graph and unit variances are known.")
    print("\nCoefficients (OLS uses the same observed sample):")
    print(coefficients.to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nModel-implied population moments:")
    print(moments.to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nFull-data fit diagnostics:")
    for name, value in diagnostics.items():
        print(f"  {name}: {value:.6g}")
    print("\nOLS agreement checks optimization; differences from truth include sampling error.")
    print("This example does not establish general convergence, identification, or IV validity.")


if __name__ == "__main__":
    main()
