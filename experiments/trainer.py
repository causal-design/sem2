"""Fit a separate Gaussian model by weighted covariance matching.

Run ``python -m experiments.trainer`` from the repository root after installing
SEM2. The true equations are x = u and y = 2*x + v, with independent standard
normal u and v. Only x and y are observed. The fitted model knows the graph,
zero intercepts and unit noise variances, but estimates the x-to-y slope.
Covariance alone cannot estimate intercepts, so both are explicitly fixed at zero.

WeightedFrobeniusDistance uses Covariance internally and minimizes the weighted
sum of squared covariance residuals, not a Gaussian likelihood. With diagonal
weights 1 and off-diagonal weights 1/2, each distinct covariance is counted once.
The reference minimizes this same one-dimensional objective independently in
NumPy. It is not OLS. Full batches match the covariance of the whole dataset.

The diagnostics describe this finite-sample example, not a general convergence
or identification guarantee. The IV smoke example is in experiments/iv_trainer.py.
"""
import numpy as np
import pandas as pd
import torch

from sem2.data.records import DataBatch
from sem2.default_configs import TrainingConfig
from sem2.model.models.linear import SEM2Linear
from sem2.operators.covariance import Covariance
from sem2.operators.divergence import WeightedFrobeniusDistance
from sem2.trainer.trainer import SEM2Trainer
from sem2.utils import set_seed


def _coefficients(model: SEM2Linear) -> pd.Series:
    """Read the fixed intercept and fitted slope by name, not parent order."""
    node = model.node_dict["y"]
    return pd.Series({
        "intercept (fixed)": node.intercept().detach().item(),
        "x -> y": node.effective_weight()[0, node.predecessors.index("x")].detach().item(),
    })


def _moments(model: SEM2Linear) -> pd.Series:
    """Return means and the named x/y block from the covariance operator."""
    means = model.mean_vector().detach()
    operator = Covariance(model)
    covariance, _ = operator.evaluate()
    indices = [operator.columns[name][0] for name in ["x", "y"]]
    covariance = covariance.detach()[indices][:, indices]
    return pd.Series({
        "mean(x)": means[model.node_order.index("x")].item(),
        "mean(y)": means[model.node_order.index("y")].item(),
        "var(x)": covariance[0, 0].item(),
        "cov(x,y)": covariance[0, 1].item(),
        "var(y)": covariance[1, 1].item(),
    })


def _reference_fit(sample_cov: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    """Minimize the scalar quartic independently, checking all stationary points.

    The model covariance is [[1, b], [b, b**2 + 1]]. The derivative divided by
    two is 2*Wyy*b**3 + (Wxy + Wyx + 2*Wyy*(1 - Syy))*b - (Wxy + Wyx)*Sxy.
    Positive Wyy makes the loss grow at either infinity, so a minimum is among
    the real roots. Compare their losses rather than selecting a root by sign.
    """
    cross_weight = weight[0, 1] + weight[1, 0]
    roots = np.roots([
        2 * weight[1, 1], 0.,
        cross_weight + 2 * weight[1, 1] * (1 - sample_cov[1, 1]),
        -cross_weight * sample_cov[0, 1],
    ])
    candidates = roots.real[np.abs(roots.imag) < 1e-8]
    losses = [
        np.sum(weight * (sample_cov - np.array([[1., b], [b, b**2 + 1.]]))**2)
        for b in candidates
    ]
    best = np.argmin(losses)
    return float(candidates[best]), float(losses[best])


def run_example(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Generate 512 observations, fit, and return comparisons and diagnostics."""
    set_seed(seed)
    truth = SEM2Linear("u,v ~ Normal(0,1); x <- 1*u + 0; y <- 2*x + 1*v + 0")
    with torch.no_grad():
        draws, _ = truth.sample(512)
    observations = draws.subset(["x", "y"])
    data = pd.DataFrame(observations.values.numpy(), columns=observations.names)

    # No true slope or latent draws are transferred to the separate fitted model.
    model = SEM2Linear("u,v ~ Normal(0,1); x <- 1*u + 0; y <- x + 1*v + 0")
    coefficients = pd.DataFrame({"true": _coefficients(truth), "initial": _coefficients(model)})
    moments = pd.DataFrame({"true": _moments(truth), "initial": _moments(model)})
    weight = torch.tensor([[1., 0.5], [0.5, 1.]])
    objective = WeightedFrobeniusDistance(model, observed_vars=["x", "y"], weight=weight)
    config = TrainingConfig(
        device="cpu", optim="SGD", lr=0.01, momentum=0., weight_decay=0.,
        epochs=200, batch_size=len(data), rng_seed=seed,
    )
    trainer = SEM2Trainer(config, model, data, objective)
    batch = DataBatch(trainer.loader.dataset.tensors[0], trainer.variables)
    initial_loss = objective.evaluate(batch)[0].value.detach().item()
    trainer.train()

    coefficients["fitted"] = _coefficients(model)
    moments["fitted"] = _moments(model)
    sample_cov = np.cov(data[["x", "y"]].to_numpy(dtype=float), rowvar=False, ddof=1)
    moments["sample"] = [data["x"].mean(), data["y"].mean(),
                         sample_cov[0, 0], sample_cov[0, 1], sample_cov[1, 1]]
    reference_slope, reference_loss = _reference_fit(sample_cov, weight.numpy())
    coefficients["reference"] = [0., reference_slope]

    # Evaluate after the final update on all observations, not the last batch log.
    final_loss = objective.evaluate(batch)[0].value
    gradients = torch.autograd.grad(final_loss, tuple(model.parameters()), allow_unused=True)
    diagnostics = {
        "initial weighted squared Frobenius loss": initial_loss,
        "fitted weighted squared Frobenius loss": final_loss.detach().item(),
        "independent minimum loss": reference_loss,
        "maximum absolute final gradient": max(
            gradient.abs().max().item() for gradient in gradients if gradient is not None
        ),
        "absolute fitted-minus-reference slope": abs(coefficients.loc["x -> y", "fitted"] - reference_slope),
    }
    return coefficients, moments, diagnostics


def main() -> None:
    coefficients, moments, diagnostics = run_example()
    print("Gaussian covariance fit: 512 observations, seed=0, CPU, 200 full-batch SGD updates.")
    print("Independent standard-normal noise; graph, zero intercepts and unit variances are known.")
    print("Weighted squared Frobenius objective: W = [[1, 0.5], [0.5, 1]], sample covariance uses n-1.")
    print("\nCoefficients (reference independently minimizes the same covariance loss, not OLS):")
    print(coefficients.to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nModel-implied moments and empirical moments (means are not fitted):")
    print(moments.to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nFull-data fit diagnostics:")
    for name, value in diagnostics.items():
        print(f"  {name}: {value:.6g}")
    print("\nReference agreement checks optimization; differences from truth include sampling error.")
    print("This example does not establish general convergence, identification, or IV validity.")


if __name__ == "__main__":
    main()
