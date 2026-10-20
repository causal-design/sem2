"""Fit PoliticalDemocracy with the package trainer and compare with lavaan.

Run ``python -m experiments.political_democracy --epochs 2000 --seed 0`` from
an installed repository checkout. This is a numerical experiment, not a test:
coefficient differences and fit diagnostics are reported without a tolerance.
The reference is https://lavaan.ugent.be/tutorial/sem.html (unstandardized ML).
"""
import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from sem2.data.preparation import prepare_data
from sem2.default_configs import TrainingConfig
from sem2.model.models.linear import SEM2Linear
from sem2.operators.linear import LogLikelihood
from sem2.operators.sem2_loss import SEM2Loss
from sem2.trainer.trainer import fit
from sem2.utils import set_seed

DATA = Path(__file__).resolve().parents[1] / "data" / "PoliticalDemocracy.csv"
OBSERVED = ["x1", "x2", "x3", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8"]
LAVAAN = {"dem60~ind60": 1.483, "dem65~ind60": 0.572, "dem65~dem60": 0.837}

# Shared independent noises encode the six residual-covariance pairs. Each also
# contributes to both residual variances; this is not an unrestricted covariance
# parameterization. All offsets are zero because observations are centered.
FORMULA = (
    "ex1,ex2,ex3,ey1,ey2,ey3,ey4,ey5,ey6,ey7,ey8,edem60,edem65 ~ Normal(0,1);"
    "e15,e24,e26,e37,e48,e68 ~ Normal(0,1);"
    "ind60 ~ Normal(0,1); dem60 <- ind60 + edem60 + 0; dem65 <- ind60 + dem60 + edem65 + 0;"
    "x1 <- ind60 + ex1 + 0; x2 <- ind60 + ex2 + 0; x3 <- ind60 + ex3 + 0;"
    "y1 <- dem60 + ey1 + e15 + 0; y2 <- dem60 + ey2 + e24 + e26 + 0;"
    "y3 <- dem60 + ey3 + e37 + 0; y4 <- dem60 + ey4 + e24 + e48 + 0;"
    "y5 <- dem65 + ey5 + e15 + 0; y6 <- dem65 + ey6 + e26 + e68 + 0;"
    "y7 <- dem65 + ey7 + e37 + 0; y8 <- dem65 + ey8 + e48 + e68 + 0"
)


def load_data() -> pd.DataFrame:
    """Load the bundled observed columns and center them without rescaling."""
    data = pd.read_csv(DATA).loc[:, OBSERVED]
    return data - data.mean()


def _structural_coefs(model: SEM2Linear) -> dict[str, float]:
    """Express regressions on lavaan's unit-first-loading latent scales.

    For a path from one latent to another, multiply the fitted coefficient by
    the outcome latent's reference loading and divide by the predictor latent's
    reference loading. Unit noise variances do not imply unit latent variances.
    """
    coefficients = model.coef_matrix().detach()
    index = {name: i for i, name in enumerate(model.node_order)}
    loading = {
        latent: coefficients[index[indicator], index[latent]]
        for latent, indicator in [("ind60", "x1"), ("dem60", "y1"), ("dem65", "y5")]
    }
    return {
        name: (coefficients[index[outcome], index[predictor]]
               * loading[outcome] / loading[predictor]).item()
        for name in LAVAAN
        for outcome, predictor in [name.split("~")]
    }


def run_example(epochs: int = 2000, seed: int = 0, lr: float = 0.05
                ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Return scaled coefficients, observed covariances and final-fit diagnostics.

    Fit on CPU with one full-data Adam update per epoch. The explicit seed sets
    initialization and loader randomness; no model draws enter the likelihood.
    Diagnostics are evaluated after the last update, outside the training loop.
    """
    set_seed(seed, device="cpu")
    data = load_data()
    model = SEM2Linear(FORMULA, device="cpu")
    likelihood = LogLikelihood(model, observed_vars=OBSERVED, centered=True)
    objective = SEM2Loss([(likelihood, -1.)])
    config = TrainingConfig(device="cpu", optim="Adam", lr=lr, weight_decay=0.,
                            epochs=epochs, batch_size=len(data), rng_seed=seed)
    batch, report = prepare_data(model, data)
    initial_loss = objective.evaluate(batch)[0].value.detach().item()
    start = perf_counter()
    fit(model, data, objective, config)
    elapsed = perf_counter() - start

    final_loss = objective.evaluate(batch)[0].value
    gradients = torch.autograd.grad(final_loss, tuple(model.parameters()))
    coefficients = pd.DataFrame({"SEM2": _structural_coefs(model), "lavaan": LAVAAN})
    coefficients["difference"] = coefficients["SEM2"] - coefficients["lavaan"]

    # Gaussian ML uses the empirical covariance with divisor n, not n - 1.
    values = batch.select(OBSERVED)
    empirical = values.T @ values / len(values)
    fitted = model.cov_matrix(OBSERVED).detach()
    rows, columns = torch.triu_indices(len(OBSERVED), len(OBSERVED))
    pairs = pd.MultiIndex.from_tuples(
        [(OBSERVED[i], OBSERVED[j]) for i, j in zip(rows.tolist(), columns.tolist())],
        names=["variable", "with"],
    )
    covariances = pd.DataFrame({"empirical (divisor n)": empirical[rows, columns].numpy(),
                               "SEM2": fitted[rows, columns].numpy()}, index=pairs)
    covariances["difference"] = covariances["SEM2"] - covariances["empirical (divisor n)"]
    diagnostics = {
        "observations": report["retained_rows"],
        "dropped rows": report["dropped_rows"],
        "full-batch updates": epochs,
        "fit seconds": elapsed,
        "initial mean negative log-likelihood": initial_loss,
        "final mean negative log-likelihood": final_loss.detach().item(),
        "maximum absolute final gradient": max(g.abs().max().item() for g in gradients),
        "covariance RMSE (full matrix)": (fitted - empirical).square().mean().sqrt().item(),
        "maximum absolute covariance residual": (fitted - empirical).abs().max().item(),
    }
    return coefficients, covariances, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()
    torch.set_num_threads(1)  # Small CPU matrices; no import-time thread changes.
    coefficients, _, diagnostics = run_example(args.epochs, args.seed, args.lr)
    print(f"PoliticalDemocracy: CPU, seed={args.seed}, Adam lr={args.lr}.")
    print("Centered observations; analytical Gaussian likelihood; no model sampling.")
    print("\nStructural coefficients on lavaan's unit-first-loading scales:")
    print(coefficients.to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nFull-data diagnostics after the last update:")
    for name, value in diagnostics.items():
        print(f"  {name}: {value:.6g}")
    print("\nBudget completed, not a convergence claim. No coefficient tolerance is imposed.")
    print("lavaan values are rounded reference fits, not known causal effects.")
    print("Shared residual noises do not establish equivalence to every lavaan covariance model.")


if __name__ == "__main__":
    main()
