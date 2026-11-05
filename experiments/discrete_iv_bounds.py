"""Binary-IV penalty sweeps compared with response-type linear-program bounds.

Assumptions: randomized binary Z, exclusion of Z from Y's equation, binary A/Y,
and unrestricted dependence among (A(0), A(1), Y(0), Y(1)). No monotonicity.
The fitted model learns masses on all response types, not generating parameters.
"""
from argparse import ArgumentParser
from itertools import product

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linprog
from torch import nn

from sem2.analysis.penalty_sweep import penalty_sweep
from sem2.data.records import VariableInfo
from sem2.default_configs import MLPConfig, TrainingConfig
from sem2.model.models.discrete import SEM2Discrete
from sem2.operators.discrete_iv import DiscreteIVEnergyLoss
from sem2.operators.potential_outcomes import ATE
from sem2.utils import set_seed


class _ResponseMechanism(nn.Module):
    """Fixed binary response table, producing logits in declared parent order."""

    def __init__(self, responses: torch.Tensor, selector_column: int, latent_column: int):
        super().__init__()
        values = responses.flatten().float()
        logits = torch.stack((1 - values, values), dim=1).log()
        self.table = nn.Embedding.from_pretrained(logits, freeze=True)
        self.selector_column = selector_column
        self.latent_column = latent_column

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        index = 2 * inputs[..., self.latent_column].long() + inputs[..., self.selector_column].long()
        return self.table(index)


def build_model() -> SEM2Discrete:
    """Saturated binary-IV model: learn all latent response-type probabilities."""
    model = SEM2Discrete(
        'zero ~ Bernoulli(probs=0); Z ~ Bernoulli(probs=0.5); '
        'U <- d(zero; levels=16); A <- d(Z,U; levels=2); Y <- d(A,U; levels=2)',
        node_cfgs={name: MLPConfig(hidden_dims=[]) for name in ['U', 'A', 'Y']},
        device='cpu',
    )
    responses = torch.tensor(list(product([0, 1], repeat=4)))
    for name, selector, table in [('A', 'Z', responses[:, :2]), ('Y', 'A', responses[:, 2:])]:
        node = model.node_dict[name]
        parents = list(node.predecessors)
        node.nn_module = _ResponseMechanism(table, parents.index(selector), parents.index('U'))
    return model


def reference_bounds(conditional: np.ndarray) -> tuple[float, float]:
    """Sharp ATE limits via a separate NumPy/SciPy response-type linear program.

    ``conditional[z, a, y]`` is P(A=a,Y=y | Z=z). Equality constraints enforce
    those probabilities exactly; infeasible empirical IV tables raise an error.
    These are identification bounds for that table, not confidence intervals.
    """
    types = list(product([0, 1], repeat=4))
    matrix = np.zeros((8, 16))
    effect = np.zeros(16)
    for column, (a0, a1, y0, y1) in enumerate(types):
        effect[column] = y1 - y0
        for z, a in enumerate((a0, a1)):
            y = (y0, y1)[a]
            matrix[4*z + 2*a + y, column] = 1
    lower = linprog(effect, A_eq=matrix, b_eq=conditional.reshape(-1), bounds=(0, None), method='highs')
    upper = linprog(-effect, A_eq=matrix, b_eq=conditional.reshape(-1), bounds=(0, None), method='highs')
    if not lower.success or not upper.success:
        raise ValueError(f'IV reference program failed: {lower.message}; {upper.message}')
    return float(lower.fun), float(-upper.fun)


def simulate(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Generate observations and independently enumerate the population law."""
    rng = np.random.default_rng(seed)
    types = np.array(list(product([0, 1], repeat=4)))
    masses = np.arange(1, 17, dtype=float)
    masses /= masses.sum()
    population = np.zeros((2, 2, 2))
    for mass, (a0, a1, y0, y1) in zip(masses, types):
        for z, a in enumerate((a0, a1)):
            population[z, a, (y0, y1)[a]] += mass
    latent = types[rng.choice(len(types), size=n, p=masses)]
    z = rng.binomial(1, 0.5, size=n)
    a = latent[np.arange(n), z]
    y = latent[np.arange(n), 2 + a]
    truth = float(masses @ (types[:, 3] - types[:, 2]))
    return pd.DataFrame({'Z': z, 'A': a, 'Y': y}), population, truth


def run_example(
    *, n: int = 2048, epochs: int = 1500, seed: int = 7,
    weights: tuple[float, ...] = (0.03, 0.003, -0.03, -0.003),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit fresh models; return sweep diagnostics and empirical/population bounds."""
    data, population, truth = simulate(n, seed)
    empirical = np.zeros((2, 2, 2))
    np.add.at(empirical, tuple(data[['Z', 'A', 'Y']].to_numpy().T), 1)
    counts = empirical.sum(axis=(1, 2))
    if np.any(counts < 2):
        raise ValueError('At least two observations are required at each instrument level.')
    empirical /= counts[:, None, None]
    reference = pd.DataFrame(
        [reference_bounds(population), reference_bounds(empirical)],
        index=['population', 'empirical'], columns=['lower', 'upper'],
    )
    reference['generating_ATE'] = truth
    variables = [VariableInfo(name, categorical=True, numeric_mapping={0: 0., 1: 1.})
                 for name in ['A', 'Y']]
    config = TrainingConfig(device='cpu', optim='Adam', lr=0.05, epochs=epochs,
                            batch_size=len(data), sample_bs=1, energy_score_unbiased=False)

    def factory() -> tuple[SEM2Discrete, DiscreteIVEnergyLoss, ATE]:
        model = build_model()
        loss = DiscreteIVEnergyLoss(config, model, 'Z')
        target = ATE(model, outcome='Y', alternative={'A': 1}, reference={'A': 0},
                     sample_bs=1, variable=variables[-1])
        return model, loss, target

    # One explicit seed before the sweep: successive models start afresh, without
    # resetting to the same parameters or copying a previous fitted endpoint.
    set_seed(seed, device='cpu')
    results = penalty_sweep(factory, data, weights, config, variables=variables)
    rows = []
    for result in results:
        endpoint = (reference.loc['empirical', 'lower' if result.weight > 0 else 'upper']
                    if result.weight else float('nan'))
        rows.append({
            'weight': result.weight, 'ATE': result.target, 'fit_loss': result.fit_loss,
            'objective': result.objective, 'max_gradient': result.max_abs_gradient,
            'total_variation': result.diagnostics['0:DiscreteIVEnergyLoss/total_variation'].item(),
            'max_cell_error': result.diagnostics['0:DiscreteIVEnergyLoss/max_cell_error'].item(),
            'reference_endpoint': endpoint, 'endpoint_difference': result.target - endpoint,
        })
    return pd.DataFrame(rows), reference


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    sweep, bounds = run_example(n=args.n, epochs=args.epochs, seed=args.seed)
    print('Reference identification intervals (not confidence intervals):')
    print(bounds.to_string(float_format=lambda value: f'{value:.6f}'))
    print('\nPenalized estimates (nonzero penalties allow observational misfit):')
    print(sweep.to_string(index=False, float_format=lambda value: f'{value:.6g}'))
