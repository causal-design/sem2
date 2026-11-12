"""Two-stage recovery from observed data, separate from known-mechanism tests.

The fit knows the graph and binary supports, not the generating probabilities.
Only A1, L, A2 and Y are observed; U is integrated out. A fixed 100-update budget
keeps this a small regression, not a simulation study or a convergence guarantee.
"""
from itertools import product

import numpy as np
import pandas as pd
import pytest
import torch

from sem2.data.preparation import prepare_data
from sem2.data.records import VariableInfo
from sem2.default_configs import MLPConfig, TrainingConfig
from sem2.model.models.discrete import SEM2Discrete
from sem2.operators.divergence import EnergyLoss
from sem2.operators.potential_outcomes import ATE, MeanPotentialOutcome
from sem2.trainer.trainer import fit
from sem2.utils import set_seed

OBSERVED = ['A1', 'L', 'A2', 'Y']
REGIMES = list(product([0, 1], repeat=2))
TRUE_MEANS = np.array([[0.22, 0.42], [0.38, 0.68]])  # Rows A1, columns A2.


def _simulate_two_stage(n: int, seed: int) -> pd.DataFrame:
    """Draw fresh Bernoulli innovations, then discard the latent U."""
    rng = np.random.default_rng(seed)
    u = rng.binomial(1, 0.5, n)
    a1 = rng.binomial(1, 0.5, n)
    l = rng.binomial(1, 0.2 + 0.3*a1 + 0.3*u)
    a2 = rng.binomial(1, 0.25 + 0.5*l)
    y = rng.binomial(1, 0.1 + 0.1*a1 + 0.2*a2 + 0.2*l + 0.1*u + 0.1*a1*a2)
    return pd.DataFrame({'A1': a1, 'L': l, 'A2': a2, 'Y': y})


def _g_formula(joint: np.ndarray) -> np.ndarray:
    """Sum P(Y=1 | A1,L,A2) P(L | A1); joint axes are A1,L,A2,Y."""
    means = np.empty((2, 2))
    for a1 in (0, 1):
        p_l = joint[a1].sum(axis=(1, 2)) / joint[a1].sum()
        for a2 in (0, 1):
            p_y = joint[a1, :, a2, 1] / joint[a1, :, a2, :].sum(axis=1)
            means[a1, a2] = p_l @ p_y
    return means


def _build_model() -> SEM2Discrete:
    """Learn conditional logits on fixed binary-context indicator features."""
    # A constant input makes the root probabilities trainable without introducing
    # another random common cause. All five binary nodes are enumerated exactly.
    model = SEM2Discrete(
        'zero ~ Bernoulli(probs=0); U <- d(zero; levels=2); A1 <- d(zero; levels=2); '
        'L <- d(A1,U; levels=2); A2 <- d(L; levels=2); Y <- d(A1,A2,L,U; levels=2)',
        node_cfgs={name: MLPConfig(hidden_dims=widths) for name, widths in
                   [('U', []), ('A1', []), ('L', [4]), ('A2', []), ('Y', [16])]},
        device='cpu',
    )
    for name in ['L', 'Y']:
        node = model.node_dict[name]
        states = torch.tensor(list(product([0., 1.], repeat=len(node.predecessors))))
        hidden = node.nn_module.net[0]
        # ReLU(1 - Hamming distance) selects one feature per binary context.
        # Only this feature map is fixed: output weights/biases remain random and
        # trainable. No generating probabilities are copied into the fitted model.
        with torch.no_grad():
            hidden.weight.copy_(2 * states - 1)
            hidden.bias.copy_(1 - states.sum(1))
        hidden.requires_grad_(False)
    return model


def test_sequential_g_formula_matches_population_interventions():
    # Independently enumerate the observational law, marginalizing latent U.
    joint = np.zeros((2, 2, 2, 2))
    for u, a1, l, a2, y in product([0, 1], repeat=5):
        p_l = 0.2 + 0.3*a1 + 0.3*u
        p_a2 = 0.25 + 0.5*l
        p_y = 0.1 + 0.1*a1 + 0.2*a2 + 0.2*l + 0.1*u + 0.1*a1*a2
        joint[a1, l, a2, y] += (0.25 * (p_l if l else 1-p_l)
                                * (p_a2 if a2 else 1-p_a2) * (p_y if y else 1-p_y))
    means = _g_formula(joint)
    np.testing.assert_allclose(means, TRUE_MEANS, atol=1e-12, rtol=0)
    assert means[1, 1] - means[0, 0] == pytest.approx(0.46, abs=1e-12)
    assert means[1, 1] - means[1, 0] == pytest.approx(0.30, abs=1e-12)
    # Conditioning on both treatment assignments is NOT joint intervention.
    naive = (joint[1, :, 1, 1].sum() / joint[1, :, 1, :].sum()
             - joint[0, :, 0, 1].sum() / joint[0, :, 0, :].sum())
    assert naive - 0.46 > 0.08


def test_fit_recovers_two_stage_means_and_contrasts():
    data = _simulate_two_stage(n=2048, seed=17)
    empirical_joint = np.zeros((2, 2, 2, 2))
    np.add.at(empirical_joint, tuple(data[OBSERVED].to_numpy().T), 1 / len(data))
    empirical_means = _g_formula(empirical_joint)

    set_seed(5, device='cpu')
    model = _build_model()
    variables = [VariableInfo(name, categorical=True, numeric_mapping={0: 0., 1: 1.})
                 for name in OBSERVED]
    config = TrainingConfig(device='cpu', optim='Adam', lr=0.1, epochs=100,
                            batch_size=len(data), sample_bs=1, energy_score_unbiased=False,
                            rng_seed=5)
    objective = EnergyLoss(config, model)
    batch, report = prepare_data(model, data, variables=variables)
    assert report['observed'] == tuple(OBSERVED)
    assert 'U' in report['unobserved']
    fit(model, data, objective, config, variables=variables)

    # Fit criterion: at most 2% total-variation discrepancy on observed cells,
    # with a small final gradient, evaluated AFTER the last optimizer update.
    final, diagnostics = objective.evaluate(batch)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    gradients = torch.autograd.grad(final.value, parameters)
    assert diagnostics['total_variation'].item() < 0.02
    assert max(g.abs().max().item() for g in gradients) < 0.001

    means = np.empty((2, 2))
    for a1, a2 in REGIMES:
        query = MeanPotentialOutcome(model, outcome='Y', intervention={'A1': a1, 'A2': a2},
                                     variable=variables[-1])
        means[a1, a2] = query.evaluate(sample_bs=1)[0].value.item()
    # Separate fitting/reference error from finite-sample deviation from truth.
    # These are absolute probability tolerances, not confidence intervals.
    np.testing.assert_allclose(means, empirical_means, atol=0.025, rtol=0)
    np.testing.assert_allclose(means, TRUE_MEANS, atol=0.06, rtol=0)

    for reference, truth in [((0, 0), 0.46), ((1, 0), 0.30)]:
        query = ATE(model, outcome='Y', alternative={'A1': 1, 'A2': 1},
                    reference=dict(zip(['A1', 'A2'], reference)), variable=variables[-1])
        effect = query.evaluate(sample_bs=1)[0].value.item()
        empirical_effect = empirical_means[1, 1] - empirical_means[reference]
        assert effect == pytest.approx(empirical_effect, abs=0.03)
        assert effect == pytest.approx(truth, abs=0.05)
