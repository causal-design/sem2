"""One analytic regression for sweep signs, fresh fits and final diagnostics."""
import math

import pandas as pd
import pytest
import torch

from sem2.analysis.penalty_sweep import penalty_sweep
from sem2.default_configs import TrainingConfig
from sem2.model.models.linear import SEM2Linear
from sem2.operators.linear import LogLikelihood
from sem2.operators.potential_outcomes import MeanPotentialOutcome
from sem2.operators.sem2_loss import SEM2Loss


def test_sweep_matches_two_steps_on_a_penalized_gaussian_mean():
    def factory():
        model = SEM2Linear('u ~ Normal(0,1); y <- 1*u', device='cpu')
        with torch.no_grad():
            model.node_dict['y'].linear.bias.zero_()
        loss = SEM2Loss([(LogLikelihood(model), -1.)])
        # Fix zero-mean noise at zero so the query is the intercept exactly.
        target = MeanPotentialOutcome(model, outcome='y', intervention={'u': 0.}, sample_bs=1)
        return model, loss, target

    data = pd.DataFrame({'y': [-1., 0., 1.]})
    config = TrainingConfig(device='cpu', optim='SGD', momentum=0., lr=0.5,
                            epochs=2, batch_size=3)
    results = penalty_sweep(factory, data, [-0.5, 0.5], config)
    assert [result.weight for result in results] == [-0.5, 0.5]
    assert results[0].model is not results[1].model
    for result in results:
        # Loss = constant + mean^2/2 + weight*mean. Starting at zero,
        # two gradient steps of size 1/2 give mean = -3*weight/4.
        mean = -0.75 * result.weight
        fit_loss = 0.5 * math.log(2 * math.pi) + 1/3 + 0.5 * mean**2
        assert result.target == pytest.approx(mean, abs=1e-6)
        assert result.fit_loss == pytest.approx(fit_loss, abs=1e-6)
        assert result.objective == pytest.approx(fit_loss + result.weight * mean, abs=1e-6)
        assert result.max_abs_gradient == pytest.approx(abs(mean + result.weight), abs=1e-6)
        assert result.model.node_dict['y'].intercept().item() == pytest.approx(mean, abs=1e-6)
