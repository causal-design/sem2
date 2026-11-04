"""Population means and contrasts from a known model, without fitting."""
import torch

from sem2.model.models.linear import SEM2Linear
from sem2.operators.potential_outcomes import ATE, MeanPotentialOutcome


def main():
    model = SEM2Linear('noise ~ Normal(0,1); exposure <- 1*noise + 0; '
                       'mediator <- 2*exposure + 0; response <- 3*mediator + 0')
    mean = MeanPotentialOutcome(model, outcome='response', intervention={'exposure': 2})
    contrast = ATE(model, outcome='response',
                   alternative={'exposure': 2}, reference={'exposure': 0})
    joint = ATE(model, outcome='response',
                alternative={'exposure': 2, 'mediator': 1},
                reference={'exposure': 0, 'mediator': 0})
    for query, expected in [(mean, 12.), (contrast, 12.), (joint, 3.)]:
        result, details = query.evaluate(sample_bs=8)
        torch.testing.assert_close(result.value, torch.tensor(expected))
        print(result.name, result.value.detach().item(), details)


if __name__ == '__main__':
    main()
