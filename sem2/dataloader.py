import torch

from .data import NamedBatch, VariableInfo


class DummyDiscreteDataloader:
    """ A test dataloader for discrete SEMs.
        This is the IV binary problem. The exact solution is known.
        Yields NamedBatch with shared metadata for Z (instrument), X (treatment),
        and Y (outcome). This synthetic fixture does not implement preprocessing.
    """
    outcome_probs = [0.20831, 0.12493, 0.14293, 0.023828, 0.11672, 0.049984, 0.24999, 0.083305]
    outcome_vals = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    def __init__(self, n_steps: int, batch_size: int = 128):
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.outcome_probs = torch.tensor(self.outcome_probs)
        self.outcome_vals = torch.tensor(self.outcome_vals)
        self.variables = tuple(
            VariableInfo(name, categorical=True, discrete=True, encoding={0: 0, 1: 1})
            for name in ("Z", "X", "Y")
        )

    def __iter__(self):
        for _ in range(self.n_steps):
            batch = torch.multinomial(self.outcome_probs, self.batch_size, replacement=True)
            yield NamedBatch(self.outcome_vals[batch], self.variables)

    def __len__(self):
        return self.n_steps


if __name__ == "__main__":
    dl = DummyDiscreteDataloader(n_steps=32)
    dl = iter(dl)
    for _ in range(10):
        print(next(dl))


