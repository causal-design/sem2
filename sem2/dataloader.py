import torch


class DummyDiscreteDataloader:
    """ A test dataloader for discrete SEMs.
        This is the IV binary problem. The exact solution is known.
    """
    outcome_probs = [0.20831, 0.12493, 0.14293, 0.023828, 0.11672, 0.049984, 0.24999, 0.083305]
    outcome_vals = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    def __init__(self, n_steps: int, batch_size: int = 128):
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.outcome_probs = torch.tensor(self.outcome_probs)
        self.outcome_vals = torch.tensor(self.outcome_vals)

    def __iter__(self):
        for _ in range(self.n_steps):
            batch = torch.multinomial(self.outcome_probs, self.batch_size, replacement=True)
            batch_vals = [self.outcome_vals[i] for i in batch]
            yield torch.stack(batch_vals, dim=0)


if __name__ == "__main__":
    dl = DummyDiscreteDataloader(n_steps=32)
    dl = iter(dl)
    for _ in range(10):
        print(next(dl))


