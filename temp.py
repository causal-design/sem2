
import torch

probs = torch.tensor([1, 2 ,3 ,4 ,5])
x = torch.distributions.multinomial.Multinomial(100, torch.tensor([ 1., 1., 1., 1.]))

print(x.sample())

print(x.support.upper_bound)

