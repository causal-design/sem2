import torch
import logging

from sem2.model.models.base import SEM2
from sem2.model.models.discrete import SEM2Discrete
from sem2.default_configs import MLPConfig
from sem2.dataloader.data import DataBatch

logging.basicConfig(level=logging.INFO)

# Base setup
formula1 = "U ~ Normal(0,1); eps_d ~ Uniform(0,1); eps_y ~ Uniform(0,1); Z ~ Categorical(probs=[1, 1, 1]);"
formula2 = "D <- d(Z, U, eps_d; levels=2); Y <- d(D, U, eps_y; levels=3)"
node_cfgs = {"U": 2, "eps_d": 2, "eps_y": 2,
            "D": MLPConfig(hidden_dims=[1024, 1024, 1024]),
            "Y": MLPConfig(hidden_dims=[1024, 1024, 1024])}
# Continuous SEM
print("===== Test SEM2 Continuous =====")

model = SEM2(formula1 + formula2, node_cfgs=node_cfgs, device="cpu")
model.eval()

set_nodes = DataBatch(torch.tensor([[1., 1.], [0., 0.], [2., 0.]]),
                      ["Z", "D"])
all_states, _ = model.sample(3, set_nodes=set_nodes)
print(f'{all_states = }')

# Discrete SEM
print("===== Test SEM2 Discrete =====")
# Base setup
formula1 = "U ~ Normal(0,1); eps_d ~ Uniform(0,1); eps_y ~ Uniform(0,1); Z ~ Categorical(probs=[1, 1, 1]);"
formula2 = "D <- d(Z, U, eps_d; levels=2); Y <- d(D, U, eps_y; levels=3)"
node_cfgs = {"U": 2, "eps_d": 2, "eps_y": 2,
            "D": MLPConfig(hidden_dims=[1024, 1024, 1024]),
            "Y": MLPConfig(hidden_dims=[1024, 1024, 1024])}
print(node_cfgs)
model = SEM2Discrete(formula1 + formula2, node_cfgs=node_cfgs, device="cpu")
model.eval()

set_nodes = DataBatch(torch.tensor([[1., 1.], [0., 0.], [2., 0.]]),
                      ["Z", "D"])
all_states, _ = model.sample(3, set_nodes=set_nodes)
print("===== Test SEM2 =====")
print(f'{all_states = }')
