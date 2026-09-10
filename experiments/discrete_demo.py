import torch
import logging

from sem2.model.models import SEM2, SEM2Discrete
from sem2.default_configs import MLPConfig
from sem2.data import NamedBatch, VariableInfo

logging.basicConfig(level=logging.INFO)

# Base setup
formula1 = "U ~ Normal(0,1); eps_d ~ Uniform(0,1); eps_y ~ Uniform(0,1); Z ~ Categorical(probs=[1, 1, 1]);"
formula2 = "D <- d(Z, U, eps_d); Y <- d(D, U, eps_y)"
node_cfgs = {"U": 2, "eps_d": 2, "eps_y": 2,
            "D": MLPConfig(hidden_dims=[1024, 1024, 1024], output_dim=2),
            "Y": MLPConfig(hidden_dims=[1024, 1024, 1024], output_dim=3)}
# Continuous SEM
print("===== Test SEM2 Continuous =====")

model = SEM2(formula1 + formula2, node_cfgs=node_cfgs, device="cpu")
model.eval()

set_nodes = NamedBatch(
    torch.tensor([[1, 1], [0, 0], [2, 0]]),
    (VariableInfo("Z", True, True), VariableInfo("D", True, True)),
)
all_states, _ = model.sample(3, set_nodes=set_nodes)
print(f'{all_states = }')

# Discrete SEM
print("===== Test SEM2 Discrete =====")
# Base setup
formula1 = "U ~ Normal(0,1); eps_d ~ Uniform(0,1); eps_y ~ Uniform(0,1); Z ~ Categorical(probs=[1, 1, 1]);"
formula2 = "D <- d(Z, U, eps_d); Y <- d(D, U, eps_y)"
node_cfgs = {"U": 2, "eps_d": 2, "eps_y": 2,
            "D": MLPConfig(hidden_dims=[1024, 1024, 1024], output_dim=2),
            "Y": MLPConfig(hidden_dims=[1024, 1024, 1024], output_dim=3)}
print(node_cfgs)
model = SEM2Discrete(formula1 + formula2, node_cfgs=node_cfgs, device="cpu")
model.eval()

set_nodes = NamedBatch(
    torch.tensor([[1, 1], [0, 0], [2, 0]]),
    (VariableInfo("Z", True, True), VariableInfo("D", True, True)),
)
all_states, _ = model.sample(3, set_nodes=set_nodes)
print("===== Test SEM2 =====")
print(f'{all_states = }')
