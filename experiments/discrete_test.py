from model.model import SEM2
from sem2.default_configs import MLPConfig

formula1 = "U ~ Normal(0,1); eps_d ~ Uniform(0,1); eps_y ~ Uniform(0,1); Z ~ Categorical(probs=[1, 1, 1]);"
formula2 = "D <- d(Z, U, eps_d); Y <- d(D, U, eps_y)"
mlp_cfgs = {"U": 2, "eps_d": 2, "eps_y": 2,
            "D": MLPConfig(input_dim=1 + 2 + 2, hidden_dims=[1024, 1024, 1024], output_dim=2),
            "Y": MLPConfig(input_dim=1 + 2 + 2, hidden_dims=[1024, 1024, 1024], output_dim=3)}

model = SEM2(formula1 + formula2, node_cfgs=mlp_cfgs, device="cpu")
model.eval()
model.formula_list

# [model._input_node2index]
# set_nodes = {"Z": torch.tensor([[1.], [0.], [100.]], device="cpu"),
# "D": torch.tensor([[1], [0], [0]], device="cpu"),
# }
# all_states, probs, values = model(3, set_nodes=set_nodes)
all_states, _ = model.sample(batch_size=2)

print("===== Test SEM2 =====")
print(f'{all_states = }')

