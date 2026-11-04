"""Runnable demos for the SEM2 model classes.

Run with:  python experiments/integration_test_demos.py
(Slated to be folded into integration tests.)
"""
from sem2.model.models.base import SEM2
from sem2.default_configs import MLPConfig, LinearNodeConfig


def main():

    # ── Example 1: Linear SEM ─────────────────────────────────────────────────

    # Fixed coefficients on x1 and x2, learned weight on noise.
    # use_intercept=False omits a learned intercept when the formula has no constant.
    # Otherwise usually dont need touch LinearNodeConfig at all, dont need to specify and just use defaults it should work ok.
    print("=== Linear SEM ===")
    linear_model = SEM2(
        "x1, x2, noise ~ Normal(0,1); y <- 0.5*x1 + -1*x2 + noise",
        node_cfgs={"y": LinearNodeConfig(use_intercept=False)},
    )
    print(linear_model.sample(batch_size=3))

    # ── Example 2: Pure MLP SEM ───────────────────────────────────────────────
    
    # All inputs enter a nonlinear MLP, output is y.
    print("\n=== MLP SEM ===")
    mlp_model = SEM2(
        "x1, x2, noise ~ Normal(0,1); y <- n(x1, x2, noise)",
        node_cfgs={"y": MLPConfig(hidden_dims=[16, 16], output_dim=1)},
    )
    print(mlp_model.sample(batch_size=3))

    # ── Example 3: Mixed linear + MLP SEM (implicit intermediate node) ────────

    # When a formula mixes a nonlinear term and a linear term with '+', _expand_rhs automatically lifts the n(...) part into a separate 
    # intermediate node named "y:0" (MLP) and rewrites y as a linear sum of y:0 and noise. This intermediate node is an internal detail the user never 
    # names or configures it, it uses default MLPConfig.
    #
    # This mixed syntax is supported for convenience but makes the graph harder to inspect. 
    # Honestly, prefer Example 2's way of EXPLICIT style formula typing and that recommended for as best practice:
    #        where routing all inputs into one named MLP node because is clearer and easier to configure. 
    # Have implemented this to just have the option where needed if someone wants the convinence. 
    # (Potentially useful for both power users and less technical users)
    print("\n=== Mixed Linear + MLP SEM (implicit intermediate node) ===")
    mixed_model = SEM2(
        "x1, x2, noise ~ Normal(0,1); y <- n(x1, x2) + 1*noise",
    )
    print(mixed_model.sample(batch_size=3))

    # Example 4: Mixed discrete SEM, with sampling over discrete nodes
    formula1 = "U ~ Normal(0,1); eps_d ~ Uniform(0,1); eps_y ~ Uniform(0,1); Z ~ Categorical(probs=[1, 1, 1]);"
    formula2 = "X <- d(Z, U, eps_d; levels=2); Y <- d(X, U, eps_y; levels=3)"
    mlp_cfgs = {"U": 2, "eps_d": 2, "eps_y": 2,
                "X": MLPConfig(hidden_dims=[1024, 1024, 1024]),
                "Y": MLPConfig(hidden_dims=[1024, 1024, 1024])}

    model = SEM2(formula1 + formula2, node_cfgs=mlp_cfgs, device="cpu")
    model.eval()
    all_states, _ = model.sample(batch_size=2)
    print("\n=== Discrete SEM, with sampling over discrete nodes ===")

    print(f'{all_states = }')


if __name__ == "__main__":
    main()
