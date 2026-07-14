import torch
import logging

from sem2.sem2trainer import TrainingConfig, SEM2Trainer
from sem2.model.models import SEM2


def main():
    torch.manual_seed(0)
    logging.basicConfig(level=logging.INFO)
    cfg = TrainingConfig()

    formula1 = "u ~ Uniform(0,1); eps_d ~ Uniform(0,1); eps_y ~ Uniform(0,1); eps_z ~ Uniform(0,1);"
    formula2 = "Z <- n(eps_z); D <- n(Z, u, eps_d); Y <- n(D, u, eps_y)"
    model_formula = formula1 + formula2
    sem_model = SEM2(model_formula, device=cfg.device)


    trainer = SEM2Trainer(cfg, sem_model, [[0, 0]])
    trainer.train()


if __name__ == "__main__":
    main()
