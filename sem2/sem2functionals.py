from typing import TYPE_CHECKING
import torch
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .model import SEM2
    from .default_configs import TrainingConfig


class SEM2EvaluationFunctional(ABC):
    """ An evaluation functional takes in the SEM2 model, data and returns a scalar metric.
            phi : F X V -> R
            for SEM2 function F, vector (or similar object) V and R output scalar.
        The input vector may not necessarily be used. An additional dictionary is returned that contains useful debug / tracking information.
        Must always return a scalar value.
    """
    model: SEM2

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> tuple[torch.Tensor, dict]:
        pass


class EnergyScore(SEM2EvaluationFunctional):
    def __init__(self, cfg: TrainingConfig, model: SEM2):
        """" Energy score loss function. Sample from our model and compute energy score between our sample and the true sample.
        """
        self.model = model
        self.bs = cfg.bs
        self.beta = cfg.energy_score_beta

    def evaluate(self, observations, **kwargs) -> tuple[torch.Tensor, dict]:
        # Draw sample from the model
        all_preds = self.model.forward(batch_size=self.bs, return_dict=True)
        # Filter out nodes for energy score
        # TODO: once graph is specified, dynamically set observed nodes here.
        preds = torch.cat([all_preds[node_name] for node_name in ["Z", "D", "Y"]], dim=-1)  # shape = [BS, n_nodes]

        # Compute energy score between predictions and sample.
        return self._energy_score(observations, preds, **kwargs)

    def _energy_score(self, targets, preds, **kwargs) -> tuple[torch.Tensor, dict]:
        """Loss function based on the energy score.

        Args:
            targets (torch.Tensor): iid samples from the true distribution of. Shape = [bs, ...]
            preds (torch.Tensor): Predicted. shape = [bs, ...]
        Returns:
            loss (torch.Tensor): energy loss.
            metrics (dict): dictionary of useful metrics.
        """
        m = preds.shape[1]
        s1 = torch.linalg.vector_norm(preds - targets, 2, dim=-1) ** self.beta
        s2 = (torch.cdist(preds, preds, 2) ** self.beta) * m / (m - 1)

        energy_score = (s1 - s2 / 2).mean()
        return energy_score, {"energy_score": energy_score, "estimated-true distance": s1.mean(), "estimated-estimated distance": s2.mean()}


class MeanPotentialOutcome(SEM2EvaluationFunctional):
    """ Expected potential outcome:
            E[target_node | do(intervene_node = set_value)]
    """

    def __init__(self, target_node: str, intervene_node: str, cfg: TrainingConfig, model: SEM2):
        self.model = model
        self.device = cfg.device
        self.intervene_node = intervene_node
        self.target_node = target_node

    def evaluate(self, set_value: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        """ Get potential outcome, setting 'set_node' equal to 'set_value'.
                E[target_node | do(intervene_node = set_value)]

            Args:
                set_value: the value to set the node to. Shape = [BS, 1]
        """
        bs = set_value.shape[0]
        # I think we will eventually modify this class to use a full 'PotentialOutcome'.
        set_nodes = {self.intervene_node: set_value}
        potential_outcome = self.model.forward(batch_size=bs, set_nodes=set_nodes, return_dict=True)   # shape = [BS, 1]
        target_outcome = potential_outcome[self.target_node]
        mean_target_outcome = target_outcome.mean()
        return mean_target_outcome, {}


class ATE(SEM2EvaluationFunctional):
    """ Compute ATE from the model. """
    def __init__(self, cfg: TrainingConfig, model: SEM2):
        """
        Compute the average treatment effect from the model. Treatment and Target nodes are selected from the config.
            ATE = E[Y | do(T=0)] - E[Y | do(T=1)]
        Args:
            cfg: TrainingConfig.
            model: The SEM2 model to evaluate ATE on. Must have methods to generate eps_y and eps_u, and a forward() method that can take set_nodes.
        """
        self.model = model
        self.device = cfg.device
        self.bs = cfg.bs
        # Set up potential outcome
        T_node, Y_node = cfg.T_node, cfg.Y_node       # This might be changed when graph setup is finalized.
        self.treat_pot_out = MeanPotentialOutcome(Y_node, T_node, cfg, model)

    def evaluate(self, *args, **kwargs) -> tuple[torch.Tensor, dict]:
        """ Evaluate ATE by intervening on treatment node in the model."""

        t0 = torch.zeros([self.bs, 1], device=self.device, dtype=torch.long)
        t1 = torch.ones([self.bs, 1], device=self.device, dtype=torch.long)

        # Preds for Y | T=0
        pot_out_t0, _ = self.treat_pot_out.evaluate(t0)
        # Preds for Y | T=1
        pot_out_t1, _ = self.treat_pot_out.evaluate(t1)

        # ATE = E[Y | T=1] - E[Y | T=0]
        ATE = pot_out_t1 - pot_out_t0
        return ATE, {}


class SEM2Loss(SEM2EvaluationFunctional):
    """ Computes the loss from overall loss from observations, with the ability to combine multiple losses together.
        The Overall loss is:
            L = sum_i(weight_i * loss_componenet_i)
        Loss components call the model to get predictions and compute the loss.
    """
    loss_components: list[tuple[SEM2EvaluationFunctional, float]]

    def __init__(self, cfg: TrainingConfig, model: SEM2):
        self.device = cfg.device
        loss_components = []
        if cfg.use_energy_score:
            loss_components.append((EnergyScore(cfg, model), cfg.energy_score_weight))
        if cfg.use_ate:
            loss_components.append((ATE(cfg, model), cfg.ate_weight))
        if cfg.use_mmd:
            raise NotImplementedError("MMD loss not implemented yet.")

        self.loss_components = loss_components

    def evaluate(self, observations, **kwargs) -> tuple[torch.Tensor, dict]:
        """ Compute the combined loss. """
        total_loss = torch.tensor(0.0, device=self.device)
        metrics = {}
        # Weighted sum of sub losses
        for loss_comp, weight in self.loss_components:
            loss_comp_value, sub_metrics = loss_comp.evaluate(observations, **kwargs)
            total_loss += weight * loss_comp_value
            metrics.update(sub_metrics)         # Potential collisions here.
        return total_loss, metrics

