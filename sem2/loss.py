from abc import ABC, abstractmethod
import torch

from .config import Config
from .model import SEM2

class SubLoss(ABC):
    """ Base class for loss functions.
        Loss functions take in batch (Xs, targets) and makes predictions. Then they compute the loss.
    """
    pass

    @abstractmethod
    def forward(self, Xs, targets, **kwargs) -> tuple[torch.Tensor, dict]:
        """ Compute the loss. Returns main loss tensor and additional metrics in dictionary. """
        pass


class EnergyScoreLoss(SubLoss):
    def __init__(self, cfg: Config, model: SEM2):
        """" Energy score loss function. Sample from our model and compute energy score between our sample and the true sample .
        """
        self.model = model
        self.bs = cfg.bs
        self.beta = cfg.energy_score_beta

    def forward(self, Xs, targets, **kwargs) -> tuple[torch.Tensor, dict]:
        # Draw sample from the model
        all_preds = self.model.forward(batch_size=self.bs, return_dict=True)
        # Filter out nodes for energy score
        preds = torch.cat([all_preds[node_name] for node_name in ["Z", "D", "Y"]], dim=-1)  # shape = [BS, n_nodes]

        # Compute energy score between predictions and sample.
        return self._energy_score(targets, preds, **kwargs)

    def _energy_score(self, targets, preds, **kwargs) -> tuple[torch.Tensor, dict]:
        """Loss function based on the energy score.

        Args:
            targets (torch.Tensor): iid samples from the true distribution of. Shape = [..., data_dim]
            preds (torch.Tensor): Predicted. shape = [..., data_dim]
        Returns:
            loss (torch.Tensor): energy loss.
            metrics (dict): dictionary of useful metrics.
        """
        m = preds.shape[1]
        s1 = torch.linalg.vector_norm(preds - targets, 2, dim=-1) ** self.beta
        s2 = (torch.cdist(preds, preds, 2) ** self.beta) * m / (m - 1)

        loss = (s1 - s2 / 2).mean()
        return loss, {"loss": loss, "estimated-true distance": s1.mean(), "estimated-estimated distance": s2.mean()}


class ATELoss(SubLoss):
    """ Compute ATE from the model. """
    def __init__(self, cfg: Config, model: SEM2):
        """
        Compute the average treatment effect from the model. Treatment and Target nodes are selected from the config.
            ATE = E[Y | T=0] - E[Y | T=1]
        Args:
            cfg: Config.
            model: The SEM2 model to evaluate ATE on. Must have methods to generate eps_y and eps_u, and a forward() method that can take set_nodes.
        """
        self.model = model
        self.device = cfg.device
        self.T_node = cfg.T_node
        self.Y_node = cfg.Y_node
        self.bs = cfg.bs

    def forward(self, Xs, targets, **kwargs) -> tuple[torch.Tensor, dict]:
        """ Evaluate ATE by intervening on D in the model.
            n_samples: Number of samples to evaluate ATE on.
        """

        t0 = torch.zeros([self.bs, 1], device=self.device, dtype=torch.long)
        t1 = torch.ones([self.bs, 1], device=self.device, dtype=torch.long)

        # Preds for Y | T=0
        set_nodes = {self.T_node: t0}
        preds_t0 = self.model.forward(batch_size=self.bs, set_nodes=set_nodes, return_dict=True)
        # Preds for Y | T=1
        set_nodes = {self.T_node: t1}
        prob_t1 = self.model.forward(batch_size=self.bs, set_nodes=set_nodes, return_dict=True)

        # ATE = E[Y | T=0] - E[Y | T=1]
        y_t0 = preds_t0[self.Y_node]
        y_t1 = prob_t1[self.Y_node]
        ATE = (y_t0 - y_t1).mean()
        return ATE, {}


class Loss:
    """ Computes loss from Xs and targets, with the ability to combine multiple losses together.
        The Overall loss is:
            L = sum_i(weight_i * sub_loss_i)
        Sub losses call the model to get predictions and compute the loss.
    """
    sub_losses: list[tuple[SubLoss, float]]

    def __init__(self, cfg: Config, model: SEM2):
        self.device = cfg.device
        sub_losses = []
        if cfg.use_energy_score:
            sub_losses.append((EnergyScoreLoss(cfg, model), cfg.ate_weight))
        if cfg.use_ate:
            sub_losses.append((ATELoss(cfg, model), cfg.ate_weight))
        if cfg.use_mmd:
            raise NotImplementedError("MMD loss not implemented yet.")

        self.sub_losses = sub_losses

    def forward(self, Xs, targets) -> tuple[torch.Tensor, dict]:
        """ Compute the combined loss. """
        total_loss = torch.tensor(0.0, device=self.device)
        metrics = {}
        # Weighted sum of sub losses
        for sub_loss, weight in self.sub_losses:
            sub_loss_value, sub_metrics = sub_loss.forward(Xs, targets)
            total_loss += weight * sub_loss_value
            metrics.update(sub_metrics)
        return total_loss, metrics

