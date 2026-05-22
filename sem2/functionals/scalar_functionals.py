from typing import TYPE_CHECKING
import torch
from sem2.functionals.functional_base import SEM2ScalarFunctional, SEM2SelfFunctional, SEM2DataFunctional
from sem2.functionals.vector_functionals import SampleContinuous, PotentialOutcome

if TYPE_CHECKING:
    from model import SEM2
    from default_configs import TrainingConfig


class EnergyScore(SEM2SelfFunctional, SEM2ScalarFunctional):
    def __init__(self, cfg: TrainingConfig, model: SEM2):
        """" Energy score loss function. Sample from our model and compute energy score between our sample and the true sample.
        """
        self.model = model
        self.bs = cfg.bs
        self.beta = cfg.energy_score_beta

        self.continuous_sampler = SampleContinuous(model)

    def evaluate(self, observations, **kwargs) -> tuple[torch.Tensor, dict]:
        # Draw sample from the model
        all_preds, _ = self.continuous_sampler.evaluate(batch_size=self.bs)

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


class MeanPotentialOutcome(SEM2DataFunctional, SEM2ScalarFunctional):
    """ Expected potential outcome:
            E[target_node | do(intervene_node = set_value)]
    """

    def __init__(self, target_node: str, intervene_node: str, cfg: TrainingConfig, model: SEM2):
        self.device = cfg.device
        self.intervene_node = intervene_node

        self.continuous_sampler = PotentialOutcome(model, target_node)

    def evaluate(self, set_value: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        """ Get potential outcome, setting 'set_node' equal to 'set_value'.
                E[target_node | do(intervene_node = set_value)]

            Args:
                set_value: the value to set the node to. Shape = [BS, 1]
        """
        bs = set_value.shape[0]
        set_nodes = {self.intervene_node: set_value}
        target_outcome, _ = self.continuous_sampler.evaluate(batch_size=bs, set_nodes=set_nodes)
        mean_target_outcome = target_outcome.mean()
        return mean_target_outcome, {}


class ATE(SEM2SelfFunctional, SEM2ScalarFunctional):
    """ Compute ATE from the model. """
    def __init__(self, cfg: TrainingConfig, model: SEM2):
        """
        Compute the average treatment effect from the model. Treatment and Target nodes are selected from the config.
            ATE = E[Y | do(T=1)] - E[Y | do(T=0)]
        Args:
            cfg: TrainingConfig.
            model: The SEM2 model to evaluate ATE on. Must have methods to generate eps_y and eps_u, and a forward() method that can take set_nodes.
        """
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


class SEM2Loss(SEM2ScalarFunctional):
    """ Computes the loss from overall loss from observations, with the ability to combine multiple losses together.
        The Overall loss is:
            L = sum_i(weight_i * loss_componenet_i)
        Loss components call the model to get predictions and compute the loss.
    """
    loss_components: list[tuple[SEM2ScalarFunctional, float]]

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

