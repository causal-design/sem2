from typing import TYPE_CHECKING
import torch
from .operators_base import OperatorInput, OperatorOutput, SEM2Operator
from .vector_operators import PotentialOutcome
from ..model.model import SEMType
from .discrete_energy_loss import DiscreteEnergyLoss

if TYPE_CHECKING:
    from ..model.model import SEM2, SEM2Discrete
    from ..default_configs import TrainingConfig


class EnergyLoss(SEM2Operator):
    """" Energy score loss function. Sample from our model and compute the energy score between our sample and the true sample.
    """

    input_type = OperatorInput.SELF
    output_type = OperatorOutput.SCALAR

    def __init__(self, cfg: TrainingConfig, model: SEM2):
        self.model = model
        self.device = cfg.device
        self.bs = cfg.sample_bs
        self.beta = cfg.energy_score_beta

        # Select the correct forward method based on the SEM type
        sem_type = self.model.sem_type
        if SEMType.is_continuous(sem_type):
            self.evaluate = self._evaluate_continuous
        elif SEMType.is_discrete(sem_type):
            self.evaluate = self._evaluate_discrete
            self.discrete_loss = DiscreteEnergyLoss(self.beta, model.all_states, self.device)
        else:
            raise NotImplementedError(f"Energy score is not implemented for this SEM type: {sem_type}")

    def _evaluate_discrete(self, observations, **kwargs) -> tuple[torch.Tensor, dict]:
        """ Discrete energy loss. Currently hard coded for the binary IV problem. """
        batch_size = observations.shape[0]

        # Split apart Z from X and Y
        z = observations[:, 0]
        xy = observations[:, 1:]
        # Observations
        set_nodes = {"Z": z.unsqueeze(-1)}
        p_all_hat, _ = self.model.sample(batch_size=batch_size, set_nodes=set_nodes)     # shape = [bs, n_states]

        loss_tot = torch.tensor(0.0, device=self.model.device)
        L_extra = []
        for i in range(2):
            z_mask = (z.squeeze(-1) == i)
            es, extra = self.discrete_loss.energy_dist(p_all_hat[z_mask].unsqueeze(0), x_true=xy[z_mask].unsqueeze(0))
            loss_tot += es
            L_extra.append(extra)

        # Average stats over all Z
        L_extra_mean = {}
        for extra in L_extra:
            for k, v in extra.items():
                if k not in L_extra_mean:
                    L_extra_mean[k] = []
                L_extra_mean[k].append(v)
        for k, v in L_extra_mean.items():
            L_extra_mean[k] = torch.stack(v).mean(dim=0)

        return loss_tot, L_extra_mean

    def _evaluate_continuous(self, observations, **kwargs) -> tuple[torch.Tensor, dict]:
        # Draw sample from the model
        all_preds, _ = self.model.sample(batch_size=self.bs)

        # Filter out nodes for energy score
        # TODO: once graph is specified, dynamically set observed nodes here.
        preds = torch.cat([all_preds[node_name] for node_name in ["Z", "X", "Y"]], dim=-1)  # shape = [BS, n_nodes]

        # Compute energy score between predictions and sample.
        return self._energy_score(observations, preds, **kwargs)

    def evaluate(self, observations, **kwargs) -> tuple[torch.Tensor, dict]:
        """ This is a placeholder that gets replaced by the appropriate method in __init__ based on the SEM type. """
        raise NotImplementedError("Energy score is not implemented for this SEM type.")

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


class MeanPotentialOutcome(SEM2Operator):
    """ Expected potential outcome:
            E[target_node | do(intervene_node = set_value)]
    """
    input_type = OperatorInput.DATA
    output_type = OperatorOutput.SCALAR

    def __init__(self, target_node: str, intervene_node: str, cfg: TrainingConfig, model: SEM2):
        self.device = cfg.device
        self.intervene_node = intervene_node
        self.model = model

        # Select the correct forward method based on the SEM type
        sem_type = self.model.sem_type
        if SEMType.is_continuous(sem_type):
            self.evaluate = self._evaluate_continuous
            self.sampler = PotentialOutcome(model, target_node)

        elif SEMType.is_discrete(sem_type):
            self.evaluate = self._evaluate_discrete
            # In the discrete case, mean marginal outcome is sum_a(p(y=a|T) * a)
            # Model outputs look like:
            #         [[0, 0], [0, 1], [1, 0], [1, 1]], shape = [n_conf, n_nodes]
            # Need to find the target value corresponding to each outcome.
            target_id = model.state_node_names.index(target_node)   # Position of target_node in the output index
            self.target_value = model.all_states[:, target_id]      # Value corresponding to target_node. Shape = [n_conf]
        else:
            raise NotImplementedError(f"Mean Potential Outcome is not implemented for this SEM type: {sem_type}")

    def _evaluate_discrete(self, set_value: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        """ Get the marginal probability of target_node. """
        # Type hint to specific SEM type, for syntax highlighting.
        model: SEM2Discrete = self.model
        bs = set_value.shape[0]

        set_nodes = {self.intervene_node: set_value}

        preds_all, _ = model.sample(batch_size=bs, set_nodes=set_nodes)  # shape = [BS, n_conf]
        preds_all = preds_all.mean(dim=0)  # shape = [n_conf]
        # Marginal outcome is sum_a(p(x=a) * a)
        mean_outcome = preds_all * self.target_value
        mean_outcome = mean_outcome.sum()
        return mean_outcome, {}


    def _evaluate_continuous(self, set_value: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        """ Get the potential outcome, setting 'set_node' equal to 'set_value'.
                E[target_node | do(intervene_node = set_value)]

            Args:
                set_value: the value to set the node to. Shape = [BS, 1]
        """
        bs = set_value.shape[0]
        set_nodes = {self.intervene_node: set_value}
        target_outcome, _ = self.sampler.evaluate(batch_size=bs, set_nodes=set_nodes)
        mean_target_outcome = target_outcome.mean()
        return mean_target_outcome, {}

    def evaluate(self, set_value: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        """ This is a placeholder that gets replaced by the appropriate method in __init__ based on the SEM type. """
        raise NotImplementedError("Mean Potential Outcome is not implemented for this SEM type.")


class ATE(SEM2Operator):
    """ Compute ATE from the model. """
    input_type = OperatorInput.SELF
    output_type = OperatorOutput.SCALAR

    def __init__(self, cfg: TrainingConfig, model: SEM2):
        """
        Compute the average treatment effect from the model. Treatment and Target nodes are selected from the config.
            ATE = E[Y | do(T=1)] - E[Y | do(T=0)]
        Args:
            cfg: TrainingConfig.
            model: The SEM2 model to evaluate ATE on. Must have methods to generate eps_y and eps_u, and a forward() method that can take set_nodes.
        """
        self.device = cfg.device
        self.bs = cfg.sample_bs
        self.model = model
        # Set up potential outcome
        T_node, Y_node = cfg.T_node, cfg.Y_node       # This might be changed when graph setup is finalized.
        self.treat_pot_out = MeanPotentialOutcome(Y_node, T_node, cfg, model)

    def evaluate(self, *args, **kwargs) -> tuple[torch.Tensor, dict]:
        """ Evaluate ATE by intervening on the treatment node in the model.
            Works for both continuous and discrete SEM2 models.
        """

        t0 = torch.zeros([self.bs, 1], device=self.device, dtype=torch.long)
        t1 = torch.ones([self.bs, 1], device=self.device, dtype=torch.long)

        # Preds for Y | T=0
        pot_out_t0, _ = self.treat_pot_out.evaluate(t0)
        # Preds for Y | T=1
        pot_out_t1, _ = self.treat_pot_out.evaluate(t1)

        # ATE = E[Y | T=1] - E[Y | T=0]
        ATE = pot_out_t1 - pot_out_t0
        return ATE, {}


class SEM2Loss(SEM2Operator):
    """ Computes the loss from overall loss from observations, with the ability to combine multiple losses together.
        The Overall loss is:
            L = sum_i(weight_i * loss_componenet_i)
        Loss components call the model to get predictions and compute the loss.
    """
    input_type = OperatorInput.DATA
    output_type = OperatorOutput.SCALAR

    loss_components: list[tuple[SEM2Operator, float]]

    def __init__(self, cfg: TrainingConfig, model: SEM2):
        self.device = cfg.device
        loss_components = []
        if cfg.use_energy_score:
            loss_components.append((EnergyLoss(cfg, model), cfg.energy_score_weight))
        if cfg.use_ate:
            loss_components.append((ATE(cfg, model), cfg.ate_weight))
        if cfg.use_mmd:
            raise NotImplementedError("MMD loss not implemented yet.")

        # Check components are all scalar
        for loss_comp, weight in loss_components:
            assert loss_comp.output_type == OperatorOutput.SCALAR, "Loss components must be scalar."
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
