import torch
import torch.nn.functional as F

class DiscreteEnergyLoss:
    def __init__(self, p: float, states: torch.Tensor, device, unbiased=True):
        """
        p: norm power
        states: tensor of shape [num_states, state_dim] containing all possible states
                Example binary state: tensor([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])
        unbiased: whether to use unbiased estimator for finite sample size correction. Unbiased can be negative.
        """
        self.unbiased = unbiased
        self.states = states.to(device)
        self.num_states = states.shape[0]

        # Dynamically generate distance matrix D_mat
        # D_mat[i,j] = ||states[i] - states[j]||^p
        self.D_mat = self._compute_distance_matrix(states, p, device)

    def _compute_distance_matrix(self, states, p, device):
        """Compute pairwise distance matrix between states."""
        num_states = states.shape[0]
        D_mat = torch.zeros((num_states, num_states), dtype=torch.float32, device=device)

        for i in range(num_states):
            for j in range(num_states):
                # Euclidean distance raised to power p
                dist = torch.norm(states[i].float() - states[j].float(), p=2) ** p
                D_mat[i, j] = dist

        return D_mat

    def idx_to_onehot(self, input_tensor):
        """Convert input tensor to one-hot encoding based on self.states.
            input_tensor: shape = [bs, n_samples, n_nodes]
                         where each element represents a state configuration
        Returns:
            onehot: shape = [bs, n_samples, num_states]
                   where num_states = self.states.shape[0]
        """
        # Get dimensions
        bs, n_samples, n_nodes = input_tensor.shape
        num_states = self.states.shape[0]

        # Flatten input for easier comparison
        input_flat = input_tensor.view(-1, n_nodes)  # [bs*n_samples, n_nodes]

        # Vectorized comparison: find matching indices in self.states
        # Expand dimensions for broadcasting: input_flat [bs*n_samples, 1, n_nodes], states [1, num_states, n_nodes]
        input_expanded = input_flat.unsqueeze(1)  # [bs*n_samples, 1, n_nodes]
        states_expanded = self.states.unsqueeze(0)  # [1, num_states, n_nodes]

        # Find exact matches for each configuration
        matches = (input_expanded == states_expanded).all(dim=2)  # [bs*n_samples, num_states]

        # Get the index of the matching state for each input configuration
        indices = torch.argmax(matches.float(), dim=1)  # [bs*n_samples]

        # Convert to one-hot encoding
        onehot = F.one_hot(indices, num_classes=num_states).float()

        # Reshape back to original batch structure
        onehot = onehot.view(bs, n_samples, num_states)

        return onehot

    def _energy_components(self, p_hat, x_true=None, p_true_mean=None):
        """Compute shared components: pred_err and unbiased pair_pred_dist.

        Args:
            p_hat: [bs, n_samples, n_states]
            x_true: [bs, n_samples, n_nodes] (optional, used if p_true_mean not given)
            p_true_mean: [bs, n_states] (optional, precomputed)
        Returns:
            (pred_err, pair_pred_dist, p_true_mean, p_true)
                p_true is None if p_true_mean was precomputed.
        """
        bs, n_samples, dim = p_hat.shape
        precomputed = p_true_mean is not None
        p_true = None

        if not precomputed:
            p_true = self.idx_to_onehot(x_true.long())
            p_true_mean = p_true.mean(dim=1)

        p_hat_mean = p_hat.mean(dim=1)

        # Cross distance: true vs pred
        prob_mat = p_true_mean.unsqueeze(-1) @ p_hat_mean.unsqueeze(1)
        pred_err = (prob_mat * self.D_mat.unsqueeze(0)).mean(dim=0).sum()

        # Pairwise pred distance
        prob_mat = p_hat_mean.unsqueeze(-1) @ p_hat_mean.unsqueeze(1)
        pair_pred_dist = (prob_mat * self.D_mat.unsqueeze(0)).mean(dim=0).sum()

        # Bias correction
        if self.unbiased:
            n_tot, n_corr = n_samples / (n_samples - 1), 1 / (n_samples * (n_samples - 1))
            correct_hat = torch.einsum("bni,ij,bnj->bn", p_hat, self.D_mat, p_hat)
            correct_hat = correct_hat.mean(dim=0).sum()
            pair_pred_dist = pair_pred_dist * n_tot - correct_hat * n_corr

        return pred_err, pair_pred_dist, p_true_mean, p_true

    def energy_score(self, x_true, p_hat, p_true_mean=None):
        """Energy score: E(|x_h - x_true|) - 1/2 E(|x_hi - x_hj|)

        Args:
            x_true: [bs, n_samples, n_nodes] iid samples from true distribution.
            p_hat:  [bs, n_samples, n_states] predicted probabilities.
            p_true_mean: [bs, n_states] optional precomputed mean.
        Returns:
            (loss, log_dict)
        """
        pred_err, pair_pred_dist, _, _ = self._energy_components(
            p_hat, x_true=x_true, p_true_mean=p_true_mean
        )
        loss = pred_err - pair_pred_dist / 2
        return loss, {"estimated-true distance": pred_err,
                      "estimated-estimated distance": pair_pred_dist}

    def energy_dist(self, p_hat, x_true=None, p_true_mean=None):
        """Energy distance: E|x_h - x_true| - 1/2 E|x_hi - x_hj| - 1/2 E|x_truei - x_truej|

        Args:
            p_hat:  [bs, n_samples, n_states] predicted probabilities.
            x_true: [bs, n_samples, n_nodes] samples (optional if p_true_mean given).
            p_true_mean: [bs, n_states] optional precomputed mean.
        Returns:
            (loss, log_dict)
        """
        pred_err, pair_pred_dist, p_true_mean, p_true = self._energy_components(
            p_hat, x_true=x_true, p_true_mean=p_true_mean
        )

        # Pairwise true distance
        prob_mat = p_true_mean.unsqueeze(-1) @ p_true_mean.unsqueeze(1)
        pair_true_dist = (prob_mat * self.D_mat.unsqueeze(0)).mean(dim=0).sum()

        if self.unbiased and p_true is not None:
            n_samples = p_hat.shape[1]
            n_tot, n_corr = n_samples / (n_samples - 1), 1 / (n_samples * (n_samples - 1))
            correct_pred = torch.einsum("bni,ij,bnj->bn", p_true, self.D_mat, p_true)
            correct_pred = correct_pred.mean(dim=0).sum()
            pair_true_dist = pair_true_dist * n_tot - correct_pred * n_corr

        loss = pred_err - pair_pred_dist / 2 - pair_true_dist / 2
        return loss, {"estimated-true distance": pred_err,
                      "estimated-estimated distance": pair_pred_dist,
                      "true-true distance": pair_true_dist}
