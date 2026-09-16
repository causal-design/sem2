"""Finite-state energy objectives with named observations and predictions."""
import torch
import torch.nn.functional as F

from ..data import NamedBatch, FiniteDistribution


class DiscreteEnergyLoss:
    """Compare observations with probabilities [exogenous draws, states].

    The prediction owns its named state table; probabilities never travel as an
    unlabelled tensor argument. Distances use stored numerical values, not an
    inferred scoring of category IDs. Unbiased estimates can be negative.
    """

    def __init__(self, p: float, unbiased=True):
        self.p = p
        self.unbiased = unbiased

    def _pair_distance(self, probabilities, distances):
        n = probabilities.shape[0]
        if n == 0 or (self.unbiased and n < 2):
            raise ValueError("Unbiased pair distances need at least two samples in each population.")
        mean = probabilities.mean(0)
        pair = mean @ distances @ mean
        if self.unbiased:
            diagonal = torch.einsum("ni,ij,nj->", probabilities, distances, probabilities)
            pair = (pair * n * n - diagonal) / (n * (n - 1))
        return pair

    def _energy_components(self, x_true: NamedBatch, p_hat: FiniteDistribution):
        if not isinstance(x_true, NamedBatch):
            raise TypeError("Observations must be a NamedBatch.")
        if not isinstance(p_hat, FiniteDistribution):
            raise TypeError("Predictions must be a FiniteDistribution.")
        probabilities = p_hat.probabilities
        if probabilities.ndim != 2:
            raise ValueError("Predicted probabilities must have shape [exogenous draws, states].")
        states = p_hat.states
        if not len(x_true) or not len(states) or not states.names:
            raise ValueError("Energy fitting requires nonempty observations and named states.")
        observed = x_true.subset(*states.names)
        if tuple(v.width for v in observed.variables) != tuple(v.width for v in states.variables):
            raise ValueError("Observation variable widths must match the named states.")
        # Compare before coercion: an invalid fractional code must not become a
        # valid integer, and an unmatched observation must not become state zero.
        values = observed.values.to(states.values.device)
        matches = (values[:, None, :] == states.values[None, :, :]).all(-1)
        if not (matches.sum(-1) == 1).all():
            raise ValueError("Every observation must match exactly one declared state; invalid values are not coerced.")
        truth = F.one_hot(matches.long().argmax(-1), len(states)).to(probabilities)
        numeric_states = states.values.to(probabilities)
        distances = torch.cdist(
            numeric_states, numeric_states, compute_mode="donot_use_mm_for_euclid_dist",
        ) ** self.p
        pair_pred = self._pair_distance(probabilities, distances)
        cross = truth.mean(0) @ distances @ probabilities.mean(0)
        return cross, pair_pred, truth, distances

    def energy_score(self, x_true: NamedBatch, p_hat: FiniteDistribution):
        """Cross distance minus half the model-pair distance."""
        cross, pair_pred, _, _ = self._energy_components(x_true, p_hat)
        return cross - pair_pred / 2, {
            "estimated-true distance": cross,
            "estimated-estimated distance": pair_pred,
        }

    def energy_dist(self, x_true: NamedBatch, p_hat: FiniteDistribution):
        """Half the conventional energy distance, as in the fitting objective.

        Each pair correction uses its own population's sample count.
        """
        cross, pair_pred, truth, distances = self._energy_components(x_true, p_hat)
        pair_true = self._pair_distance(truth, distances)
        return cross - pair_pred / 2 - pair_true / 2, {
            "estimated-true distance": cross,
            "estimated-estimated distance": pair_pred,
            "true-true distance": pair_true,
        }
