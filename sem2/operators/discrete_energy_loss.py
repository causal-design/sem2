"""Energy calculations for named finite distributions."""
import torch
import torch.nn.functional as F
from ..data import DataBatch, FiniteDistribution, ScalarResult
from .operators_base import _numeric_values


class DiscreteEnergyLoss:
    def __init__(self, p: float, unbiased=True):
        self.p = p
        self.unbiased = unbiased

    def _pair_distance(self, probabilities, distances):
        mean = probabilities.mean(0)
        pair = mean @ distances @ mean
        if self.unbiased:
            n = probabilities.shape[0]
            if n < 2:
                raise ValueError("Unbiased pair distances require at least two rows.")
            diagonal = torch.einsum("ni,ij,nj->n", probabilities, distances, probabilities)
            pair = (pair * n * n - diagonal.sum()) / (n * (n - 1))
        return pair

    def _energy_components(self, observations: DataBatch, prediction: FiniteDistribution):
        if not isinstance(observations, DataBatch) or not isinstance(prediction, FiniteDistribution):
            raise TypeError("Expected a DataBatch and a FiniteDistribution.")
        if not len(observations) or not observations.variables:
            raise ValueError("Energy calculations require observations and named variables.")
        prediction = prediction.marginalize(observations.names)
        probabilities = prediction.probabilities
        if not probabilities.shape[0] or not torch.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("Predicted probabilities must be finite, nonnegative and nonempty.")
        if not torch.allclose(probabilities.sum(-1), probabilities.new_ones(probabilities.shape[0]), atol=1e-5):
            raise ValueError("Predicted probabilities must sum to one per draw.")
        observations = observations.to(probabilities.device)
        states = prediction.states.values
        if observations.values.shape[1] != states.shape[1]:
            raise ValueError("Observation widths must match the state table.")
        # Match original values: never turn fractional or unknown states into codes.
        matches = (observations.values[:, None, :] == states[None, :, :]).all(-1)
        if not (matches.sum(-1) == 1).all():
            raise ValueError("Every observation must match exactly one declared state.")
        empirical = F.one_hot(matches.long().argmax(-1), len(states)).to(probabilities)
        numeric = _numeric_values(DataBatch(states, observations.variables)).to(probabilities)
        distances = torch.cdist(numeric, numeric) ** self.p
        cross = empirical.mean(0) @ distances @ probabilities.mean(0)
        pair = self._pair_distance(probabilities, distances)
        return cross, pair, empirical, distances

    def energy_score(self, observations: DataBatch, prediction: FiniteDistribution) -> tuple[ScalarResult, dict]:
        cross, pair, _, _ = self._energy_components(observations, prediction)
        result = ScalarResult(cross - pair / 2, "energy_score", observations.variables)
        return result, {"estimated-true distance": cross, "estimated-estimated distance": pair}

    def energy_dist(self, observations: DataBatch, prediction: FiniteDistribution) -> tuple[ScalarResult, dict]:
        """Retain the existing half-energy-distance normalization."""
        cross, pair, empirical, distances = self._energy_components(observations, prediction)
        observed_pair = self._pair_distance(empirical, distances)
        result = ScalarResult(cross - pair / 2 - observed_pair / 2, "energy_distance", observations.variables)
        return result, {"estimated-true distance": cross, "estimated-estimated distance": pair,
                        "true-true distance": observed_pair}
