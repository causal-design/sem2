"""Energy calculations for named finite distributions."""
import torch
from ..data import DataBatch, DiscreteDistribution, ScalarResult
from .operators_base import _apply_numeric_mapping


class DiscreteEnergyLoss:
    """Mathematical implementation used by EnergyLoss.

    observations.values has shape [N, D], prediction.states.values [S, D],
    and prediction.probabilities [M, S], where M counts exogenous draws.
    Callers supply valid state values and normalized probabilities; these are
    not rechecked here on each evaluation.
    """

    def __init__(self, p: float, unbiased=True):
        """Set the distance power and whether to exclude same-draw pairs."""
        self.p = p
        self.unbiased = unbiased

    def _validate(self, observations, prediction, observed_pairs=False):
        """Check sizes only, avoiding empty means and zero pair denominators."""
        n, m = len(observations), prediction.probabilities.shape[0]
        if not observations.variables or n < 1 or m < 1:
            raise ValueError("Energy calculations require named observations and model draws.")
        if self.unbiased and (m < 2 or (observed_pairs and n < 2)):
            raise ValueError("Unbiased pair distances require at least two rows.")

    def _pair_distance(self, probabilities, distances):
        """Average pair distance for probabilities [M, S] and distances [S, S].

        The uncorrected term is mean(p).T @ D @ mean(p). Multiply by M*M
        to recover the sum over all row pairs, subtract the same-row terms
        p_i.T @ D @ p_i, then divide by M*(M-1). These terms can be nonzero
        even when diag(D)=0:
        a probability row is a distribution, not a realized state.
        """
        mean = probabilities.mean(0)
        pair = mean @ distances @ mean
        if self.unbiased:
            n = probabilities.shape[0]
            diagonal = torch.einsum("ni,ij,nj->n", probabilities, distances, probabilities)
            pair = (pair * n * n - diagonal.sum()) / (n * (n - 1))
        return pair

    def _energy_components(self, observations: DataBatch, prediction: DiscreteDistribution):
        """Return scalar cross/pair distances, empirical rows [N, S] and distances [S, S]."""
        # Sum over unobserved state coordinates and align the remainder by name.
        prediction = prediction.marginalize(observations.names)
        probabilities = prediction.probabilities
        observations = observations.to(probabilities.device)
        states = prediction.states.values
        # One indicator row per observation, in enumerated-state order [N, S].
        # Match stored identities directly, without coercing them to integer indices.
        empirical = (observations.values[:, None, :] == states[None, :, :]).all(-1).to(probabilities)
        numeric = _apply_numeric_mapping(DataBatch(states, observations.variables)).to(probabilities)
        # D[i,j] = ||state_i - state_j||_2**p, using the chosen numeric mapping.
        distances = torch.cdist(numeric, numeric) ** self.p
        # Average the cross-distance over independent data and model draws.
        cross = empirical.mean(0) @ distances @ probabilities.mean(0)
        pair = self._pair_distance(probabilities, distances)
        return cross, pair, empirical, distances

    def energy_score(self, observations: DataBatch, prediction: DiscreteDistribution) -> tuple[ScalarResult, dict]:
        """Energy score from observations [N, D] and predictions [M, S].

        The prediction owns its named state table [S, D]; N and M may differ.
        Computes cross-distance minus half the model/model pair-distance.
        Returns a ScalarResult and scalar distance diagnostics.
        """
        self._validate(observations, prediction)
        cross, pair, _, _ = self._energy_components(observations, prediction)
        result = ScalarResult(cross - pair / 2, "energy_score", observations.variables)
        return result, {"estimated-true distance": cross, "estimated-estimated distance": pair}

    def energy_dist(self, observations: DataBatch, prediction: DiscreteDistribution) -> tuple[ScalarResult, dict]:
        """Half-energy distance from observations [N, D] and predictions [M, S].

        The prediction owns its named state table [S, D]; each pair correction
        uses its own sample count. Returns a ScalarResult and scalar diagnostics.
        Subtracts half of each within-distribution distance from the cross term.
        The unbiased estimate can be negative at finite sample sizes.
        """
        self._validate(observations, prediction, observed_pairs=True)
        cross, pair, empirical, distances = self._energy_components(observations, prediction)
        observed_pair = self._pair_distance(empirical, distances)
        result = ScalarResult(cross - pair / 2 - observed_pair / 2, "energy_distance", observations.variables)
        return result, {"estimated-true distance": cross, "estimated-estimated distance": pair,
                        "true-true distance": observed_pair}
