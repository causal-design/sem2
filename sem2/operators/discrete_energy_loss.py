"""Energy calculations for named finite distributions."""
from copy import deepcopy
import torch
from ..data.records import DataBatch, ScalarResult
from ..model.distributions import DiscreteJointDistribution
from ..data.conversion import databatch_to_tensor
from ..default_configs import TrainingConfig


def _energy_values(batch: DataBatch) -> torch.Tensor:
    """Use quantitative values or explicit scores, never arbitrary category codes."""
    unscored = [v.name for v in batch.variables if v.categorical and v.numeric_mapping is None]
    if unscored:
        raise ValueError(f"Energy distances require a numeric_mapping for categories: {unscored}")
    return databatch_to_tensor(batch)


def _energy_distance(x: torch.Tensor, y: torch.Tensor, beta: float) -> torch.Tensor:
    """Powered Euclidean distances, assigning zero gradients at coincident points."""
    return _power_distances(torch.cdist(x, y), beta)


def _power_distances(distances: torch.Tensor, beta: float) -> torch.Tensor:
    """Apply the exponent without differentiating a fractional power at zero."""
    # Avoid differentiating zero**beta for fractional beta; masking afterwards
    # leaves an infinite derivative in the autograd graph.
    positive = distances > 0
    return distances.where(positive, 1).pow(beta).where(positive, 0)


def _check_beta(beta: float) -> None:
    """Validate a default at construction or an explicit exponent override."""
    if not 0 < beta < 2:
        raise ValueError("Energy beta must be strictly between zero and two.")


class DiscreteEnergyLoss:
    """Energy scores and half-energy distances for finite probability laws.

    observations.values has shape [N, D], prediction.states.values [S, D],
    and prediction.probabilities [M, S], where M counts exogenous draws.
    Callers supply valid state values and normalized probabilities; these are
    not rechecked here on each evaluation.

    Parameters
    ----------
    cfg : TrainingConfig
        Supplies the default exponent and pair-correction setting.
    states : DataBatch
        Fixed state support and numerical mappings. Distances are precomputed
        at construction; use a new loss for a different support or scoring.
    beta : float, optional
        Overrides ``cfg.energy_score_beta``.
    unbiased : bool, optional
        Overrides ``cfg.energy_score_unbiased``.
        Exclude same-row pairs from within-distribution distances. Corrected
        terms require at least two probability rows, and two observations when
        computing ``energy_dist``.
    """

    def __init__(self, cfg: TrainingConfig, states: DataBatch, *,
                 beta: float | None = None, unbiased: bool | None = None) -> None:
        """Precompute state distances in the same ordering used by marginalize()."""
        self.unbiased = cfg.energy_score_unbiased if unbiased is None else unbiased
        self.beta = cfg.energy_score_beta if beta is None else beta
        _check_beta(self.beta)
        self.states = DataBatch(torch.unique(states.values, dim=0), deepcopy(states.variables))
        numeric = _energy_values(self.states)
        # D[i,j] = ||state_i - state_j||_2**beta, using the chosen numeric mapping.
        self._distances = torch.cdist(numeric, numeric)
        self.D_mat = _power_distances(self._distances, self.beta)

    def _validate(self, observations: DataBatch, prediction: DiscreteJointDistribution, observed_pairs: bool = False) -> None:
        """Check sizes only, avoiding empty means and zero pair denominators."""
        n, m = len(observations), prediction.probabilities.shape[0]
        if not observations.variables or n < 1 or m < 1:
            raise ValueError("Energy calculations require named observations and model draws.")
        if self.unbiased and (m < 2 or (observed_pairs and n < 2)):
            raise ValueError("Unbiased pair distances require at least two rows.")

    def _pair_distance(self, probabilities: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
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

    def _energy_components(
        self, observations: DataBatch, prediction: DiscreteJointDistribution,
        *, beta: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cross/pair distances, empirical rows, state distances and cell errors."""
        # Sum over unobserved state coordinates and align the remainder by name.
        prediction = prediction.marginalize(self.states.names)
        probabilities = prediction.probabilities
        observations = observations.subset(self.states.names).to(probabilities.device)
        states = self.states.values.to(probabilities.device)
        # One indicator row per observation, in enumerated-state order [N, S].
        # Match stored identities directly, without coercing them to integer indices.
        empirical = (observations.values[:, None, :] == states[None, :, :]).all(-1).to(probabilities)
        # Reuse the precomputed matrix; overrides only change the power, not D_mat.
        if beta is not None and beta != self.beta:
            _check_beta(beta)
        distances = (self.D_mat if beta is None or beta == self.beta
                     else _power_distances(self._distances, beta)).to(probabilities)
        # Average the cross-distance over independent data and model draws.
        cross = empirical.mean(0) @ distances @ probabilities.mean(0)
        pair = self._pair_distance(probabilities, distances)
        error = (probabilities.mean(0) - empirical.mean(0)).abs()
        return cross, pair, empirical, distances, error

    def energy_score(self, observations: DataBatch, prediction: DiscreteJointDistribution,
                     *, beta: float | None = None) -> tuple[ScalarResult, dict]:
        """Energy score from observations [N, D] and predictions [M, S].

        Prediction states must match the fixed support; N and M may differ.
        Computes cross-distance minus half the model/model pair-distance.
        Returns a ScalarResult and scalar distance diagnostics.

        Parameters
        ----------
        observations : DataBatch
            Nonempty observed states; distance scoring is fixed at construction.
        prediction : DiscreteJointDistribution
            Normalized conditional probabilities and their named state table.
        beta : float, optional
            Temporary exponent override; leaves the precomputed default unchanged.

        Returns
        -------
        result : ScalarResult
            Cross-distance minus half the model-pair distance, with gradients.
        diagnostics : dict
            Cross and model-pair distances, ``total_variation`` and
            ``max_cell_error`` between the empirical and predicted margins.
        """
        self._validate(observations, prediction)
        cross, pair, _, _, error = self._energy_components(observations, prediction, beta=beta)
        result = ScalarResult(cross - pair / 2, "energy_score", observations.variables)
        return result, {"estimated-true distance": cross, "estimated-estimated distance": pair,
                        "total_variation": error.sum() / 2, "max_cell_error": error.max()}

    def energy_dist(self, observations: DataBatch, prediction: DiscreteJointDistribution,
                    *, beta: float | None = None) -> tuple[ScalarResult, dict]:
        """Half-energy distance from observations [N, D] and predictions [M, S].

        Prediction states must match the fixed support; each pair correction
        uses its own sample count. Returns a ScalarResult and scalar diagnostics.
        Subtracts half of each within-distribution distance from the cross term.
        The unbiased estimate can be negative at finite sample sizes.

        Parameters
        ----------
        observations : DataBatch
            Observed states; corrected data-pair distances require two rows.
        prediction : DiscreteJointDistribution
            Normalized probability rows and the aligned named state table.
        beta : float, optional
            Temporary exponent override; leaves the precomputed default unchanged.

        Returns
        -------
        result : ScalarResult
            Half-energy distance, labelled ``energy_distance``, with gradients.
        diagnostics : dict
            Cross, model-pair and data-pair terms under ``estimated-true distance``,
            ``estimated-estimated distance`` and ``true-true distance``.
            ``total_variation`` and ``max_cell_error`` compare the empirical and
            predicted probability margins, independently of numerical scores.
        """
        self._validate(observations, prediction, observed_pairs=True)
        cross, pair, empirical, distances, error = self._energy_components(observations, prediction, beta=beta)
        observed_pair = self._pair_distance(empirical, distances)
        result = ScalarResult(cross - pair / 2 - observed_pair / 2, "energy_distance", observations.variables)
        return result, {"estimated-true distance": cross, "estimated-estimated distance": pair,
                        "true-true distance": observed_pair,
                        "total_variation": error.sum() / 2, "max_cell_error": error.max()}
