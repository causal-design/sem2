"""Nonparametric identification from a single observed joint distribution."""
from __future__ import annotations

from collections.abc import Collection

from y0.algorithm.identify import identify_outcomes
from y0.dsl import Variable
from y0.graph import NxMixedGraph

from sem2.model.construction.admg import _ADMG
from sem2.model.construction.formula_parser import FormulaParser
from sem2.model.models.base import SEM2


def project_to_admg(model: SEM2 | _ADMG | str, *, observed: Collection[str]) -> NxMixedGraph:
    """Marginalize the formula's ADMG to the observed variables for y0.

    Parameters
    ----------
    model : SEM2, _ADMG or str
        Model, parsed graph, or formula. Only declared dependencies are used;
        coefficients, distributions and fitted parameters impose no restrictions.
    observed : collection of str
        User-declared variables in one observed joint distribution. All other
        scientific variables are treated as latent; helpers cannot be observed.

    Returns
    -------
    y0.graph.NxMixedGraph
        Observed ADMG, including isolated variables. Its ``undirected`` component
        stores bidirected confounding edges, following y0's convention.

    Notes
    -----
    Works directly on the named graph. Neither executable nodes nor clique
    sources are needed. If all vertices are observed, only the format changes.
    """
    if isinstance(model, str):
        _, graph = FormulaParser(model).parse()
    elif isinstance(model, SEM2):
        graph = model.admg
    elif isinstance(model, _ADMG):
        graph = model
    else:
        raise TypeError("model must be a SEM2 model, parsed ADMG, or formula string.")
    observed = _names(observed, "observed")
    margin = graph.marginalize(observed)
    return NxMixedGraph.from_str_edges(
        nodes=sorted(observed), directed=sorted(margin.directed), undirected=sorted(margin.bidirected),
    )


def is_identified(
    model: SEM2 | _ADMG | str,
    *,
    observed: Collection[str],
    treatments: str | Collection[str],
    outcomes: str | Collection[str],
) -> bool:
    """Check graphical identification of an intervention distribution using ID.

    Parameters
    ----------
    model : SEM2, _ADMG or str
        Structural model, parsed mixed graph, or formula; no fitting is required.
    observed : collection of str
        Variables available together in one observed joint distribution.
    treatments : str or collection of str
        Observed variables jointly intervened on. Empty means no intervention.
    outcomes : str or collection of str
        Nonempty set of observed outcome variables, disjoint from treatments.

    Returns
    -------
    bool
        Whether the full intervention distribution is identified in the
        nonparametric causal model represented by the projected graph.

    Raises
    ------
    ValueError
        For unknown/helper variables, unobserved targets, empty outcomes, or
        overlapping treatments and outcomes. Dependency errors propagate.

    Notes
    -----
    Uses Shpitser--Pearl ID through y0. Assumes the causal graph is correct and
    the usual support conditions for identification; it does not check data
    positivity. ``False`` does not rule out identification of a particular mean,
    or identification under additional parametric or deterministic restrictions.
    Conditional queries, data fusion and latent targets are not supported.
    """
    graph = project_to_admg(model, observed=observed)
    treatments = _names(treatments, "treatments")
    outcomes = _names(outcomes, "outcomes")
    if not outcomes:
        raise ValueError("outcomes must be nonempty.")
    if treatments & outcomes:
        raise ValueError("treatments and outcomes must be disjoint.")
    if missing := (treatments | outcomes) - {node.name for node in graph.nodes()}:
        raise ValueError(f"Treatments and outcomes must be observed variables: {sorted(missing)}")
    return identify_outcomes(
        graph,
        treatments={Variable(name) for name in treatments},
        outcomes={Variable(name) for name in outcomes},
    ) is not None


def _names(names: str | Collection[str], argument: str) -> set[str]:
    """Normalize node names without interpreting a string as individual letters."""
    if isinstance(names, str):
        return {names}
    if not isinstance(names, Collection) or any(not isinstance(name, str) for name in names):
        raise TypeError(f"{argument} must contain variable names as strings.")
    return set(names)
