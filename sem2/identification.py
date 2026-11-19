"""Nonparametric identification from a single observed joint distribution."""
from __future__ import annotations

from collections.abc import Collection
from itertools import combinations

from y0.algorithm.identify import identify_outcomes
from y0.dsl import Variable
from y0.graph import NxMixedGraph

from sem2.model.construction.dag import _DAG


def project_to_admg(model: _DAG | str, *, observed: Collection[str]) -> NxMixedGraph:
    """Project a structural graph onto its observed variables.

    Parameters
    ----------
    model : SEM2, _DAG or str
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
    Compiler helpers are inlined before latent projection: they are computations,
    not additional latent common causes. Directed edges represent directed paths
    with latent internal nodes. Bidirected edges represent latent common causes
    with paths to both endpoints having only latent internal nodes.
    The supplied model and its parameters are not modified or sampled.
    """
    dag = _DAG(model) if isinstance(model, str) else model
    if not isinstance(dag, _DAG):
        raise TypeError("model must be a SEM2 model, parsed DAG, or formula string.")
    observed = _names(observed, "observed")
    scientific = {name for name, spec in dag.node_specs.items() if not spec.is_helper}
    if unknown := observed - scientific:
        raise ValueError(f"Unknown or internal observed variables: {sorted(unknown)}")

    # Inline helper inputs in topological order, stopping at scientific variables.
    parents: dict[str, set[str]] = {}
    for name in dag.topological_order:
        parents[name] = set()
        for parent in dag.node_specs[name].predecessors:
            if dag.node_specs[parent].is_helper:
                parents[name].update(parents[parent])
            else:
                parents[name].add(parent)

    children: dict[str, set[str]] = {name: set() for name in scientific}
    for child in scientific:
        for parent in parents[child]:
            children[parent].add(child)

    # First observed descendants: stop each path as soon as it reaches the margin.
    frontier: dict[str, set[str]] = {}
    for name in reversed(dag.topological_order):
        if name not in scientific:
            continue
        frontier[name] = set()
        for child in children[name]:
            frontier[name].update({child} if child in observed else frontier[child])

    directed = [(parent, child) for parent in sorted(observed)
                for child in sorted(frontier[parent])]
    bidirected = set()
    for latent in scientific - observed:
        bidirected.update(combinations(sorted(frontier[latent]), 2))
    return NxMixedGraph.from_str_edges(
        nodes=sorted(observed), directed=directed, undirected=sorted(bidirected),
    )


def is_identified(
    model: _DAG | str,
    *,
    observed: Collection[str],
    treatments: str | Collection[str],
    outcomes: str | Collection[str],
) -> bool:
    """Check graphical identification of an intervention distribution using ID.

    Parameters
    ----------
    model : SEM2, _DAG or str
        Structural model, parsed graph, or formula; no fitting is required.
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
