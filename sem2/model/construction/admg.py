"""The formula's mixed graph, before compiling exogenous cliques to DAG roots."""
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from itertools import combinations
import keyword

import networkx as nx

from .sem_ast import NodeSpec


@dataclass(frozen=True)
class _ADMG:
    """Named vertices and edges; bidirected loops on exogenous variables are implicit."""

    endogenous: frozenset[str]
    exogenous: frozenset[str]
    directed: frozenset[tuple[str, str]]
    bidirected: frozenset[tuple[str, str]]

    @classmethod
    def from_equations(cls, specs: dict[str, NodeSpec], edges: set[tuple[str, str]]) -> "_ADMG":
        endogenous = frozenset(n for n, s in specs.items() if not s.is_helper and s.node_type != "exogenous")

        def parents(name):
            # Nested expressions introduce deterministic helpers, not graph vertices.
            for p in specs[name].predecessors:
                if p in specs and specs[p].is_helper:
                    yield from parents(p)
                else:
                    yield p

        directed = frozenset((p, n) for n in endogenous for p in parents(n))
        declared = {n for n, s in specs.items() if s.node_type == "exogenous"}
        exogenous = frozenset(declared | ({p for p, _ in directed} - endogenous))
        for n in exogenous:
            if not n.isidentifier() or keyword.iskeyword(n):
                raise ValueError(f"Invalid exogenous variable name: {n!r}")
        endpoints = {n for edge in edges for n in edge}
        if unknown := endpoints - exogenous:
            raise ValueError(f"Bidirected edges require exogenous endpoints: {sorted(unknown)}")
        # An explicit self-loop adds nothing to an implicit one.
        return cls(endogenous, exogenous, directed, frozenset((a, b) for a, b in edges if a != b))

    def marginalize(self, observed: set[str]) -> "_ADMG":
        """Remove hidden vertices, preserving paths through hidden noncolliders."""
        vertices = self.endogenous | self.exogenous
        if unknown := observed - vertices:
            raise ValueError(f"Unknown or internal observed variables: {sorted(unknown)}")
        directed, bidirected = set(self.directed), set(self.bidirected)
        for h in sorted(vertices - observed):
            parents = {a for a, b in directed if b == h}
            children = {b for a, b in directed if a == h}
            spouses = {b if a == h else a for a, b in bidirected if h in (a, b)}
            directed = {edge for edge in directed if h not in edge}
            bidirected = {edge for edge in bidirected if h not in edge}
            directed.update((p, c) for p in parents for c in children)
            # a <- h -> b and a <-> h -> b induce a <-> b. Two spouses
            # do not: the two arrowheads meet at h, making it a collider.
            bidirected.update(combinations(sorted(children), 2))
            bidirected.update((min(s, c), max(s, c)) for s in spouses for c in children if s != c)
        return _ADMG(self.endogenous & observed, self.exogenous & observed,
                     frozenset(directed), frozenset(bidirected))

    def expand(self, specs: dict[str, NodeSpec]) -> dict[tuple[str, ...], str]:
        """Lower the parsed equations in place; return clique-to-source names."""
        noises = nx.Graph()
        noises.add_nodes_from(sorted(self.exogenous))
        noises.add_edges_from(sorted(self.bidirected))
        bundles: dict[str, list[str]] = {}
        sources = {}
        for members in nx.enumerate_all_cliques(noises):
            clique = tuple(sorted(members))
            if len(clique) == 1 and not noises.degree(clique[0]):
                name = clique[0]  # Isolated: keep its name and any supplied law.
            else:
                # ':' is reserved in user formulas. Each clique gets one draw.
                name = "U:" + ":".join(clique)
                for n in clique:
                    if n in specs:
                        raise ValueError(f"Cannot declare a distribution for bidirected variable {n!r}.")
                    bundles.setdefault(n, []).append(name)
            sources[clique] = name
            if name not in specs:
                specs[name] = NodeSpec(name, "Normal(0,1)", "exogenous", [],
                                       dist_name="Normal", dist_params=([0., 1.], {}))

        for s in specs.values():
            if not any(p in bundles for p in s.predecessors):
                continue
            if s.node_type == "linear":
                raise ValueError(f"Bidirected exogenous variable in linear term of {s.name!r} is unsupported.")
            # Within one n(...) or d(...), a shared input occurs only once.
            s.predecessors = list(dict.fromkeys(u for p in s.predecessors for u in bundles.get(p, [p])))
        return sources


def evaluation_order(specs: dict[str, NodeSpec]) -> list[str]:
    """Order equations with parents first; generated sources are roots."""
    try:
        return list(TopologicalSorter({n: s.predecessors for n, s in specs.items()}).static_order())
    except CycleError as e:
        raise ValueError(f"Formula contains a cycle: {e.args[1]}") from None
