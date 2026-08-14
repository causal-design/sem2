""" Build a SEM2 DAG structure from a parsed formula and infer the SEM type.

Formula parsing lives in formula_parser.FormulaParser; this module keeps the
graph structure (topological order, cycle detection) and SEM-type inference.
"""
# Defer annotation evaluation so forward references in type hints resolve lazily
# on Python < 3.14.
from __future__ import annotations

from graphlib import TopologicalSorter, CycleError

from .sem_ast import NodeSpec
from .formula_parser import FormulaParser
from .sem_types import SEMType


# ══════════════════════════════════════════════════════════════════════════════
# DAG
# ══════════════════════════════════════════════════════════════════════════════


class _DAG:
    """Directed Acyclic Graph built from a model formula string.

    Owns the graph structure and topological sorting. Stores only raw graph
    info (node names, rhs strings, predecessor lists, fixed coefficients)
    So no tensors, no device, no SEM2Node objects. (no pytorch related things basically)

    Attributes:
        model_formula:     The original formula string.
        input_node:        Set of node names that appear on any LHS.
        statements:        The parsed AST (list of Statement) for downstream tooling.
        unobserved_vars:   Ordered node names declared by unobs(...), if any.
        observed_vars:     Topological node order excluding unobserved_vars when
                           unobs(...) is declared; empty otherwise.
        node_specs:        Dict of node_name -> NodeSpec. Single source of
                           truth for graph structure : built once, read by SEM2.
        graph:             Dict of node_name -> list of predecessor names.
                           Derived from node_specs for topological sorting.
        topological_order: Node names in topologically sorted order.
        node_order:        Canonical node ordering for matrices over the node set
                           (a copy of topological_order).
        inferred_sem_type:          Inferred SEMType enum member. See SEMType above for
                           the full taxonomy.
    """

    # NodeSpec is the lowered IR shared with SEM2; defined in sem_ast and
    # re-exported here for convenience.
    NodeSpec = NodeSpec

    def __init__(self, model_formula: str):
        self.model_formula = model_formula

        # Parse the formula into lowered NodeSpec objects. All surface-syntax
        # handling (and distribution validation) lives in FormulaParser; _DAG
        # only consumes the resulting specs.
        parser = FormulaParser(model_formula)
        self.node_specs, self.input_node = parser.parse()
        self.statements = parser.statements  # AST, for downstream tooling
        self.unobserved_vars = list(parser.unobserved_vars)
        self._unobserved_declared = parser._unobserved_declared

        self._validate_referenced_nodes()

        # Derive graph from node_specs for topological sorting
        self.graph: dict[str, list[str]] = {
            name: ns.predecessors for name, ns in self.node_specs.items()
        }

        try:
            self.topological_order: list[str] = list(
                TopologicalSorter(self.graph).static_order()
            )
        except CycleError as e:
            raise ValueError(
                f"Formula contains a cycle. Please check your formula: {e.args[1]}"
            ) from None

        # Canonical node ordering for anything matrix-valued over the node set
        # (topological order makes coefficient matrices strictly lower-triangular).
        self.node_order: list[str] = list(self.topological_order)
        self._node_index: dict[str, int] = {n: i for i, n in enumerate(self.node_order)}

        if self._unobserved_declared:
            hidden = set(self.unobserved_vars)
            self.observed_vars = [n for n in self.topological_order if n not in hidden]
        else:
            self.observed_vars = []

        # Infer and store the SEM type from the formula. The desired-vs-inferred
        # comparison (and any warning/error) is handled by the SEM2 layer, which
        # knows which concrete SEM2 subclass the user actually instantiated.
        self.inferred_sem_type: SEMType = self._infer_sem_type()

    def static_order(self):
        return iter(self.topological_order)

    def _index(self, nodes: list[str] | None) -> list[int]:
        nodes = nodes if nodes is not None else self.node_order
        return [self._node_index[n] for n in nodes]

    def _validate_referenced_nodes(self):
        defined = set(self.node_specs)
        referenced = {
            parent
            for ns in self.node_specs.values()
            for parent in ns.predecessors
        }
        missing = sorted(referenced - defined)
        if missing:
            raise ValueError(
                f"Formula references undefined parent nodes: {missing}"
            )

    # ── SEM type inference ────────────────────────────────────────────────────

    def _infer_sem_type(self) -> SEMType:
        """Infer the SEM type from node_specs.

        Automatically infer the type of SEM:

        Types:
            - LINEAR_GAUSSIAN: all linear endogenous nodes, all exogenous nodes are scalar Normal.
            - LINEAR: all linear endogenous nodes, with non-Gaussian exogenous nodes.
            - NONLINEAR: all endogenous nodes are MLPs.
            - DISCRETE: all endogenous nodes are discrete.
            - MIXED: some endogenous nodes are continuous, some are discrete.

        Returns: SEMType.
        """
        exog_specs = [ns for ns in self.node_specs.values() if ns.node_type == "exogenous"]
        endog_specs = [ns for ns in self.node_specs.values() if ns.node_type != "exogenous"]

        # Any MLP present -> nonlinear; otherwise linear (including exog-only).
        has_mlp = any(ns.node_type in ("mlp", "discrete") for ns in endog_specs)
        is_linear = not has_mlp
        if is_linear:
            # Is exogenous noise all Gaussian?
            if all(ns.dist_name == "Normal" for ns in exog_specs):
                return SEMType.LINEAR_GAUSSIAN
            else:
                return SEMType.LINEAR
        else:
            # Is discrete, continuous or mixed?
            n_discrete = len([ns for ns in endog_specs if ns.node_type == "discrete"])
            n_continuous = len(endog_specs) - n_discrete
            if n_discrete == 0:
                return SEMType.NONLINEAR
            elif n_continuous == 0:
                return SEMType.DISCRETE
            else:
                return SEMType.MIXED
