"""Abstract syntax tree (AST) for the SEM2 formula language.

These typed nodes are the *frontend* representation produced by FormulaParser.
They are then *lowered* into NodeSpec objects, which the DAG / SEM2 layers
consume. Keeping the AST separate from NodeSpec gives future language features
(interventions ``do(...)``, query expressions, named mechanisms) a typed place
to grow, instead of bolting more string-matching onto the parser.

Pipeline:  formula string --(FormulaParser.parse)--> AST --(lower)--> NodeSpec
"""
from __future__ import annotations
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════════
# AST nodes (frontend IR)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Distribution:
    """A named exogenous distribution call, e.g. Normal(0, 1) or
    Categorical(probs=[1, 1, 1]). args/kwargs are already evaluated to numbers.
    """
    name: str
    args: list
    kwargs: dict


@dataclass
class LinearTerm:
    """A single additive term on a linear RHS.

    coeff is None for a learned weight, or a float for a fixed weight
    (e.g. the 1.0 in ``y <- 1*x``).
    """
    coeff: float | None
    parent: str


@dataclass
class MechanismTerm:
    """A structural mechanism applied to parents: an MLP ``n(...)`` or a
    discrete ``d(...; levels=N)`` node. kind is "mlp" or "discrete".
    """
    kind: str
    parents: list[str]
    levels: int | None = None  # Required category count for discrete mechanisms.


@dataclass
class ExogenousDecl:
    """``name ~ Distribution(...)`` — a root node whose randomness is external."""
    name: str
    rhs: str          # raw RHS substring, preserved for NodeSpec/repr parity
    dist: Distribution


@dataclass
class EndogenousDecl:
    """``name <- <structural expression>`` — a node determined by its parents.

    For linear nodes, ``terms`` is populated and ``mechanism`` is None.
    For mlp/discrete nodes, ``mechanism`` is populated and ``terms`` is empty.
    """
    name: str
    rhs: str          # raw RHS substring, preserved for NodeSpec/repr parity
    node_type: str    # "linear" | "mlp" | "discrete"
    terms: list[LinearTerm] = field(default_factory=list)
    mechanism: MechanismTerm | None = None
    intercept: float | None = None  # None -> estimated; number -> fixed
    is_helper: bool = False


# A single parsed formula statement.
Statement = ExogenousDecl | EndogenousDecl


# ══════════════════════════════════════════════════════════════════════════════
# NodeSpec (lowered IR consumed by _DAG and SEM2)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NodeSpec:
    """Lowered specification of a single node, consumed by _DAG and SEM2.

    This is the stable interface between the parsing frontend and the graph /
    model layers; it carries no tensors, device, or pytorch objects.

    Fields:
        name:         Node name (LHS of formula).
        rhs:          Raw RHS string after expansion.
        node_type:    One of "exogenous", "linear", "mlp", "discrete".
        predecessors: Ordered list of predecessor node names.
        coefficients: One entry per predecessor for linear nodes.
                        float -> fixed at that value (e.g. 1.0, -0.5)
                        None  -> learned weight
                      Empty list for exogenous and mlp nodes.
        dist_name:    Distribution name string for exogenous nodes (e.g. "Normal"),
                      None for endogenous nodes.
        dist_params:  (args, kwargs) tuple for exogenous nodes, None otherwise.
        intercept:    Fixed linear intercept, or None to estimate it.
        levels:       Number of categories for discrete mechanisms, otherwise None.
        is_helper:    Whether expansion generated this private computation node.
    """
    name: str
    rhs: str
    node_type: str
    predecessors: list[str]
    coefficients: list[float | None] = field(default_factory=list)
    # Only for exogenous nodes:
    dist_name: str | None = None
    dist_params: tuple | None = None        # Tuple of (arguments, keyword_arguments)
    intercept: float | None = None
    is_helper: bool = False
    levels: int | None = None
