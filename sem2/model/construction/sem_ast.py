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

    Parameters
    ----------
    name : str
        Validated torch.distributions class name.
    args : list
        Positional numeric literals, including nested lists or tuples.
    kwargs : dict
        Named numeric literals, including nested lists or tuples.
    """
    name: str
    args: list
    kwargs: dict


@dataclass
class LinearTerm:
    """A single additive term on a linear RHS.

    coeff is None for a learned weight, or a float for a fixed weight
    (e.g. the 1.0 in ``y <- 1*x``).

    Parameters
    ----------
    coeff : float or None
        Fixed multiplier or None for a learned coefficient.
    parent : str
        Referenced predecessor name, including a generated helper when expanded.
    """
    coeff: float | None
    parent: str


@dataclass
class MechanismTerm:
    """A structural mechanism applied to parents: an MLP ``n(...)`` or a
    discrete ``d(...; levels=N)`` node. kind is "mlp" or "discrete".

    Parameters
    ----------
    kind : {"mlp", "discrete"}
        Mechanism family.
    parents : list[str]
        Parent names in mechanism input order.
    levels : int, optional
        Positive category count for discrete mechanisms; otherwise None.
    """
    kind: str
    parents: list[str]
    levels: int | None = None  # Required category count for discrete mechanisms.


@dataclass
class ExogenousDecl:
    """``name ~ Distribution(...)`` — a root node whose randomness is external.

    Parameters
    ----------
    name : str
        Declared node name.
    rhs : str
        Original right-hand-side expression, retained for display.
    dist : Distribution
        Parsed distribution name and literal arguments, not a live Torch law.
    """
    name: str
    rhs: str          # raw RHS substring, preserved for NodeSpec/repr parity
    dist: Distribution


@dataclass
class EndogenousDecl:
    """``name <- <structural expression>`` — a node determined by its parents.

    For linear nodes, ``terms`` is populated and ``mechanism`` is None.
    For mlp/discrete nodes, ``mechanism`` is populated and ``terms`` is empty.

    Parameters
    ----------
    name : str
        Declared or generated node name.
    rhs : str
        Expanded right-hand-side expression.
    node_type : {"linear", "mlp", "discrete"}
        Endogenous mechanism family.
    terms : list[LinearTerm], optional
        Additive linear terms in input order; defaults to an empty list.
    mechanism : MechanismTerm, optional
        Neural or discrete mechanism; None for linear equations.
    intercept : float, optional
        Formula-specified fixed linear offset, including zero. None leaves the
        intercept policy to the node configuration.
    is_helper : bool, default=False
        Whether formula expansion introduced this private computation node.
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

    Parameters
    ----------
    name : str
        Node name on the left-hand side of the formula.
    rhs : str
        Raw right-hand-side string after expansion.
    node_type : {"exogenous", "linear", "mlp", "discrete"}
        Mechanism family.
    predecessors : list[str]
        Parent names in input order.
    coefficients : list[float or None], optional
        One entry per linear parent: a number fixes the coefficient (e.g. 1.0
        or -0.5), and None marks a learned weight. Empty for other mechanisms.
    dist_name : str, optional
        Exogenous distribution name, e.g. Normal; None for endogenous nodes.
    dist_params : tuple, optional
        Exogenous (args, kwargs); None for endogenous nodes.
    intercept : float, optional
        Fixed linear intercept; None leaves it to the node configuration.
    is_helper : bool, default=False
        Whether expansion generated this private computation node.
    levels : int, optional
        Discrete category count; None for other mechanisms.
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
