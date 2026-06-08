""" Parse the formula used to build a SEM2 into a DAG structure. Determine the type of SEM2."""
import re
from enum import Enum

from dataclasses import dataclass, field
from graphlib import TopologicalSorter, CycleError
import ast

# ══════════════════════════════════════════════════════════════════════════════
# SEM type
# ══════════════════════════════════════════════════════════════════════════════

class SEMType(Enum):
    """Enum representing the inferred type of a SEM.

    Types:
        - LINEAR_GAUSSIAN: all linear endogenous nodes, all exogenous nodes are Normal/MultivariateNormal.
        - LINEAR: all linear endogenous nodes, with non-Gaussian exogenous nodes.
        - NONLINEAR: all endogenous nodes are MLPs.
        - DISCRETE: all endogenous nodes are discrete.
        - MIXED: some endogenous nodes are continuous, some are discrete.

    Note: sem_type is descriptive metadata. It does not change core SEM
    behaviour. Its intended future use is to gate type-specific functionals
    (e.g. closed-form log-likelihood for LINEAR_GAUSSIAN) and guide the user
    toward a specialised implementation when one exists.
    """
    LINEAR_GAUSSIAN     = "linear_gaussian"
    LINEAR              = "linear"
    NONLINEAR           = "nonlinear"
    DISCRETE            = "discrete"
    MIXED               = "mixed"

    @staticmethod
    def is_continuous(sem_type: SEMType):
        """ If SEM is purely continuous"""
        return sem_type in {
            SEMType.LINEAR_GAUSSIAN,
            SEMType.LINEAR,
            SEMType.NONLINEAR,
        }

    @staticmethod
    def is_discrete(sem_type: SEMType):
        """ If SEM is purely discrete"""
        return sem_type in {
            SEMType.DISCRETE
        }

# Discrete distributions in torch.distributions. This is used for sem_type detection. Probably would be nice to move it elsewhere or compact all the formula parsing.
_DISCRETE_DIST_NAMES = frozenset({
    "Bernoulli",
    "Binomial",
    "Categorical",
    "Geometric",
    "Multinomial",
    "NegativeBinomial",
    "OneHotCategorical",
    "Poisson",
})

# ══════════════════════════════════════════════════════════════════════════════
# DAG
# ══════════════════════════════════════════════════════════════════════════════


def is_list_or_float_like(value):
    """
    Allows:
      - float or int values
      - lists/tuples containing only allowed values

    Rejects:
      - strings
      - dicts
      - booleans
      - None
    """
    if isinstance(value, bool):
        return False

    if isinstance(value, (int, float)):
        return True

    if isinstance(value, (list, tuple)):
        return all(is_list_or_float_like(item) for item in value)

    return False


class DAG:
    """Directed Acyclic Graph built from a model formula string.

    Owns all formula parsing and topological sorting. Stores only raw graph
    info (node names, rhs strings, predecessor lists, fixed coefficients)
    So no tensors, no device, no SEM2Node objects. (no pytorch related things basically)

    Attributes:
        model_formula:     The original formula string.
        formula_list:      Expanded, one-node-per-entry formula list.
        input_node:        Set of node names that appear on any LHS.
        node_specs:        Dict of node_name -> NodeSpec. Single source of
                           truth for graph structure : built once, read by SEM2.
        graph:             Dict of node_name -> list of predecessor names.
                           Derived from node_specs for topological sorting.
        topological_order: Node names in topologically sorted order.
        sem_type:          Inferred SEMType enum member. See SEMType above for
                           the full taxonomy.
    """

    @dataclass
    class NodeSpec:
        """Parsed specification of a single node from the formula string.
        NESTED inside DAG as it is a DAG implementation detail it is not part
        of the public API and should NOT be used outside of DAG and SEM2.

        Fields:
            name:         Node name (LHS of formula).
            rhs:          Raw RHS string after expansion.
            node_type:    One of "exogenous", "linear", "mlp".
            predecessors: Ordered list of predecessor node names.
            coefficients: One entry per predecessor for linear nodes.
                            float -> fixed at that value (e.g. 1.0, -0.5)
                            None  -> learned weight
                          Empty list for exogenous and mlp nodes.
            dist_name:    Distribution name string for exogenous nodes (e.g. "Normal"),
                          None for endogenous nodes.
            dist_params:  Raw parameter strings for exogenous nodes, None for endogenous.
        """
        name: str
        rhs: str
        node_type: str
        predecessors: list[str]
        coefficients: list[float | None] = field(default_factory=list)
        # Only for exogenous nodes:
        dist_name: str | None = None
        dist_params: tuple | None = None        # Tuple of (arguments, keyword_arguments)

    def __init__(self, model_formula: str):
        self.model_formula = model_formula
        self.formula_list, self.input_node = self._parse_formula(model_formula)

        # Build node_specs: single parse, stored as NodeSpec objects
        self.node_specs: dict[str, DAG.NodeSpec] = {}
        for formula in self.formula_list:
            if "~" in formula:
                lhs, rhs = formula.split("~")
                dist_name, dist_args, dist_kwargs = self._parse_exogenous_rhs(lhs, rhs)
                self.node_specs[lhs] = DAG.NodeSpec(
                    name=lhs, rhs=rhs, node_type="exogenous", predecessors=[],
                    dist_name=dist_name, dist_params=(dist_args, dist_kwargs),
                )
            elif "<-" in formula:
                lhs, rhs = formula.split("<-")
                if rhs.startswith("n("):
                    # Continuous neural network node
                    predecessors = rhs[2:-1].split(",")
                    self.node_specs[lhs] = DAG.NodeSpec(
                        name=lhs, rhs=rhs, node_type="mlp", predecessors=predecessors,
                    )
                elif rhs.startswith("d("):
                    predecessors = rhs[2:-1].split(",")
                    self.node_specs[lhs] = DAG.NodeSpec(
                        name=lhs, rhs=rhs, node_type="discrete", predecessors=predecessors,
                    )
                else:
                    # Linear node: parse coefficients and predecessors
                    terms = rhs.split("+")
                    parsed_terms = [self._parse_linear_term(t) for t in terms]
                    self.node_specs[lhs] = DAG.NodeSpec(
                        name=lhs,
                        rhs=rhs,
                        node_type="linear",
                        predecessors=[name for _, name in parsed_terms],
                        coefficients=[coeff for coeff, _ in parsed_terms],
                    )

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

        self.sem_type: SEMType = self._infer_sem_type()

    def static_order(self):
        return iter(self.topological_order)

    # ── SEM type inference ────────────────────────────────────────────────────

    def _infer_sem_type(self) -> SEMType:
        """Infer the SEM type from node_specs.

        Automatically infer the type of SEM:

        Types:
            - LINEAR_GAUSSIAN: all linear endogenous nodes, all exogenous nodes are Normal/MultivariateNormal.
            - LINEAR: all linear endogenous nodes, with non-Gaussian exogenous nodes.
            - NONLINEAR: all endogenous nodes are MLPs.
            - DISCRETE: all endogenous nodes are discrete.
            - MIXED: some endogenous nodes are continuous, some are discrete.
g
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

    # ── Exogenous rhs parsing (with error handling) ───────────────────────────

    def _parse_exogenous_rhs(self, node_name: str, rhs: str):
        """
        Parse a function call string like:
            "foo(1, a=2, b=[1, 2, 3])"
        Returns:
            ("foo", [1], {"a": 2, "b": [1, 2, 3]})
        Check if all values are numeric or lists.

        """
        node = ast.parse(rhs.strip(), mode="eval").body

        if not isinstance(node, ast.Call):
            raise ValueError("Input must be a function call")

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Handles things like module.func(...)
            func_name = ast.unparse(node.func)
        else:
            raise ValueError("Unsupported function name")

        args = [ast.literal_eval(arg) for arg in node.args]

        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise ValueError("**kwargs are not supported")
            kwargs[kw.arg] = ast.literal_eval(kw.value)

        # Ensure all arguments are valid, either int, float or list of int/float.
        all_values = args + list(kwargs.values())

        if not all(is_list_or_float_like(value) for value in all_values):
            raise ValueError(
                f"Invalid argument for node {node_name}. All arguments must be numeric or lists of numeric values. \n"
                f"  Your input: {rhs} \n"
            )
        return func_name, args, kwargs

    # ── Linear term parsing helper fx ───────────────────────────────────────

    def _parse_linear_term(self, term: str) -> tuple[float | None, str]:
        """Parse a single linear RHS term into (coefficient, predecessor_name).

        coefficient is None  -> weight is learned (no prefix).
        coefficient is float -> weight is fixed at that value.

        Examples:
            'x'      -> (None,  'x')
            '1*x'    -> (1.0,   'x')
            '-1*x'   -> (-1.0,  'x')
            '0.5*x'  -> (0.5,   'x')
            '-0.5*x' -> (-0.5,  'x')
        """
        match = re.match(r'^(-?[0-9]*\.?[0-9]+)\*(.+)$', term)
        if match:
            return float(match.group(1)), match.group(2)
        return None, term

    # ── Formula parsing ───────────────────────────────────────────────────────

    def _parse_formula(self, model_formula: str):
        formula_list = model_formula.replace(" ", "").split(";")

        expanded_list = []
        input_node = set()
        for formula in formula_list:
            vars_, formulas = self._expand_lhs(formula)
            input_node.update(vars_)
            expanded_list.extend(formulas)

        formula_list = []
        for formula in expanded_list:
            formula_list.extend(self._expand_rhs(formula))

        return formula_list, input_node

    def _expand_lhs(self, formula: str):
        if "~" in formula:
            sep = "~"
        elif "<-" in formula:
            sep = "<-"
        else:
            raise ValueError(f"Formula must contain '~' or '<-': {formula!r}")

        lhs, rhs = formula.replace(" ", "").split(sep)
        variables = lhs.split(",")
        return variables, [var + sep + rhs for var in variables]

    def _split_rhs(self, rhs: str, sep: str):
        """Split by sep but ignore separators inside parentheses."""
        rhs = rhs.replace(" ", "")
        terms, current, depth = [], "", 0
        for ch in rhs:
            if ch == sep and depth == 0:
                terms.append(current)
                current = ""
            else:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                current += ch
        if current:
            terms.append(current)
        return terms

    def _expand_rhs(self, formula: str):
        formula = formula.replace(" ", "")

        if "~" in formula:
            return [formula]
        if formula.count("<-") != 1:
            raise ValueError(f"Formula must contain exactly one '<-': {formula!r}")

        lhs, rhs = formula.split("<-")

        if "n(" not in rhs:
            return [formula]

        plus_terms = self._split_rhs(rhs, "+")
        if len(plus_terms) == 1 and rhs.startswith("n(") and rhs.count("n(") == 1:
            return [formula]  # single flat nonlinear term, nothing to expand

        expanded = []
        if len(plus_terms) == 1 and rhs.startswith("n("):
            # Single nested nonlinear term
            new_formula = lhs + "<-n("
            comma_terms = self._split_rhs(rhs[2:-1], ",")
            for i, term in enumerate(comma_terms):
                if term.startswith("n(") or "+" in term:
                    new_rv = lhs + ":" + str(i)
                    expanded.extend(self._expand_rhs(new_rv + "<-" + term))
                    new_formula += new_rv
                else:
                    new_formula += term
                if i < len(comma_terms) - 1:
                    new_formula += ","
            new_formula += ")"
        else:
            # Mix of linear and nonlinear plus-terms
            new_formula = lhs + "<-"
            for i, term in enumerate(plus_terms):
                if term.startswith("n("):
                    new_rv = lhs + ":" + str(i)
                    expanded.extend(self._expand_rhs(new_rv + "<-" + term))
                    new_formula += new_rv
                else:
                    new_formula += term
                if i < len(plus_terms) - 1:
                    new_formula += "+"

        expanded.append(new_formula)
        return expanded
