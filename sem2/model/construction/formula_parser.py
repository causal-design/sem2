"""Frontend for the SEM2 formula language.

FormulaParser turns a formula string into an AST (see sem_ast), validates it,
and lowers it to NodeSpec objects. This is the natural home for everything that
knows about the *surface syntax* of the language, including which distributions
exist and which nodes are declared hidden via unobs(...) — so _DAG can stay a
pure graph structure with no pytorch knowledge.
"""
from __future__ import annotations
import re
import ast
import difflib

from .sem_ast import (
    Distribution,
    LinearTerm,
    MechanismTerm,
    ExogenousDecl,
    EndogenousDecl,
    Statement,
    NodeSpec,
)


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


# ══════════════════════════════════════════════════════════════════════════════
# Distribution registry
# ══════════════════════════════════════════════════════════════════════════════

class DistributionRegistry:
    """The set of exogenous distributions the formula language recognises.

    Names are sourced from torch.distributions so the language stays in sync
    with the backend. The registry is the single source of truth for validating
    distribution names and suggesting fixes for typos, keeping that knowledge in
    the language frontend rather than scattered through node construction.
    """

    def __init__(self):
        import torch.distributions as D

        self._names = {
            name
            for name in dir(D)
            if isinstance(getattr(D, name), type)
            and issubclass(getattr(D, name), D.Distribution)
            and name != "Distribution"
        }

    def __contains__(self, name: str) -> bool:
        return name in self._names

    def validate(self, name: str):
        """Raise ValueError (with a typo suggestion if possible) for an unknown name."""
        if name not in self._names:
            suggestion = difflib.get_close_matches(name, self._names, n=1)
            hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
            raise ValueError(f"Unknown distribution '{name}'.{hint}")


# Shared across parser instances; building the name set touches torch once.
_DISTRIBUTIONS = DistributionRegistry()


# ══════════════════════════════════════════════════════════════════════════════
# FormulaParser
# ══════════════════════════════════════════════════════════════════════════════

class FormulaParser:
    """Parse a SEM2 model formula into an AST and lower it to NodeSpec objects.

    Usage:
        node_specs, input_node = FormulaParser(formula).parse()

    After parse(), ``self.statements`` holds the AST (the typed parse of the
    formula), which downstream tooling (interventions, queries) can build on.
    """

    def __init__(self, model_formula: str):
        self.model_formula = model_formula
        self.statements: list[Statement] = []
        self.unobserved_vars: list[str] = []
        self._unobserved_declared = False

    def parse(self) -> tuple[dict[str, NodeSpec], set[str]]:
        """Parse the formula, returning (node_specs, input_node).

        node_specs maps node name -> lowered NodeSpec. input_node is the set of
        names that appear on any LHS.
        """
        formula_list, input_node = self._parse_formula(self.model_formula)
        self.statements = [self._parse_statement(f) for f in formula_list]
        node_specs = {st.name: self._lower(st) for st in self.statements}
        self._validate_unobserved_vars(node_specs)
        return node_specs, input_node

    # ── Statement parsing (string -> AST) ─────────────────────────────────────

    def _parse_statement(self, formula: str) -> Statement:
        """Parse one fully-expanded ``name <sep> rhs`` formula into an AST node."""
        if "~" in formula:
            lhs, rhs = formula.split("~")
            dist_name, dist_args, dist_kwargs = self._parse_exogenous_rhs(lhs, rhs)
            return ExogenousDecl(
                name=lhs, rhs=rhs,
                dist=Distribution(name=dist_name, args=dist_args, kwargs=dist_kwargs),
            )

        if "<-" in formula:
            lhs, rhs = formula.split("<-")
            if rhs.startswith("n("):
                # Continuous neural network node
                parents = rhs[2:-1].split(",")
                return EndogenousDecl(
                    name=lhs, rhs=rhs, node_type="mlp",
                    mechanism=MechanismTerm(kind="mlp", parents=parents),
                )
            if rhs.startswith("d("):
                parents = rhs[2:-1].split(",")
                return EndogenousDecl(
                    name=lhs, rhs=rhs, node_type="discrete",
                    mechanism=MechanismTerm(kind="discrete", parents=parents),
                )
            # Linear node: parse coefficients and predecessors
            terms = [self._parse_linear_term(t) for t in rhs.split("+")]
            return EndogenousDecl(
                name=lhs, rhs=rhs, node_type="linear",
                terms=[LinearTerm(coeff=c, parent=p) for c, p in terms],
            )

        raise ValueError(f"Formula must contain '~' or '<-': {formula!r}")

    # ── Lowering (AST -> NodeSpec) ────────────────────────────────────────────

    def _lower(self, st: Statement) -> NodeSpec:
        """Lower an AST statement into the NodeSpec consumed by _DAG / SEM2."""
        if isinstance(st, ExogenousDecl):
            return NodeSpec(
                name=st.name, rhs=st.rhs, node_type="exogenous", predecessors=[],
                dist_name=st.dist.name, dist_params=(st.dist.args, st.dist.kwargs),
            )
        # EndogenousDecl
        if st.node_type == "linear":
            return NodeSpec(
                name=st.name, rhs=st.rhs, node_type="linear",
                predecessors=[t.parent for t in st.terms],
                coefficients=[t.coeff for t in st.terms],
            )
        # mlp / discrete mechanism
        return NodeSpec(
            name=st.name, rhs=st.rhs, node_type=st.node_type,
            predecessors=list(st.mechanism.parents),
        )

    # ── Exogenous rhs parsing (with error handling) ───────────────────────────

    def _parse_exogenous_rhs(self, node_name: str, rhs: str):
        """
        Parse a function call string like:
            "foo(1, a=2, b=[1, 2, 3])"
        Returns:
            ("foo", [1], {"a": 2, "b": [1, 2, 3]})
        Check if all values are numeric or lists.

        """
        try:
            node = ast.parse(rhs.strip(), mode="eval").body
        except SyntaxError as exc:
            raise ValueError(
                f"Could not parse the distribution for node '{node_name}'. "
                f"Expected a call like 'Normal(0, 1)', got: {rhs.strip()!r}"
            ) from exc

        if not isinstance(node, ast.Call):
            raise ValueError(
                f"Could not parse the distribution for node '{node_name}'. "
                f"Expected a call like 'Normal(0, 1)', got: {rhs.strip()!r}"
            )

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Handles things like module.func(...)
            func_name = ast.unparse(node.func)
        else:
            raise ValueError("Unsupported function name")

        # Reject unknown distribution names up front, with a typo suggestion.
        _DISTRIBUTIONS.validate(func_name)

        try:
            args = [ast.literal_eval(arg) for arg in node.args]
            kwargs = {}
            for kw in node.keywords:
                if kw.arg is None:
                    raise ValueError("**kwargs are not supported")
                kwargs[kw.arg] = ast.literal_eval(kw.value)
        except ValueError as exc:
            # "**kwargs are not supported" is intentional; re-raise it unchanged.
            if "**kwargs" in str(exc):
                raise
            raise ValueError(
                f"Non-numeric distribution parameter for node '{node_name}': "
                f"{rhs.strip()!r}. Parameters must be numbers or lists of numbers."
            ) from exc

        # Ensure all arguments are valid, either int, float or list of int/float.
        all_values = args + list(kwargs.values())

        if not all(is_list_or_float_like(value) for value in all_values):
            raise ValueError(
                f"Invalid argument for node {node_name}. All arguments must be numeric or lists of numeric values. \n"
                f"  Your input: {rhs} \n"
            )
        return func_name, args, kwargs

    # ── Linear term parsing helper fx ───────────────────────────────────────

    @staticmethod
    def _parse_linear_term(term: str) -> tuple[float | None, str]:
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
            if formula.startswith("unobs("):
                self._parse_unobserved_statement(formula)
                continue
            vars_, formulas = self._expand_lhs(formula)
            input_node.update(vars_)
            expanded_list.extend(formulas)

        formula_list = []
        for formula in expanded_list:
            formula_list.extend(self._expand_rhs(formula))

        return formula_list, input_node

    def _parse_unobserved_statement(self, formula: str):
        match = re.match(r"^unobs\((.*)\)$", formula)
        if match is None:
            raise ValueError(
                f"Could not parse unobserved-variable statement: {formula!r}. "
                "Expected unobs(x1,x2,...)."
            )
        body = match.group(1)
        variables = [] if body == "" else body.split(",")
        if any(v == "" for v in variables):
            raise ValueError(
                f"Could not parse unobserved-variable statement: {formula!r}. "
                "Expected comma-separated variable names."
            )
        seen = set(self.unobserved_vars)
        duplicates = []
        for v in variables:
            if v in seen:
                duplicates.append(v)
            seen.add(v)
        if duplicates:
            raise ValueError(f"Duplicate unobserved variables: {duplicates}")
        self._unobserved_declared = True
        self.unobserved_vars.extend(variables)

    def _validate_unobserved_vars(self, node_specs: dict[str, NodeSpec]):
        missing = [v for v in self.unobserved_vars if v not in node_specs]
        if missing:
            raise ValueError(f"Unobserved variables are not defined in the formula: {missing}")

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
