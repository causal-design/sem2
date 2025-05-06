import torch
from torch import nn
import torch.distributions as dist
import re
from graphlib import TopologicalSorter
import scipy.stats as stats
from nn_models import MLP
import inspect

class rv_distribution:
    """Random distributions for nodes using torch.distributions, initialized from a string like 'Beta(0.5,0.5)'."""
    
    def __init__(self, dist_string: str, device: str = "cpu"):
        """
        Initialize a distribution from a string specifying the distribution and parameters.

        Args:
            dist_string (str): Distribution specification, e.g., 'Beta(0.5,0.5)', 'Normal(0.0,1.0)'.
            device (str): Device for tensor operations ('cpu' or 'cuda').

        Raises:
            ValueError: If the string format is invalid, distribution is not found, or parameters are incorrect.
        """
        self.device = device
        
        # Parse the distribution string
        match = re.match(r'^([A-Za-z]+)\((.*)\)$', dist_string.strip())
        if not match:
            raise ValueError(f"Invalid distribution string format: '{dist_string}'. Expected 'DistributionName(param1,param2,...)',e.g.,'Beta(0.5,0.5)'.")
        
        dist_name = match.group(1)  # e.g., 'Beta'
        params_str = match.group(2)  # e.g., '0.5,0.5'
        
        # Parse parameters
        if params_str.strip():
            try:
                params = [float(p.strip()) for p in params_str.split(',')]
            except ValueError:
                raise ValueError(f"Parameters in '{dist_string}' must be valid numbers, got '{params_str}'.")
        else:
            params = []

        # Dynamically get the distribution class from torch.distributions
        try:
            dist_class = getattr(dist, dist_name)
            if not isinstance(dist_class, type) or not issubclass(dist_class, dist.Distribution):
                raise ValueError(f"'{dist_name}' is not a valid torch.distributions distribution.")
        except AttributeError:
            raise ValueError(f"Distribution '{dist_name}' not found in torch.distributions.")

        # Inspect the distribution's __init__ parameters
        sig = inspect.signature(dist_class.__init__)
        valid_params = [p for p in sig.parameters.keys() if p not in ('self', 'args', 'kwargs')]
        
        # Check parameter count
        if len(params) > len(valid_params):
            raise ValueError(f"Too many parameters for {dist_name}: expected at most {len(valid_params)} ({valid_params}), got {len(params)}.")
        if not params and valid_params:
            raise ValueError(f"No parameters provided for {dist_name}, but it requires parameters: {valid_params}.")

        # Map parameters to named arguments
        processed_args = {}
        for i, param_value in enumerate(params):
            param_name = valid_params[i]
            processed_args[param_name] = torch.tensor(param_value, device=self.device, dtype=torch.float32)

        # Initialize the distribution
        try:
            self.dist = dist_class(**processed_args)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to initialize {dist_name} with parameters {processed_args}: {str(e)}")

    def sample(self, bs: int) -> torch.Tensor:
        """
        Sample from the distribution.

        Args:
            bs (int): Batch size (number of samples).

        Returns:
            torch.Tensor: Samples of shape (bs,).
        """
        return self.dist.sample((bs,))


class rv_sem2:
    """Parses a formula for a node, supporting random variables and non-linear relationships."""
    def __init__(self, formula: str, device="cpu"):
        self.device = device
        formula = formula.replace(" ", "")  # Remove spaces
        self.formula = formula  # Store original formula for debugging
        self.additional_nodes = []  # For multiple nodes in random variable formulas

        if "<-" in formula:
            # Deterministic node: e.g., "X3<-(ex1*ex2)#+(ex1-ex2-ex3)+ex3+n(X1+X2,ex1)" or "X3<-n(X1+X2,ex1)+(ex1*ex2)#+(ex1-ex2-ex3)+ex3"
            lhs, rhs = formula.split("<-")
            self.node = lhs
            self.is_random = False
            self.dist = None

            # Parse non-linear and linear parts
            nonlinear_match = re.search(r"n\((.*?)\)", rhs)
            if nonlinear_match:
                nonlinear_part = nonlinear_match.group(1)
                nonlinear_start, nonlinear_end = nonlinear_match.span()
                linear_part = (rhs[:nonlinear_start] + rhs[nonlinear_end:]).strip("+")
                self.nonlinear_args = self._parse_nonlinear_args(nonlinear_part)
            else:
                linear_part = rhs.strip("+")
                self.nonlinear_args = []

            # Parse linear part for single nodes and combination operations
            self.linear_terms = self._parse_linear_terms(linear_part)
            self.trainable_linear_terms = [term for term in self.linear_terms if not term[0].endswith("#")]
            self.fixed_linear_terms = [term for term in self.linear_terms if term[0].endswith("#")]
            # Extract predecessors from all terms
            self.predecessors = list(set(
                self._extract_predecessors_from_linear_terms() +
                self._extract_predecessors_from_nonlinear_args()
            ))

        elif "~" in formula:
            # Random node: e.g., "ex1,ex2,ex3~norm"
            lhs, rhs = formula.split("~")
            if not lhs or not rhs:
                raise ValueError(f"Invalid formula format: {formula}")
            nodes = [n.strip() for n in lhs.split(",") if n.strip()]
            if not nodes:
                raise ValueError(f"No valid nodes in formula: {formula}")
            if len(nodes) > 1:
                self.node = nodes[0]
                self.is_random = True
                self.dist = rv_distribution(rhs, device=device)
                self.predecessors = []
                self.trainable_linear_terms = []
                self.fixed_linear_terms = []
                self.additional_nodes = nodes[1:]
            else:
                self.node = lhs
                self.is_random = True
                self.dist = rv_distribution(rhs, device=device)
                self.predecessors = []
                self.trainable_linear_terms = []
                self.fixed_linear_terms = []
                self.additional_nodes = []
        else:
            raise ValueError(f"Formula must contain '<-' or '~': {formula}")

        self.npredecessors = len(self.predecessors)
        self.successors = []  # For tracking successors

    def _parse_linear_terms(self, linear_part: str):
        """Parse linear terms, e.g., '(ex1*ex2)#+(ex1+ex2+ex3)+ex3'."""
        if not linear_part:
            return []
        # Split by '+' but preserve content inside parentheses
        terms = []
        current_term = ""
        paren_count = 0
        for char in linear_part:
            if char == "(":
                paren_count += 1
            elif char == ")":
                paren_count -= 1
            elif char == "+" and paren_count == 0:
                if current_term:
                    terms.append(current_term)
                    current_term = ""
                continue
            current_term += char
        if current_term:
            terms.append(current_term)

        parsed_terms = []
        for term in terms:
            term = term.strip()
            if not term:
                continue
            # Check if term has fixed weight
            is_fixed = term.endswith("#")
            clean_term = term.rstrip("#")
            # Check if term is a combination operation (enclosed in parentheses)
            if clean_term.startswith("(") and clean_term.endswith(")"):
                expr = clean_term[1:-1]  # Remove parentheses
                for op in ["+", "*", "-", "/"]:
                    if op in expr:
                        vars = [v.strip() for v in expr.split(op) if v.strip()]
                        if len(vars) >= 2:
                            parsed_terms.append((term, op, vars))
                            break
                else:
                    raise ValueError(f"Invalid combination term: {clean_term}")
            else:
                # Single variable
                parsed_terms.append((term, "single", [clean_term]))
        return parsed_terms

    def _extract_predecessors_from_linear_terms(self):
        """Extract predecessors from linear terms."""
        predecessors = []
        for _, op, vars in self.linear_terms:
            predecessors.extend(vars)
        return list(set(predecessors))

    def _parse_nonlinear_args(self, nonlinear_part: str):
        """Parse non-linear arguments, e.g., 'X1+X2+X3,X1*X2,X1-X2,X1/X2,X1'."""
        args = [arg.strip() for arg in nonlinear_part.split(",") if arg.strip()]
        parsed_args = []
        for arg in args:
            for op in ["+", "*", "-", "/"]:
                if op in arg:
                    vars = [v.strip() for v in arg.split(op) if v.strip()]
                    if len(vars) >= 2:
                        parsed_args.append((op, vars))
                        break
            else:
                parsed_args.append(("single", [arg]))
        return parsed_args

    def _extract_predecessors_from_nonlinear_args(self):
        """Extract individual predecessors from non-linear args."""
        predecessors = []
        for op, vars in self.nonlinear_args:
            predecessors.extend(vars)
        return list(set(predecessors))

    def __repr__(self):
        return self.formula

class sem2(TopologicalSorter, nn.Module):
    """Structural Equation Model with support for non-linear relationships and fixed weights."""
    def __init__(self, model_formula: str, nonlinear_models: dict = None, device="cpu", mlp_config=None, use_bias=True):
        self._graph = {}
        TopologicalSorter.__init__(self, graph=self._graph)
        nn.Module.__init__(self)
        self.device = device
        nonlinear_models = nonlinear_models if nonlinear_models is not None else {}
        mlp_config = mlp_config if mlp_config is not None else {'hidden_dims': [100, 100], 'dropout': 0.0, 'prob': False}

        formula_list = model_formula.replace(" ", "").split(";")
        formula_list = [f for f in formula_list if f]
        rvs = []
        for formula in formula_list:
            rv = rv_sem2(formula, device=device)
            rvs.append(rv)
            for add_node in rv.additional_nodes:
                new_rv = rv_sem2(f"{add_node}~{formula.split('~')[1]}", device=device)
                rvs.append(new_rv)
        self._node2info = {rv.node: rv for rv in rvs}

        for node, v in self._node2info.items():
            self.add(node)
            if v.predecessors:
                self.add(node, *v.predecessors)
        for node, v in self._node2info.items():
            for p in v.predecessors:
                self._node2info[p].successors.append(node)

        self.layer = nn.ModuleDict()
        for node, v in self._node2info.items():
            if v.is_random:
                self.layer[node] = None
            else:
                linear_dim = len(v.trainable_linear_terms)
                if linear_dim > 0:
                    self.layer[f"{node}_linear"] = nn.Linear(linear_dim, 1, bias=use_bias)
                else:
                    self.layer[f"{node}_linear"] = None
                if v.nonlinear_args:
                    nonlinear_dim = len(v.nonlinear_args)
                    if node in nonlinear_models.keys():
                        self.layer[f"{node}_nonlinear"] = nonlinear_models.pop(node)
                    else:
                        self.layer[f"{node}_nonlinear"] = MLP(
                            input_dim=nonlinear_dim,
                            hidden_dims=mlp_config['hidden_dims'],
                            output_dim=1,
                            prob=mlp_config['prob'],
                            dropout=mlp_config['dropout']
                        )
                else:
                    self.layer[f"{node}_nonlinear"] = None
        assert not nonlinear_models, f"Unused model components: {nonlinear_models.keys()}"
        self.layer.to(device)

        self._topological_order = list(super(sem2, self).static_order())
        self._node2info = {node: self._node2info[node] for node in self._topological_order}
        self._node2index = {node: index for index, node in enumerate(self._topological_order)}

    def static_order(self):
        """Override static_order to return cached topological order."""
        return iter(self._topological_order)

    def forward(self, batch_size=1, set_nodes: dict[str, torch.Tensor] = None):
        set_nodes = set_nodes if set_nodes is not None else {}
        n_nodes = len(self._topological_order)
        x = torch.zeros(batch_size, n_nodes, device=self.device)
        for node in self._topological_order:
            v = self._node2info[node]
            index = self._node2index[node]
            if node in set_nodes:
                x[:, index] = set_nodes.pop(node).squeeze(-1)
            elif v.is_random:
                x[:, index] = v.dist.sample(batch_size)
            else:
                linear_output = 0.0
                if v.trainable_linear_terms:
                    linear_inputs = []
                    for _, op, vars in v.trainable_linear_terms:
                        var_indices = [self._node2index[v] for v in vars]
                        if op == "single":
                            input_val = x[:, var_indices[0]]
                        elif op == "+":
                            input_val = x[:, var_indices].sum(dim=1)
                        elif op == "*":
                            input_val = x[:, var_indices].prod(dim=1)
                        elif op == "-":
                            input_val = x[:, var_indices[0]]
                            for i in range(1, len(var_indices)):
                                input_val = input_val - x[:, var_indices[i]]
                        elif op == "/":
                            input_val = x[:, var_indices[0]]
                            for i in range(1, len(var_indices)):
                                denominator = x[:, var_indices[i]] + 1e-8
                                input_val = input_val / denominator
                        linear_inputs.append(input_val.unsqueeze(-1))
                    linear_input = torch.cat(linear_inputs, dim=-1)
                    linear_output = self.layer[f"{node}_linear"](linear_input).squeeze(-1)
                if v.fixed_linear_terms:
                    fixed_contribution = 0.0
                    for _, op, vars in v.fixed_linear_terms:
                        var_indices = [self._node2index[v] for v in vars]
                        if op == "single":
                            fixed_contribution = fixed_contribution + x[:, var_indices[0]]
                        elif op == "+":
                            fixed_contribution = fixed_contribution + x[:, var_indices].sum(dim=1)
                        elif op == "*":
                            fixed_contribution = fixed_contribution + x[:, var_indices].prod(dim=1)
                        elif op == "-":
                            temp = x[:, var_indices[0]]
                            for i in range(1, len(var_indices)):
                                temp = temp - x[:, var_indices[i]]
                            fixed_contribution = fixed_contribution + temp
                        elif op == "/":
                            temp = x[:, var_indices[0]]
                            for i in range(1, len(var_indices)):
                                denominator = x[:, var_indices[i]] + 1e-8
                                temp = temp / denominator
                            fixed_contribution = fixed_contribution + temp
                    linear_output = linear_output + fixed_contribution
                nonlinear_output = 0.0
                if v.nonlinear_args:
                    nonlinear_inputs = []
                    for op, vars in v.nonlinear_args:
                        var_indices = [self._node2index[v] for v in vars]
                        if op == "single":
                            input_val = x[:, var_indices[0]].unsqueeze(-1)
                        elif op == "+":
                            input_val = x[:, var_indices].sum(dim=1, keepdim=True)
                        elif op == "*":
                            input_val = x[:, var_indices].prod(dim=1, keepdim=True)
                        elif op == "-":
                            input_val = x[:, var_indices[0]]
                            for i in range(1, len(var_indices)):
                                input_val = input_val - x[:, var_indices[i]]
                            input_val = input_val.unsqueeze(-1)
                        elif op == "/":
                            input_val = x[:, var_indices[0]]
                            for i in range(1, len(var_indices)):
                                denominator = x[:, var_indices[i]] + 1e-8
                                input_val = input_val / denominator
                            input_val = input_val.unsqueeze(-1)
                        nonlinear_inputs.append(input_val)
                    nonlinear_input = torch.cat(nonlinear_inputs, dim=-1)
                    nonlinear_output = self.layer[f"{node}_nonlinear"](nonlinear_input).squeeze(-1)
                x[:, index] = linear_output + nonlinear_output
        assert not set_nodes, f"Unused set nodes: {set_nodes.keys()}"
        return x

    def _get_nodeinfo(self, node: str):
        try:
            return self._node2info[node]
        except KeyError:
            raise ValueError(f"Node {node} not found in the model.")