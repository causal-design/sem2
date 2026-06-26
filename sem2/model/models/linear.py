# Defer annotation evaluation so forward references in type hints resolve lazily
# on Python < 3.14.
from __future__ import annotations
import logging
import torch
from torch import Tensor

from ..components.nodes import NodeExogenous, NodeLinear
from ..construction import SEMType
from .base import SEM2


# ══════════════════════════════════════════════════════════════════════════════
# SEM2Linear
# ══════════════════════════════════════════════════════════════════════════════

class SEM2Linear(SEM2):
    """Linear SEM2: every endogenous node is a linear function of its parents.

    For a linear SEM the whole model is summarised by two matrices over the node
    set (ordered by `node_order`, the topological order):

        - the path-coefficient matrix B, where B[i, j] is the coefficient of
          node j in node i's structural equation (0 if j is not a parent of i),
        - the model-implied covariance Sigma = (I - B)^-1 Omega (I - B)^-T,
          where Omega is the diagonal covariance of the exogenous inputs.

    Only scalar nodes (out_dim == 1) and scalar exogenous distributions are
    supported; multi-output nodes or vector-valued exogenous distributions raise
    a clear error.
    """
    sem_type = SEMType.LINEAR

    def _check_sem_type(self):
        """A linear SEM requires only linear endogenous mechanisms. Validate
        against the type inferred from the formula (LINEAR or LINEAR_GAUSSIAN),
        rejecting nonlinear/discrete formulas up front.
        """
        if not SEMType.is_linear(self.inferred_sem_type):
            raise ValueError(
                f'SEM2Linear can only be used with a linear model. '
                f'Found "{self.inferred_sem_type.value}" model.'
            )
        logging.info("Building a linear SEM2 model.")

    def __init__(self, model_formula: str, node_cfgs: dict | None = None, device="cpu"):
        super().__init__(model_formula, node_cfgs, device=device)
        self.device = device

        # Canonical node ordering for all matrices (topological order makes B
        # strictly lower-triangular, so I - B is unit lower-triangular).
        self.node_order: list[str] = list(self.topological_order)
        self._node_index: dict[str, int] = {n: i for i, n in enumerate(self.node_order)}

        self._assert_scalar_nodes()
        # Validate that path coefficients can be extracted (issue: "Call coef_matrix").
        self.coef_matrix()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _assert_scalar_nodes(self):
        """coef_matrix / cov_matrix assume every node is scalar (out_dim == 1)."""
        bad = [n for n in self.node_order if self.node_dict[n].out_dim != 1]
        if bad:
            raise NotImplementedError(
                f"SEM2Linear matrices currently support only scalar nodes "
                f"(out_dim == 1). Multi-output nodes: {bad}."
            )

        vector_exog = []
        for name in self.node_order:
            node = self.node_dict[name]
            if isinstance(node, NodeExogenous) and (node.dist.batch_shape or node.dist.event_shape):
                vector_exog.append(name)
        if vector_exog:
            raise NotImplementedError(
                f"SEM2Linear matrices currently support only scalar independent "
                f"exogenous distributions. Vector-valued or multivariate "
                f"exogenous nodes: {vector_exog}."
            )

    def _index(self, nodes: list[str] | None) -> list[int]:
        nodes = nodes if nodes is not None else self.node_order
        return [self._node_index[n] for n in nodes]

    # ── path coefficients ─────────────────────────────────────────────────────

    def coef_matrix(self) -> Tensor:
        """Path-coefficient matrix B over all nodes in `node_order`.

        B[i, j] is the (effective, i.e. fixed-or-learned) coefficient of node j
        in node i's linear equation, and 0 if j is not a parent of i. Exogenous
        nodes have an all-zero row. Recomputed from current weights on each call.
        """
        N = len(self.node_order)
        B = torch.zeros(N, N, device=self.device)
        for name in self.node_order:
            node = self.node_dict[name]
            if isinstance(node, NodeLinear):
                # effective_weight is [1, input_dim]; scalar parents -> one col each.
                w = node.effective_weight().reshape(-1)
                for col, parent in enumerate(node.predecessors):
                    B[self._node_index[name], self._node_index[parent]] = w[col]
        return B

    # ── model-implied mean and covariance ─────────────────────────────────────

    def mean_vector(self) -> Tensor:
        """Model-implied mean over all nodes: mu = (I - B)^-1 c, where c[i] is the
        exogenous distribution mean for exogenous nodes and the bias term for
        linear nodes (0 if the node has no bias).
        """
        N = len(self.node_order)
        B = self.coef_matrix()
        c = torch.zeros(N, device=self.device)
        for name in self.node_order:
            node = self.node_dict[name]
            i = self._node_index[name]
            if isinstance(node, NodeExogenous):
                mean = torch.as_tensor(node.dist.mean, device=self.device).reshape(-1)[0]
                if not torch.isfinite(mean):
                    raise ValueError(
                        f"SEM2Linear mean_vector requires finite exogenous means. "
                        f"Node '{name}' has mean {mean}."
                    )
                c[i] = mean
            elif isinstance(node, NodeLinear) and node.bias() is not None:
                c[i] = node.bias().reshape(-1)[0]
        eye = torch.eye(N, device=self.device)
        return torch.linalg.solve(eye - B, c)

    def cov_matrix(self, nodes: list[str] | None = None) -> Tensor:
        """Model-implied covariance Sigma = (I - B)^-1 Omega (I - B)^-T.

        Omega is the diagonal covariance of the exogenous inputs (the variance of
        each exogenous node's distribution; 0 for endogenous nodes). If `nodes`
        is given, the covariance is restricted to those nodes (e.g. the observed
        variables), in the given order.
        """
        N = len(self.node_order)
        B = self.coef_matrix()

        omega = torch.zeros(N, device=self.device)
        for name in self.node_order:
            node = self.node_dict[name]
            if isinstance(node, NodeExogenous):
                variance = torch.as_tensor(
                    node.dist.variance, device=self.device
                ).reshape(-1)[0]
                if not torch.isfinite(variance):
                    raise ValueError(
                        f"SEM2Linear covariance requires finite second moments. "
                        f"Node '{name}' has variance {variance}."
                    )
                omega[self._node_index[name]] = variance

        eye = torch.eye(N, device=self.device)
        a_inv = torch.linalg.inv(eye - B)
        sigma = a_inv @ torch.diag(omega) @ a_inv.T

        if nodes is not None:
            idx = torch.tensor(self._index(nodes), device=self.device)
            sigma = sigma.index_select(0, idx).index_select(1, idx)
        return sigma
