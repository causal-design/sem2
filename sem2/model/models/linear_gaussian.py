"""Linear Gaussian SEMs with covariance zeros specified by bidirected edges."""
import torch
from torch import nn

from ..construction.sem_ast import NodeSpec
from ..construction.sem_types import SEMType
from .linear import SEM2Linear


class SEM2LinearGaussian(SEM2Linear):
    """Scalar linear equations driven by a single jointly Gaussian noise vector.

    Absent bidirected edges fix covariances to zero. An undeclared noise has
    zero mean and learned scale initialized at one. A declaration fixes its
    marginal mean and scale, leaving its permitted correlations learnable.
    """

    sem_type = SEMType.LINEAR_GAUSSIAN

    def _compile(self, parser):
        specs, graph = parser.parse(joint_gaussian=True)
        self.noise_names = tuple(sorted(graph.exogenous))
        self.free_scales = tuple(i for i, n in enumerate(self.noise_names) if n not in specs)
        for n in self.noise_names:
            specs.setdefault(n, NodeSpec(n, "Normal(0,1)", "exogenous", [],
                                         dist_name="Normal", dist_params=([0., 1.], {})))
        return specs, graph, {}

    def __init__(self, model_formula, node_cfgs=None, device="cpu"):
        super().__init__(model_formula, node_cfgs, device)
        self.log_scales = nn.Parameter(torch.zeros(len(self.free_scales), device=device))
        self.edge_weights = nn.Parameter(torch.zeros(len(self.admg.bidirected), device=device))
        index = {n: i for i, n in enumerate(self.noise_names)}
        pairs = [(index[a], index[b]) for a, b in sorted(self.admg.bidirected)]
        self.register_buffer("_edges", torch.tensor(pairs, device=device, dtype=torch.long).reshape(-1, 2).T)
        laws = [self.node_dict[n].dist for n in self.noise_names]
        self.register_buffer("_locations", torch.tensor([d.loc.item() for d in laws], device=device))
        self.register_buffer("_base_scales", torch.tensor([d.scale.item() for d in laws], device=device))

    def noise_covariance(self):
        """Exogenous covariance in ``noise_names`` order (positive-definite interior)."""
        scale = self._base_scales.clone()
        if self.free_scales:
            scale[list(self.free_scales)] = self.log_scales.exp()
        corr = torch.eye(len(scale), device=scale.device, dtype=scale.dtype)
        if self.edge_weights.numel():
            i, j = self._edges
            raw = torch.zeros_like(corr)
            raw[i, j] = self.edge_weights
            raw = raw + raw.T
            # Trace(raw)=0; its least eigenvalue is nonpositive. Radial scaling
            # leaves zeros unchanged and covers every PD correlation with this
            # pattern: the inverse at R is raw=(R-I)/lambda_min(R).
            corr = corr + raw / (1 - torch.linalg.eigvalsh(raw)[0])
        return scale[:, None] * corr * scale

    def _innovation_covariance(self, reference):
        matrix = reference.new_zeros(reference.shape)
        ix = torch.tensor([self._node_index[n] for n in self.noise_names],
                          dtype=torch.long, device=matrix.device)
        matrix[ix[:, None], ix] = self.noise_covariance().to(matrix)
        return matrix

    def _joint_draws(self, batch_size):
        if not self.noise_names:
            return {}
        factor = torch.linalg.cholesky(self.noise_covariance())
        draws = torch.randn(batch_size, len(factor), device=factor.device, dtype=factor.dtype)
        draws = draws @ factor.T + self._locations
        return dict(zip(self.noise_names, draws.split(1, dim=1)))
