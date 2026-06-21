"""Set-DEQ model: a permutation-equivariant update iterated to a fixed point.

The latent Z (B, N, d_latent) is equivariant under row permutations of the input
set X (B, N, d_in); the readout pools over the set dimension, making the final
prediction permutation-invariant. The fixed point Z* is the equilibrium
representation whose properties (path independence, exact unlearning) we probe.

Two update blocks are provided:
  - DeepSetsUpdate: z_i <- MLP([z_i, x_i, mean_j z_j])  (cheap, Lipschitz-analyzable)
  - AttnUpdate:     a single self-attention block with the input injected
                    (more expressive; uniqueness is harder to reason about)

`spectral` optionally spectral-normalizes the DeepSets MLP, our first knob for
imposing contractivity when path independence does not come for free.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.utils.parametrizations import spectral_norm
except Exception:  # older torch
    from torch.nn.utils import spectral_norm

from torchdeq import get_deq

from .solver import fixed_point_solve


def _maybe_sn(linear, on):
    return spectral_norm(linear) if on else linear


def _weighted_pool(z, w):
    """Permutation-invariant pooled context over the set dimension.

    w is an optional (B, N, 1) presence weight, the continuous relaxation of set
    membership that the Jacobian probe differentiates: w_i = 1 keeps point i,
    w_i = 0 removes it. With w=None this is exactly the plain mean (so the
    trained forward pass is unchanged); with w given it is the weighted mean
    (sum_j w_j z_j) / (sum_j w_j), whose derivative dpool/dw_k is the only
    channel through which removing point k perturbs the equilibrium.
    """
    if w is None:
        return z.mean(dim=1, keepdim=True)
    return (w * z).sum(dim=1, keepdim=True) / (w.sum(dim=1, keepdim=True) + 1e-8)


class DeepSetsUpdate(nn.Module):
    """Permutation-equivariant DeepSets update with mean-pool context."""

    def __init__(self, d_latent, d_in, hidden, spectral=False):
        super().__init__()
        self.net = nn.Sequential(
            _maybe_sn(nn.Linear(2 * d_latent + d_in, hidden), spectral),
            nn.ReLU(),
            _maybe_sn(nn.Linear(hidden, d_latent), spectral),
        )

    def forward(self, z, x, w=None):
        pool = _weighted_pool(z, w).expand_as(z)
        return self.net(torch.cat([z, x, pool], dim=-1))


class AttnUpdate(nn.Module):
    """Single multi-head self-attention block with input injection."""

    def __init__(self, d_latent, d_in, hidden, n_heads=4, spectral=False):
        super().__init__()
        del spectral  # spectral norm on attention deferred to a later layer
        self.x_proj = nn.Linear(d_in, d_latent)
        self.attn = nn.MultiheadAttention(d_latent, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_latent)
        self.ff = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.ReLU(), nn.Linear(hidden, d_latent)
        )
        self.norm2 = nn.LayerNorm(d_latent)

    def forward(self, z, x, w=None):
        if w is not None:
            # Attention has no single additive sufficient statistic to weight:
            # presence enters through every query-key-value interaction, so the
            # clean w:1->0 removal knob does not exist here (the same reason it is
            # not cleanly federatable). For attention we characterize sensitivity
            # via the spectral radius / IFT amplifier rather than dZ*/dw_k.
            raise NotImplementedError("AttnUpdate has no presence-weight knob; "
                                      "use spectral_radius-based sensitivity")
        zx = z + self.x_proj(x)
        a, _ = self.attn(zx, zx, zx, need_weights=False)
        z = self.norm1(z + a)
        z = self.norm2(z + self.ff(z))
        return z


class NormDeepSetsUpdate(nn.Module):
    """Controlled DeepSets block: identical wrapper to AttnUpdate (input
    injection + residual + two LayerNorms + feed-forward), but the mixing is a
    mean-pool aggregator instead of multi-head attention. Isolates the effect of
    normalization (well-posedness) from attention (task performance).
    """

    def __init__(self, d_latent, d_in, hidden, spectral=False):
        super().__init__()
        del spectral
        self.x_proj = nn.Linear(d_in, d_latent)
        self.agg = nn.Sequential(
            nn.Linear(2 * d_latent, hidden), nn.ReLU(), nn.Linear(hidden, d_latent)
        )
        self.norm1 = nn.LayerNorm(d_latent)
        self.ff = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.ReLU(), nn.Linear(hidden, d_latent)
        )
        self.norm2 = nn.LayerNorm(d_latent)

    def forward(self, z, x, w=None):
        zx = z + self.x_proj(x)
        pool = _weighted_pool(zx, w).expand_as(zx)
        a = self.agg(torch.cat([zx, pool], dim=-1))
        z = self.norm1(z + a)
        z = self.norm2(z + self.ff(z))
        return z


class LinAttnUpdate(nn.Module):
    """Linear-attention block: identical wrapper to AttnUpdate (input injection +
    residual + two LayerNorms + feed-forward), but the softmax attention is
    replaced by linear attention with feature map phi(t)=elu(t)+1.

    The point of this block in the project: it is *more expressive* than mean-pool
    yet, unlike softmax attention, its aggregate is an ADDITIVE sufficient
    statistic over the set --

        S = sum_j phi(k_j) (x) v_j   (d x d),   Zsum = sum_j phi(k_j)   (d),
        out_i = phi(q_i)^T S / (phi(q_i)^T Zsum).

    So it is decomposable / federatable, and it admits the presence-weight knob w
    (weight each point's contribution to S and Zsum), which softmax attention does
    not. It is the pivot experiment for whether exact unlearning is governed by
    decomposability (then linattn stays unique) or by raw expressiveness (then it
    goes multistable like softmax attention).
    """

    def __init__(self, d_latent, d_in, hidden, spectral=False):
        super().__init__()
        del spectral
        self.x_proj = nn.Linear(d_in, d_latent)
        self.q = nn.Linear(d_latent, d_latent)
        self.k = nn.Linear(d_latent, d_latent)
        self.v = nn.Linear(d_latent, d_latent)
        self.o = nn.Linear(d_latent, d_latent)
        self.norm1 = nn.LayerNorm(d_latent)
        self.ff = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.ReLU(), nn.Linear(hidden, d_latent)
        )
        self.norm2 = nn.LayerNorm(d_latent)

    @staticmethod
    def _phi(t):
        return F.elu(t) + 1.0  # positive feature map (Katharopoulos et al. 2020)

    def forward(self, z, x, w=None):
        zx = z + self.x_proj(x)
        q = self._phi(self.q(zx))
        k = self._phi(self.k(zx))
        v = self.v(zx)
        if w is not None:
            k = k * w  # presence weight enters the additive sufficient statistic
        S = torch.einsum("bnd,bne->bde", k, v)          # sum_j phi(k_j) (x) v_j
        zsum = k.sum(dim=1)                              # sum_j phi(k_j)
        num = torch.einsum("bnd,bde->bne", q, S)
        den = torch.einsum("bnd,bd->bn", q, zsum).unsqueeze(-1) + 1e-6
        a = self.o(num / den)
        z = self.norm1(z + a)
        z = self.norm2(z + self.ff(z))
        return z


def _soft_neighbors(positions, radius, tau=0.5):
    """Soft neighbor weights: w_ij = sigmoid((r - ||p_i - p_j||) / tau).

    Smooth approximation to the hard ball 1[d < r]. Critical for state-dependent
    graphs (graph_source="latent"): the hard threshold causes binary flip-flop
    of neighbors across iterations, preventing convergence. The sigmoid makes the
    operator continuous in positions, restoring contractivity.
    """
    dists = torch.cdist(positions, positions)
    return torch.sigmoid((radius - dists) / tau)


class LocalMeanUpdate(nn.Module):
    """Local mean-pool update: aggregate only over radius-r neighbors.

    graph_source="input" builds the graph from input x (fixed across iterations).
    graph_source="latent" rebuilds from current z each iteration (state-dependent).
    """

    def __init__(self, d_latent, d_in, hidden, spectral=False,
                 radius=2.0, graph_source="input", tau=0.5, pos_dim=None):
        super().__init__()
        del spectral
        self.radius = radius
        self.graph_source = graph_source
        self.tau = tau
        self.pos_dim = pos_dim  # build the graph from x[..., :pos_dim] only (e.g.
        # geometric coordinates) when the input carries extra non-positional
        # channels like an injected seed signal; None = use all of x.
        self.x_proj = nn.Linear(d_in, d_latent)
        self.agg = nn.Sequential(
            nn.Linear(2 * d_latent, hidden), nn.ReLU(), nn.Linear(hidden, d_latent)
        )
        self.norm1 = nn.LayerNorm(d_latent)
        self.ff = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.ReLU(), nn.Linear(hidden, d_latent)
        )
        self.norm2 = nn.LayerNorm(d_latent)

    def forward(self, z, x, w=None):
        if self.graph_source == "latent":
            pos = z
        else:
            pos = x if self.pos_dim is None else x[..., :self.pos_dim]
        weights = _soft_neighbors(pos, self.radius, self.tau).unsqueeze(-1)
        degree = weights.sum(dim=2).clamp(min=1e-6)
        pool = (z.unsqueeze(1).expand(-1, z.shape[1], -1, -1) * weights).sum(dim=2)
        pool = pool / degree

        zx = z + self.x_proj(x)
        a = self.agg(torch.cat([zx, pool], dim=-1))
        z = self.norm1(z + a)
        z = self.norm2(z + self.ff(z))
        return z


class LocalAttnUpdate(nn.Module):
    """Local attention update: softmax attention with soft radius-gated weights.

    Attention logits are additively biased by the soft neighbor kernel so that
    distant pairs are exponentially downweighted without the hard-mask
    discontinuity that breaks state-dependent convergence.
    """

    def __init__(self, d_latent, d_in, hidden, n_heads=4, spectral=False,
                 radius=2.0, graph_source="input", tau=0.5, pos_dim=None):
        super().__init__()
        del spectral
        if d_latent % n_heads != 0:
            raise ValueError(f"d_latent ({d_latent}) must be divisible by "
                             f"n_heads ({n_heads})")
        self.radius = radius
        self.graph_source = graph_source
        self.tau = tau
        self.pos_dim = pos_dim
        self.n_heads = n_heads
        self.x_proj = nn.Linear(d_in, d_latent)
        self.qkv = nn.Linear(d_latent, 3 * d_latent)
        self.o_proj = nn.Linear(d_latent, d_latent)
        self.norm1 = nn.LayerNorm(d_latent)
        self.ff = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.ReLU(), nn.Linear(hidden, d_latent)
        )
        self.norm2 = nn.LayerNorm(d_latent)

    def forward(self, z, x, w=None):
        B, N, d = z.shape
        h = self.n_heads
        dk = d // h

        if self.graph_source == "latent":
            pos = z
        else:
            pos = x if self.pos_dim is None else x[..., :self.pos_dim]
        locality_bias = _soft_neighbors(pos, self.radius, self.tau)
        locality_logit = torch.log(locality_bias.clamp(min=1e-8))

        zx = z + self.x_proj(x)
        qkv = self.qkv(zx).reshape(B, N, 3, h, dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / (dk ** 0.5)
        attn = attn + locality_logit.unsqueeze(1)
        attn = torch.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, d)
        a = self.o_proj(out)
        z = self.norm1(z + a)
        z = self.norm2(z + self.ff(z))
        return z


_UPDATES = {
    "deepsets": DeepSetsUpdate,
    "attn": AttnUpdate,
    "normdeepsets": NormDeepSetsUpdate,
    "linattn": LinAttnUpdate,
    "local_mean": LocalMeanUpdate,
    "local_attn": LocalAttnUpdate,
}


class SetDEQ(nn.Module):
    def __init__(self, d_in, d_latent=64, hidden=128, update="deepsets",
                 n_classes=5, max_iter=30, tol=1e-4, damping=0.5, spectral=False,
                 solver="fixed_point_iter", pi_train=False, pi_min_iter=10,
                 radius=2.0, graph_source="input", tau=0.5, node_readout=False,
                 pos_dim=None):
        """solver: a TorchDEQ f_solver name ('fixed_point_iter', 'broyden',
        'anderson', ...) or 'damped' for the legacy hand-rolled iteration.

        Note (empirical, 2026-06-19): on these normalized set maps, Anderson
        acceleration STAGNATES at ~5e-3 residual, while 'fixed_point_iter' reaches
        ~1e-7. Default is therefore 'fixed_point_iter'. TorchDEQ still provides the
        implicit/phantom backward pass regardless of forward solver.
        """
        super().__init__()
        if update not in _UPDATES:
            raise ValueError(f"unknown update {update!r}; choose from {list(_UPDATES)}")
        self.d_latent = d_latent
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping
        self.solver = solver
        self.pi_train = pi_train
        self.pi_min_iter = pi_min_iter
        self.node_readout = node_readout
        update_kwargs = dict(spectral=spectral)
        if update.startswith("local_"):
            update_kwargs.update(radius=radius, graph_source=graph_source, tau=tau,
                                 pos_dim=pos_dim)
        self.update = _UPDATES[update](d_latent, d_in, hidden, **update_kwargs)
        self.readout = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
        )
        if solver != "damped":
            self.deq = get_deq(f_solver=solver, f_max_iter=max_iter, f_tol=tol)

    def solve(self, x, z0=None, max_iter=None, tol=None, damping=None):
        """Run the fixed-point solver for input set(s) x: (B, N, d_in).

        Returns (z_star, info) with info keys n_iter, converged, diverged.
        """
        if z0 is None:
            z0 = torch.randn(x.shape[0], x.shape[1], self.d_latent, device=x.device)
        f = lambda z: self.update(z, x)

        if self.solver != "damped":
            # Only f_max_iter overrides reliably; the construction-time f_tol governs
            # the solver's stopping, so the convergence flag is judged against self.tol.
            mi = self.max_iter if max_iter is None else max_iter
            out, info = self.deq(f, z0, solver_kwargs={"f_max_iter": mi})
            z_star = out[-1]
            rel = float(info["rel_lowest"].max())
            nstep = int(info["nstep"].max())
            diverged = not math.isfinite(rel)
            return z_star, {"n_iter": nstep, "converged": (not diverged) and rel < self.tol,
                            "diverged": diverged, "rel": rel}

        return fixed_point_solve(
            f, z0,
            max_iter=self.max_iter if max_iter is None else max_iter,
            tol=self.tol if tol is None else tol,
            damping=self.damping if damping is None else damping,
        )

    def pool(self, z):
        return z.mean(dim=1)

    def head(self, z):
        """Apply the readout. For node_readout=True the readout is applied
        per node, returning (B, N, out) for per-node regression/labeling tasks
        (e.g. propagation); otherwise it pools to a single (B, out) prediction.
        The nn.Sequential acts on the last dim either way."""
        return self.readout(z) if self.node_readout else self.readout(self.pool(z))

    def forward(self, x, z0=None):
        if self.training and self.pi_train and z0 is None:
            B, N = x.shape[0], x.shape[1]
            # mixed init: zeros on (at least) half the batch -- includes the
            # zero init used at test time, so there is no train/test init shift --
            # and Gaussian noise on the rest.
            z0 = torch.randn(B, N, self.d_latent, device=x.device)
            z0[: B // 2] = 0.0
            # randomized compute budget: a path-independent solution must reach the
            # same behaviour for any depth, so we vary it during training.
            mi = int(torch.randint(self.pi_min_iter, self.max_iter + 1, (1,)).item())
            z, info = self.solve(x, z0=z0, max_iter=mi)
            return self.head(z), info
        z, info = self.solve(x, z0)
        return self.head(z), info
