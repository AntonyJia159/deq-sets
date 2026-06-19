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


_UPDATES = {
    "deepsets": DeepSetsUpdate,
    "attn": AttnUpdate,
    "normdeepsets": NormDeepSetsUpdate,
}


class SetDEQ(nn.Module):
    def __init__(self, d_in, d_latent=64, hidden=128, update="deepsets",
                 n_classes=5, max_iter=30, tol=1e-4, damping=0.5, spectral=False,
                 solver="fixed_point_iter"):
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
        self.update = _UPDATES[update](d_latent, d_in, hidden, spectral=spectral)
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

    def forward(self, x, z0=None):
        z, info = self.solve(x, z0)
        return self.readout(self.pool(z)), info
