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

import torch
import torch.nn as nn

try:
    from torch.nn.utils.parametrizations import spectral_norm
except Exception:  # older torch
    from torch.nn.utils import spectral_norm

from .solver import fixed_point_solve


def _maybe_sn(linear, on):
    return spectral_norm(linear) if on else linear


class DeepSetsUpdate(nn.Module):
    """Permutation-equivariant DeepSets update with mean-pool context."""

    def __init__(self, d_latent, d_in, hidden, spectral=False):
        super().__init__()
        self.net = nn.Sequential(
            _maybe_sn(nn.Linear(2 * d_latent + d_in, hidden), spectral),
            nn.ReLU(),
            _maybe_sn(nn.Linear(hidden, d_latent), spectral),
        )

    def forward(self, z, x):
        pool = z.mean(dim=1, keepdim=True).expand_as(z)
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

    def forward(self, z, x):
        zx = z + self.x_proj(x)
        a, _ = self.attn(zx, zx, zx, need_weights=False)
        z = self.norm1(z + a)
        z = self.norm2(z + self.ff(z))
        return z


_UPDATES = {"deepsets": DeepSetsUpdate, "attn": AttnUpdate}


class SetDEQ(nn.Module):
    def __init__(self, d_in, d_latent=64, hidden=128, update="deepsets",
                 n_classes=5, max_iter=30, tol=1e-4, damping=0.5, spectral=False):
        super().__init__()
        if update not in _UPDATES:
            raise ValueError(f"unknown update {update!r}; choose from {list(_UPDATES)}")
        self.d_latent = d_latent
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping
        self.update = _UPDATES[update](d_latent, d_in, hidden, spectral=spectral)
        self.readout = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
        )

    def solve(self, x, z0=None, max_iter=None, tol=None, damping=None):
        """Run the fixed-point solver for input set(s) x: (B, N, d_in)."""
        if z0 is None:
            z0 = torch.randn(x.shape[0], x.shape[1], self.d_latent, device=x.device)
        f = lambda z: self.update(z, x)
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
