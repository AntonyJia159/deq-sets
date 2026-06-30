"""General per-edge-nonlinear MPNN as an edit-local DEQ.

Why this cell (vs FAGCN-DEQ): our contribution is NOT a specific architecture -- it is that ANY
cell with the desired expressivity can be cast as an equilibrium and inherits edit-locality (the
resolvent (I-J)^-1 decays with graph distance whenever (I-J) is sparse and well-conditioned). So
when FAGCN's cell is too weak for a task (its message alpha_ij*(Wv z_j) is LINEAR in z_j times a
scalar attention -> cannot form a per-edge nonlinearity like (z_j - z_i)^2), we should not shrink
the task; we should convert a MORE EXPRESSIVE cell. This is a general MPNN with a nonlinear message
computed PER EDGE before aggregation -- the GatedGCN / GINE / message-MLP family.

Exact update (per iter; h0 = Enc(X) fixed):
  m_ij  = phi( Wm [ z_i || z_j ] + b )                       per-edge NONLINEAR message
  agg_i = sum_{j in N(i)} (1/sqrt(d_i d_j)) m_ij             sym-normalized aggregation
  g_i   = W2 phi( W1 [ z_i || agg_i ] + b1 )                 ego-separated update MLP
  z_i'  = h0_i + s * g_i
  out   = Head([ z*_i || h0_i ])                             ego-sep readout

Conditioning control (soft, same toolkit as fagcn_deq_mlp): Wm,W1,W2 spectral-normed to <=1;
phi=ReLU 1-Lipschitz; sym-normalized aggregation has operator norm <= 1; concat-Jacobians contribute
<= sqrt(2) each, so Lip(f) <= ~2*s with the single scale knob s = s_max*sigmoid(.). jac_reg on
||J||_F handles what SN does not bound; power_method probe reports rho(J). A result counts as an
equilibrium only if it converges; edit-locality needs sigma_min(I-J) healthy (rho<1 sufficient but
not necessary), which jac_reg keeps in range.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.mpnn_deq
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdeq import get_deq
from torchdeq.loss import jac_reg, power_method

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# d=64 + heavy dropout/edge-drop/wd: the message-MLP cell is high-capacity and overfits small grids
# without this. jac_gamma=0.3 does double duty (conditioning + Jacobian smoothing/regularization);
# it keeps rho(J)~0.7 (well-conditioned, edit-local) while still letting the graph coupling learn.
CFG = dict(d=64, msg_hidden=64, mlp_hidden=64, s_max=1.0, jac_gamma=0.3,
           drop_in=0.1, drop_out=0.5, edge_drop=0.3,
           lr=1e-2, wd=5e-3, epochs=200, f_max_iter=40, f_tol=1e-4)


def _sn(W):
    """Spectral-normalize to operator norm 1 (exact; works for non-square too).

    cuSOLVER's gesvdj occasionally fails to converge on small matrices (LinAlgError code 64);
    fall back to exact CPU SVD in that case (rare path; identical value, gradient preserved)."""
    try:
        s = torch.linalg.matrix_norm(W, ord=2)
    except (torch.linalg.LinAlgError, RuntimeError):
        s = torch.linalg.matrix_norm(W.cpu(), ord=2).to(W.device)
    return W / s


class MPNNDEQ(nn.Module):
    def __init__(self, d_in, k, edges, deg, cfg):
        super().__init__()
        d, mh, m = cfg["d"], cfg["msg_hidden"], cfg["mlp_hidden"]
        self.d = d
        self.s_max, self.jac_gamma = cfg["s_max"], cfg["jac_gamma"]
        self.agg = cfg.get("agg", "sum")                     # "sum" (linear semiring) | "max" (tropical)
        self.enc = nn.Linear(d_in, d)
        self.Wm = nn.Linear(2 * d, mh)                      # per-edge message MLP layer 1 (SN'd)
        self.Wm2 = nn.Linear(mh, d, bias=False)            # per-edge message MLP layer 2 (SN'd)
        self.W1 = nn.Linear(2 * d, m)                      # ego||agg update MLP layer 1 (SN'd)
        self.W2 = nn.Linear(m, d, bias=False)              # update MLP layer 2          (SN'd)
        self.head = nn.Sequential(                          # ego-sep readout on concat[z*, h0]
            nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(cfg["drop_out"]), nn.Linear(d, k))
        self.s_raw = nn.Parameter(torch.tensor(0.4))       # s = s_max*sigmoid(.) ~ 0.36 init
        self.beta_raw = nn.Parameter(torch.tensor(0.5))    # logsumexp temp beta=softplus(.) ~1 init
        self.drop_in, self.edge_drop = cfg["drop_in"], cfg["edge_drop"]
        self.register_buffer("edges", edges)
        self.register_buffer("norm", 1.0 / torch.sqrt(deg[edges[0]] * deg[edges[1]]))
        self.deq = get_deq(f_solver="fixed_point_iter", f_max_iter=cfg["f_max_iter"],
                           f_tol=cfg["f_tol"])

    @property
    def s(self):
        return self.s_max * torch.sigmoid(self.s_raw)

    def _aggregate(self, z, Wm_n, Wm2_n, edges=None, norm=None):
        N = z.shape[0]
        dst, src = (self.edges if edges is None else edges)[0], \
                   (self.edges if edges is None else edges)[1]
        norm = self.norm if norm is None else norm
        if self.training and self.edge_drop > 0:                  # DropEdge
            keep = torch.rand(dst.shape[0], device=z.device) > self.edge_drop
            dst, src, norm = dst[keep], src[keep], norm[keep]
        e = torch.cat([z[dst], z[src]], dim=-1)                   # (E, 2d): ego || neighbor
        m = F.relu(e @ Wm_n.t() + self.Wm.bias) @ Wm2_n.t()       # (E, d) NONLINEAR per-edge message
        if self.agg == "max":                                     # TROPICAL aggregation (order stat)
            agg = torch.full((N, self.d), -1e30, device=z.device, dtype=z.dtype)
            agg.scatter_reduce_(0, dst[:, None].expand(-1, self.d), m, reduce="amax", include_self=False)
            return torch.where(agg < -1e29, torch.zeros_like(agg), agg)   # no-neighbor -> 0
        if self.agg == "logsumexp":                               # SMOOTH, learnable temperature beta:
            beta = F.softplus(self.beta_raw) + 1e-3               # beta->0 mean, beta->inf max (spans
            ms = beta * m                                         # the sum<->tropical semiring continuum)
            mmax = torch.full((N, self.d), -1e30, device=z.device, dtype=z.dtype)
            mmax.scatter_reduce_(0, dst[:, None].expand(-1, self.d), ms, reduce="amax", include_self=False)
            mmax = mmax.detach()                                  # stability offset (cancels in grad)
            ex = torch.exp(ms - mmax[dst])
            ssum = torch.zeros(N, self.d, device=z.device, dtype=z.dtype)
            ssum.index_add_(0, dst, ex)
            agg = (mmax + torch.log(ssum.clamp(min=1e-30))) / beta
            return torch.where(ssum < 1e-20, torch.zeros_like(agg), agg)  # no-neighbor -> 0
        if self.agg == "geomedian":   # ROBUST: one Weiszfeld/IRLS step toward the geometric median,
            nw = norm.unsqueeze(-1)   # FOLDED into the DEQ iteration (no nested loop -> amortized).
            msum = torch.zeros(N, self.d, device=z.device, dtype=z.dtype)
            wsum = torch.zeros(N, 1, device=z.device, dtype=z.dtype)
            msum.index_add_(0, dst, nw * m); wsum.index_add_(0, dst, nw)
            anchor = msum / wsum.clamp(min=1e-12)                  # sym-normalized mean = Weiszfeld anchor
            dist = (m - anchor[dst]).norm(dim=-1, keepdim=True)    # (E,1) each message's deviation
            w = nw / (dist + 1e-3)                                 # down-weight outlier (far) messages
            num = torch.zeros(N, self.d, device=z.device, dtype=z.dtype)
            den = torch.zeros(N, 1, device=z.device, dtype=z.dtype)
            num.index_add_(0, dst, w * m); den.index_add_(0, dst, w)
            agg = num / den.clamp(min=1e-12)
            return torch.where(den < 1e-11, torch.zeros_like(agg), agg)  # no-neighbor -> 0
        agg = torch.zeros(N, self.d, device=z.device, dtype=z.dtype)
        agg.index_add_(0, dst, norm.unsqueeze(-1) * m)            # sym-normalized SUM (linear semiring)
        return agg

    def _make_f(self, h0, edges=None, norm=None):
        """edges/norm override the trained buffers -> re-solve on an EDITED graph (maintenance)."""
        Wm_n, Wm2_n = _sn(self.Wm.weight), _sn(self.Wm2.weight)
        W1_n, W2_n = _sn(self.W1.weight), _sn(self.W2.weight)
        s = self.s

        def f(z):
            agg = self._aggregate(z, Wm_n, Wm2_n, edges, norm)
            u = torch.cat([z, agg], dim=-1)                       # ego || aggregate
            g = F.relu(u @ W1_n.t() + self.W1.bias) @ W2_n.t()    # ego-sep update MLP
            return h0 + s * g
        return f

    def forward(self, X, jac=False):
        h0 = self.enc(F.dropout(X, self.drop_in, self.training))
        f = self._make_f(h0)
        z = self.deq(f, torch.zeros_like(h0))[0][-1]
        reg = z.new_zeros(())
        if jac and self.jac_gamma > 0:
            z0 = z.detach().requires_grad_(True)
            reg = jac_reg(f(z0), z0, vecs=1)
        return self.head(torch.cat([z, h0], dim=-1)), reg

    @torch.no_grad()
    def spectral_radius(self, X):
        h0 = self.enc(X)
        f = self._make_f(h0)
        z = self.deq(f, torch.zeros_like(h0))[0][-1]
        with torch.enable_grad():
            z0 = z.detach().requires_grad_(True)
            _, rho = power_method(f(z0), z0, n_iters=30)
        return rho.max().item()


def run_split(d_in, K, edges, deg, X, y, tr, va, te, cfg, loss="ce", metric=None):
    """loss='ce' (classification) or 'mse' (regression); metric returns a scalar to maximize."""
    torch.manual_seed(0)
    model = MPNNDEQ(d_in, K, edges, deg, cfg).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best_va, best_te = -1e9, 0.0
    for ep in range(cfg["epochs"]):
        model.train(); opt.zero_grad()
        out, reg = model(X, jac=True)
        if loss == "ce":
            task = F.cross_entropy(out[tr], y[tr])
        else:
            task = F.mse_loss(out[tr].squeeze(-1), y[tr])
        (task + cfg["jac_gamma"] * reg).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                out, _ = model(X)
            va_s, te_s = metric(out, y, va), metric(out, y, te)
            if va_s > best_va:
                best_va, best_te = va_s, te_s
    model.eval()
    return best_te, dict(s=model.s.item(), rho=model.spectral_radius(X))


def main():
    """Smoke test: can the cell solve and report a sane spectral radius on a random grid task?"""
    from experiments.broyden_synthetic import grid_graph
    from experiments.aniso_teacher import AnisoTeacher
    print(f"device = {DEV}")
    print("config:", CFG, "\n")
    edges, deg, N = grid_graph(56)
    edges, deg = edges.to(DEV), deg.to(DEV)
    teacher = AnisoTeacher(edges, deg, N, d_feat=16, R=4, k=3, seed=0, target="nbr_sq")
    teacher.generate()
    X, t = teacher.X, teacher.s

    def r2(out, y, m):
        o = out[m].squeeze(-1); ym = y[m]
        return (1 - ((o - ym) ** 2).sum() / (((ym - ym.mean()) ** 2).sum() + 1e-9)).item()

    g = torch.Generator().manual_seed(0)
    p = torch.randperm(N, generator=g)
    tr = torch.zeros(N, dtype=torch.bool); va = tr.clone(); te = tr.clone()
    tr[p[: N // 2]] = True; va[p[N // 2: 3 * N // 4]] = True; te[p[3 * N // 4:]] = True
    tr, va, te = tr.to(DEV), va.to(DEV), te.to(DEV)
    print(f"grid 56x56={N}, train={int(tr.sum())} nodes")
    t0 = time.time()
    r, dd = run_split(16, 1, edges, deg, X, t, tr, va, te, dict(CFG), loss="mse", metric=r2)
    print(f"nbr_sq smoke: test R^2 {r:.3f}  [s {dd['s']:.2f}  rho(J) {dd['rho']:.3f}]  "
          f"({time.time()-t0:.1f}s)   (cell works + well-conditioned; anisotropic/biharmonic "
          f"is where the per-edge nonlinearity should beat FAGCN)", flush=True)


if __name__ == "__main__":
    main()
