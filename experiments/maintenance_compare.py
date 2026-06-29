"""HEAD-TO-HEAD: ours (nonlinear equilibrium) vs the InstantGNN-style LINEAR editable incumbent,
on the SAME task and the SAME node deletions. The point of the paper in one table.

InstantGNN's propagation IS a linear fixed point: z = alpha*h0 + (1-alpha)*A_hat z  (= personalized
PageRank). So the incumbent is OUR model with the nonlinear per-edge message replaced by LINEAR
propagation -- identical encoder, ego-sep head, solver, and deletion machinery. One variable changes:
propagation nonlinearity. Expected:
  * BOTH are edit-local & maintainable by warm-start re-solve (both are well-conditioned fixed
    points; the linear one trivially, rho=1-alpha<1) -> maintainability is NOT unique to us.
  * ONLY ours is EXPRESSIVE on the (beyond-linear, local) Laplacian task.
=> "expressive AND editable": contraction/conditioning (not linearity) is the enabler; we enlarge
the editable class from linear to nonlinear/interleaved.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.maintenance_compare
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdeq import get_deq
from torchdeq.loss import power_method

from experiments.broyden_synthetic import grid_graph
from experiments.aniso_teacher import AnisoTeacher
from experiments.mpnn_deq import MPNNDEQ, CFG, _sn
from experiments.fagcn_deq_locality import delete_node, build_adj, bfs_hops

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, D_FEAT, R, K_OP, MAXHOP = 40, 16, 4, 2, 14


class LinearPPRDEQ(nn.Module):
    """InstantGNN-style LINEAR editable incumbent: z = alpha*h0 + (1-alpha)*A_hat z (PPR fixed
    point), then the SAME ego-sep node head as ours. Same interface as MPNNDEQ (enc/_make_f/head/
    forward/spectral_radius) so the identical probe runs on both. Decoupled: linear propagation +
    node-local head = the APPNP/InstantGNN expressivity class; propagation stays linear => exactly
    maintainable (push)."""
    def __init__(self, d_in, k, edges, deg, cfg):
        super().__init__()
        d = cfg["d"]
        self.d = d
        self.enc = nn.Linear(d_in, d)
        self.head = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(cfg["drop_out"]), nn.Linear(d, k))
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))     # alpha = sigmoid(.) in (0,1)
        self.drop_in = cfg["drop_in"]
        self.register_buffer("edges", edges)
        self.register_buffer("norm", 1.0 / torch.sqrt(deg[edges[0]] * deg[edges[1]]))
        self.deq = get_deq(f_solver="fixed_point_iter", f_max_iter=cfg["f_max_iter"], f_tol=cfg["f_tol"])

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw)

    def _make_f(self, h0, edges=None, norm=None):
        edges = self.edges if edges is None else edges
        norm = self.norm if norm is None else norm
        dst, src, a = edges[0], edges[1], self.alpha

        def f(z):
            agg = torch.zeros_like(z)
            agg.index_add_(0, dst, norm.unsqueeze(-1) * z[src])    # A_hat z (linear propagation)
            return a * h0 + (1 - a) * agg
        return f

    def forward(self, X, jac=False):
        h0 = self.enc(F.dropout(X, self.drop_in, self.training))
        z = self.deq(self._make_f(h0), torch.zeros_like(h0))[0][-1]
        return self.head(torch.cat([z, h0], dim=-1)), z.new_zeros(())

    @torch.no_grad()
    def spectral_radius(self, X):
        return (1 - self.alpha).item()                       # rho(J) = (1-alpha)*rho(A_hat) <= 1-alpha


def r2(out, t, m):
    o = out[m].squeeze(-1); tm = t[m]
    return (1 - ((o - tm) ** 2).sum() / (((tm - tm.mean()) ** 2).sum() + 1e-9)).item()


def train(model, X, t, tr, va, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    best_va, state = -1e9, None
    for e in range(epochs):
        model.train(); opt.zero_grad()
        out, reg = model(X, jac=True)
        (F.mse_loss(out[tr].squeeze(-1), t[tr]) + CFG["jac_gamma"] * reg).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if e % 10 == 0:
            model.eval()
            with torch.no_grad():
                out, _ = model(X)
            v = r2(out, t, va)
            if v > best_va:
                best_va = v
                state = {k: x.detach().clone() for k, x in model.state_dict().items()}
    model.load_state_dict(state); model.eval()
    return best_va


@torch.no_grad()
def solve(f, z, tol=1e-7, maxit=400):
    for k in range(1, maxit + 1):
        zn = f(z); r = (zn - z).norm() / (zn.norm() + 1e-9); z = zn
        if r < tol:
            return z, k
    return z, maxit


@torch.no_grad()
def probe(model, X, edges, N, adj, targets):
    """Run the identical maintenance probe: full solve, then warm/cold re-solve per deletion."""
    h0 = model.enc(X)

    def pred(z):
        return model.head(torch.cat([z, h0], dim=-1)).squeeze(-1)
    z0 = torch.zeros(N, model.d, device=DEV)
    z_star, it_full = solve(model._make_f(h0), z0)
    pstar = pred(z_star)
    ptol = 1e-3 * float(pstar.abs().mean() + 1e-9)
    by_hop, gaps, wits, cits, rings = {h: [] for h in range(MAXHOP + 1)}, [], [], [], []
    for v in targets:
        ee, ne = delete_node(v, edges, N)
        fe = model._make_f(h0, ee, ne)
        zw, iw = solve(fe, z_star); zc, ic = solve(fe, z0)
        gaps.append(((zw - zc).norm() / (zw.norm() + 1e-9)).item())
        dz = (zw - z_star).norm(dim=1).cpu().numpy()
        dp = (pred(zw) - pstar).abs().cpu().numpy()
        hops = bfs_hops(adj, v, N)
        ring = 0
        for h in range(MAXHOP + 1):
            m = hops == h
            if m.any():
                by_hop[h].extend(dz[m].tolist())
                if dp[m].mean() > ptol:
                    ring = h
        wits.append(iw); cits.append(ic); rings.append(ring)
    hs, zs = [], []
    for h in range(1, MAXHOP + 1):
        if by_hop[h]:
            mz = np.mean(by_hop[h])
            if mz > 1e-6:
                hs.append(h); zs.append(mz)
    slope = np.polyfit(hs, np.log(zs), 1)[0] if len(hs) >= 2 else 0.0
    xi = -1.0 / slope if slope < 0 else float("inf")
    return dict(it_full=it_full, exact=np.mean(gaps), warm=np.mean(wits), cold=np.mean(cits),
                ring=np.mean(rings), xi=xi)


def main():
    print(f"device = {DEV}")
    edges, deg, N = grid_graph(L)
    edges, deg = edges.to(DEV), deg.to(DEV)
    teacher = AnisoTeacher(edges, deg, N, d_feat=D_FEAT, R=R, k=K_OP, seed=0, target="interleaved")
    teacher.generate()
    X, t = teacher.X, teacher.s
    g = torch.Generator().manual_seed(0); p = torch.randperm(N, generator=g)
    tr = torch.zeros(N, dtype=torch.bool); va = tr.clone()
    tr[p[: N // 2]] = True; va[p[N // 2: 3 * N // 4]] = True
    tr, va = tr.to(DEV), va.to(DEV)
    adj = build_adj(edges, N)
    idx = lambda r, c: r * L + c
    targets = [idx(L // 2, L // 2), idx(L // 2, L // 4), idx(L // 4, L // 4), idx(3, 3)]
    print(f"grid {L}x{L}={N}, teacher INTERLEAVED (prop->relu->prop->square); diameter {2*(L-1)}\n",
          flush=True)

    rows = []
    for name, Cls in [("InstantGNN-linear (PPR)", LinearPPRDEQ), ("ours (MPNN-DEQ)", MPNNDEQ)]:
        torch.manual_seed(0)
        m = Cls(D_FEAT, 1, edges, deg, CFG).to(DEV)
        t0 = time.time()
        vr = train(m, X, t, tr, va, CFG["epochs"])
        rho = m.spectral_radius(X)
        pr = probe(m, X, edges, N, adj, targets)
        rows.append((name, vr, rho, pr))
        print(f"{name}: trained R^2 {vr:.3f}  rho(J) {rho:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'model':<26} {'R^2(expr)':>9} {'rho(J)':>7} {'xi(hop)':>8} {'exact':>9} "
          f"{'warm/cold':>10} {'ringR':>6}", flush=True)
    for name, vr, rho, pr in rows:
        print(f"{name:<26} {vr:>9.3f} {rho:>7.3f} {pr['xi']:>8.2f} {pr['exact']:>9.1e} "
              f"{pr['warm']:>4.0f}/{pr['cold']:<4.0f}  {pr['ring']:>5.1f}", flush=True)
    print("\nREAD: both edit-LOCAL (finite xi) & exactly maintainable (warm==cold) -- maintainability "
          "is NOT unique to us; only OURS is expressive (R^2) on this beyond-linear local task.",
          flush=True)


if __name__ == "__main__":
    main()
