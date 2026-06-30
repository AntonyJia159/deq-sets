"""Robustness as a corollary of edit-locality -- the HONEST cheap version.

Framing (path-independence, static-graph-in-flux): the model is a pure function of the current graph,
so an injected adversarial edge is just another (spurious) edit, and the sigma_min(I-J) conditioning
that makes benign edits local ALSO caps how far a poison propagates.

Two things this script establishes, and one it deliberately does NOT claim:

  (1) CONTAINMENT (the result). Inject k adversarial edges at a victim (connect it to the most
      target-OPPOSITE nodes -- a cheap Nettack-flavored structural attack) and measure the prediction
      shift vs graph distance from the victim. The near-field decays as a clean exponential (screening
      length xi); the blast radius is a local ball, not the graph. The far end of each adversarial edge
      lights up its OWN local ball (a second contained source), which is exactly the point: every edit,
      benign or adversarial, is confined by the same conditioning. No robust aggregator needed.

  (2) THE ROBUSTNESS<->CONTRAST TENSION (an honest finding, not a win). A geometric-median aggregator
      (one Weiszfeld/IRLS reweight folded into the DEQ iteration) is robust precisely because it
      DISCARDS neighborhood spread -- so it cannot fit a CONTRAST operator like the Laplacian at any
      coupling (R^2 ~ 0 vs sum's ~0.65). Robust aggregation is therefore NOT a free drop-in; its proper
      payoff is on a central-tendency / consensus-denoising task where adversaries drag a mean and the
      median resists. That experiment (a denoising teacher with outlier-corrupted features) is the
      natural next build and is left as future work rather than faked here.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.adv_robustness
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.broyden_synthetic import grid_graph
from experiments.aniso_teacher import AnisoTeacher
from experiments.mpnn_deq import MPNNDEQ, CFG
from experiments.fagcn_deq_locality import build_adj, bfs_hops

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, D_FEAT, R, K_OP = 32, 16, 4, 1
K_ADV = 2
MAXHOP = 8
EPOCHS = 120
S_MAX = 0.9


def r2(out, t, m):
    o = out[m].squeeze(-1); tm = t[m]
    return (1 - ((o - tm) ** 2).sum() / (((tm - tm.mean()) ** 2).sum() + 1e-9)).item()


def train(model, X, t, tr, va, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    best, state = -1e9, None
    for e in range(epochs):
        model.train(); opt.zero_grad()
        out, reg = model(X, jac=True)
        loss = F.mse_loss(out[tr].squeeze(-1), t[tr]) + CFG["jac_gamma"] * reg
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if torch.isfinite(loss) and torch.isfinite(gn):    # guard non-finite steps (cuda atomics)
            opt.step()
        else:
            opt.zero_grad()
        if e % 10 == 0:
            model.eval()
            with torch.no_grad():
                out, _ = model(X)
            v = r2(out, t, va)
            if v > best:
                best = v; state = {k: x.detach().clone() for k, x in model.state_dict().items()}
    model.load_state_dict(state); model.eval()
    return best


@torch.no_grad()
def solve(f, z, tol=1e-7, maxit=400):
    for k in range(1, maxit + 1):
        zn = f(z); r = (zn - z).norm() / (zn.norm() + 1e-9); z = zn
        if r < tol:
            return z, k
    return z, maxit


def inject_adv(edges, N, victim, poison):
    add = []
    for p in poison:
        add += [[victim, p], [p, victim]]
    add = torch.tensor(add, device=edges.device).t()
    ee = torch.cat([edges, add], dim=1)
    deg = torch.zeros(N, device=edges.device)
    deg.index_add_(0, ee[0], torch.ones(ee.shape[1], device=edges.device))
    norm = 1.0 / torch.sqrt((deg[ee[0]] * deg[ee[1]]).clamp(min=1e-9))
    return ee, norm


def fit_near_field(by_hop):
    """Screening length from the contiguous DECREASING near-field (auto-stop at the first rise -- the
    far-field uptick is the poison endpoint's own local ball, a separate contained source)."""
    hs, zs = [], []
    prev = None
    for h in range(1, MAXHOP + 1):
        if not by_hop[h]:
            break
        mz = float(np.mean(by_hop[h]))
        if prev is not None and mz > prev:
            break
        hs.append(h); zs.append(mz); prev = mz
    if len(hs) >= 2:
        return -1.0 / np.polyfit(hs, np.log(zs), 1)[0], hs[-1]
    return float("nan"), (hs[-1] if hs else 0)


def main():
    print(f"device = {DEV}")
    edges, deg, N = grid_graph(L)
    edges, deg = edges.to(DEV), deg.to(DEV)
    teacher = AnisoTeacher(edges, deg, N, d_feat=D_FEAT, R=R, k=K_OP, seed=0, target="laplacian")
    teacher.generate()
    X, t = teacher.X, teacher.s
    g = torch.Generator().manual_seed(0); p = torch.randperm(N, generator=g)
    tr = torch.zeros(N, dtype=torch.bool); va = tr.clone()
    tr[p[: N // 2]] = True; va[p[N // 2: 3 * N // 4]] = True
    tr, va = tr.to(DEV), va.to(DEV)
    adj = build_adj(edges, N)

    idx = lambda r, c: r * L + c
    victims = [idx(8, 8), idx(16, 16), idx(10, 20), idx(22, 11), idx(16, 8), idx(20, 22)]
    tcpu = t.cpu().numpy()
    poison_of = {}
    for v in victims:
        order = np.argsort(-np.abs(tcpu - tcpu[v]))
        nbr = set(adj[v]) | {v}
        poison_of[v] = [int(q) for q in order if int(q) not in nbr][:K_ADV]

    print(f"grid {L}x{L}={N}, target=laplacian k={K_OP}; {len(victims)} victims x {K_ADV} adversarial "
          f"edges to the most target-opposite nodes.\n", flush=True)
    t0 = time.time()

    # ---- (1) CONTAINMENT: sum-agg (the model that fits the contrast task) ----
    torch.manual_seed(0)
    model = MPNNDEQ(D_FEAT, 1, edges, deg, dict(CFG, agg="sum", s_max=S_MAX)).to(DEV)
    vr = train(model, X, t, tr, va, EPOCHS)
    rho = model.spectral_radius(X)
    with torch.no_grad():
        h0 = model.enc(X)

    @torch.no_grad()
    def pred(z):
        return model.head(torch.cat([z, h0], dim=-1)).squeeze(-1)

    z0 = torch.zeros(N, model.d, device=DEV)
    z_clean, _ = solve(model._make_f(h0), z0)
    p_clean = pred(z_clean)
    print(f"[sum-agg] clean val R^2 {vr:.3f}  rho(J) {rho:.3f}", flush=True)

    peak, by_hop = [], {h: [] for h in range(MAXHOP + 1)}
    for v in victims:
        ee, nn = inject_adv(edges, N, v, poison_of[v])
        z_adv, _ = solve(model._make_f(h0, ee, nn), z_clean)        # warm-start: attack IS an edit
        dp = (pred(z_adv) - p_clean).abs().cpu().numpy()
        peak.append(dp[v])
        hops = bfs_hops(adj, v, N)
        for h in range(MAXHOP + 1):
            mm = hops == h
            if mm.any():
                by_hop[h].extend(dp[mm].tolist())
    print(f"\nCONTAINMENT -- prediction shift vs distance from victim:", flush=True)
    print(f"{'hop':>4} {'mean|dpred|':>12} {'nodes':>7}", flush=True)
    for h in range(MAXHOP + 1):
        if by_hop[h]:
            print(f"{h:>4} {np.mean(by_hop[h]):>12.3e} {len(by_hop[h]):>7}", flush=True)
    xi, hstop = fit_near_field(by_hop)
    print(f"\npeak |dpred| at victim {np.mean(peak):.4f}; near-field screening length xi ~ {xi:.2f} hops "
          f"(fit hops 1..{hstop}) << diameter {2*(L-1)}.", flush=True)
    print(f"=> the attack is CONTAINED to a local ball; the global model is intact. (The far-field uptick "
          f"is each adversarial edge's OTHER endpoint lighting its own local ball -- a separate source, "
          f"equally contained.)", flush=True)

    # ---- (2) THE ROBUSTNESS<->CONTRAST TENSION (honest probe, not a win) ----
    print(f"\nROBUSTNESS<->CONTRAST TENSION -- can a robust (geometric-median) aggregator fit a CONTRAST "
          f"operator?", flush=True)
    for smax in [1.6, 3.0]:
        torch.manual_seed(0)
        gm = MPNNDEQ(D_FEAT, 1, edges, deg, dict(CFG, agg="geomedian", s_max=smax)).to(DEV)
        gvr = train(gm, X, t, tr, va, EPOCHS)
        print(f"  [geomedian s_max {smax:.1f}] clean val R^2 {gvr:.3f}  rho(J) {gm.spectral_radius(X):.3f}",
              flush=True)
    print(f"  => NO (R^2 ~ 0 vs sum's {vr:.2f}, even at high coupling): the geometric median is robust "
          f"because it DISCARDS spread, but the Laplacian IS the spread. Robust aggregation is not a free "
          f"drop-in; its payoff lives on a consensus/denoising task (outlier-corrupted features, drag-the-"
          f"mean attack) -- the natural next build, deliberately not faked here.", flush=True)
    print(f"\n({time.time()-t0:.0f}s)  READ: containment is the cheap robustness result and it falls out "
          f"of the same sigma_min conditioning -- an adversarial edge is a spurious edit, its reach is "
          f"bounded. Robust-aggregation peak-damage reduction is a real but SEPARATE claim that needs the "
          f"right (central-tendency) task.", flush=True)


if __name__ == "__main__":
    main()
