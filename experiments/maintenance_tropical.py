"""Does the TROPICAL (max-aggregation) equilibrium maintain like the linear one? (the missing check)

We claimed a max-plus equilibrium "stays maintainable (gamma<1 = sup-norm contraction)" but only
MEASURED expressivity (the 2x2 R^2). Max changes the Jacobian: it is a SELECTION (subgradient picks
the argmax edge), non-smooth, and a deletion can switch the argmax discretely -- the resolvent-decay
argument assumed a smooth J. So verify directly: train the max-agg cell on the tropical task, delete
nodes, warm-start re-solve, and measure the SAME triple as the spectral demo:
  RELIABILITY  warm-start vs cold re-solve (path-independence / exactness)
  LOCALITY     edit-response decay vs hop distance -> screening length xi
  COST         warm vs cold iterations; prediction ring radius
sum-agg on the same task is run as a reference (it should also maintain; the question is whether MAX
breaks it).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.maintenance_tropical
"""

import torch

from experiments.broyden_synthetic import grid_graph
from experiments.aniso_teacher import AnisoTeacher
from experiments.mpnn_deq import MPNNDEQ, CFG
from experiments.fagcn_deq_locality import build_adj
from experiments.maintenance_demo import train, DEV, L, D_FEAT, R
from experiments.maintenance_compare import probe

K_OP = 1


def main():
    print(f"device = {DEV}")
    edges, deg, N = grid_graph(L)
    edges, deg = edges.to(DEV), deg.to(DEV)
    teacher = AnisoTeacher(edges, deg, N, d_feat=D_FEAT, R=R, k=K_OP, seed=0, target="maxreach")
    teacher.generate()
    X, t = teacher.X, teacher.s
    g = torch.Generator().manual_seed(0); p = torch.randperm(N, generator=g)
    tr = torch.zeros(N, dtype=torch.bool); va = tr.clone()
    tr[p[: N // 2]] = True; va[p[N // 2: 3 * N // 4]] = True
    tr, va = tr.to(DEV), va.to(DEV)
    adj = build_adj(edges, N)
    idx = lambda r, c: r * L + c
    targets = [idx(L // 2, L // 2), idx(L // 2, L // 4), idx(L // 4, L // 4), idx(3, 3)]
    print(f"grid {L}x{L}={N}, TROPICAL task (max-reach); diameter {2*(L-1)}\n", flush=True)

    for agg in ("max", "sum"):
        torch.manual_seed(0)
        m = MPNNDEQ(D_FEAT, 1, edges, deg, dict(CFG, agg=agg)).to(DEV)
        vr = train(m, X, t, tr, va, CFG["epochs"])
        pr = probe(m, X, edges, N, adj, targets)
        print(f"[{agg}-agg]  val R^2 {vr:.3f}  rho(J) {m.spectral_radius(X):.3f}", flush=True)
        print(f"   RELIABILITY warm==cold : {pr['exact']:.1e}  (path-independent => exact maintenance)",
              flush=True)
        print(f"   LOCALITY    screening   : xi ~ {pr['xi']:.2f} hops", flush=True)
        print(f"   COST        warm/cold   : {pr['warm']:.0f}/{pr['cold']:.0f} iters, "
              f"pred ring radius {pr['ring']:.1f} hops << diameter {2*(L-1)}\n", flush=True)


if __name__ == "__main__":
    main()
