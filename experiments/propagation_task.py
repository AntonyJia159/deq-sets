"""Stage 1a: edit-locality and contraction of a local set-DEQ on a propagation task.

The task (a vehicle to get a non-trivially-trained local operator): N points in 2D,
connected into a radius graph. A few SEED nodes carry random signal vectors. Each
node's target is the HARMONIC (diffusion) EXTENSION of those signals -- clamp the
seeds at their signal and let every other node settle to the mean of its neighbours.
This is a genuinely multi-hop quantity (a far node's value is relayed across many
edges), so the equilibrium does work a single forward pass cannot.

We then ask the two questions that gate both downstream narratives (federation,
memory unit):

  CONTRACTION  -- is the trained operator's fixed point locally contractive
                  (rho(J_f) < 1)? The decentralized-solve convergence GUARANTEE
                  rides entirely on this, so it is load-bearing, not cosmetic.

  EDIT-LOCALITY -- delete one node, re-solve, and measure how far the equilibrium
                  of each surviving node shifts as a function of its GRAPH DISTANCE
                  to the deletion. The hoped-for result: the shift DECAYS with
                  distance for a small radius (local) and is FLAT for a large radius
                  (global). The contrast is controlled -- same architecture, only
                  the radius changes.

Run:  python -m experiments.propagation_task

Pre-registered reading:
  - local (r small): rho<1, Delta-Z* decays with graph distance, warm-start cheap
  - global (r=100):  Delta-Z* roughly flat in graph distance (deletion is diffuse)
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.model import SetDEQ
from src.jacobian import spectral_radius

DEV = "cuda" if torch.cuda.is_available() else "cpu"

N_POINTS = 48
D_SIG = 8
K_SEEDS = 3
TASK_RADIUS = 2.0
BOX = 10.0

D_LATENT = 32
HIDDEN = 64
MAX_ITER = 200      # eval / edit-locality solve budget (full convergence)
TRAIN_ITER = 80     # cheaper budget during training (operator still learned)
TOL = 1e-5

RADII = [1.5, 3.0, 100.0]   # local, medium, global (controlled contrast)
EPOCHS = 25
N_TRAIN = 800
N_TEST = 120
SEED = 0


# ---------- geometry / graph helpers ----------

def geodesic_distances(adj):
    """All-pairs unweighted shortest-path (min-plus matrix closure).
    adj: (N,N) bool/float with 1 on edges. Returns (N,N) float graph distances,
    inf where disconnected. Diagonal 0."""
    N = adj.shape[0]
    D = torch.full((N, N), float("inf"))
    D[adj > 0] = 1.0
    D.fill_diagonal_(0.0)
    # repeated min-plus squaring: D = min(D, D + D). Path lengths up to N-1 are
    # resolved after ceil(log2(N)) doublings, so a fixed round count is exact.
    for _ in range(int(np.ceil(np.log2(max(N, 2)))) + 1):
        D = (D.unsqueeze(2) + D.unsqueeze(0)).min(dim=1).values
    return D


def make_sample(gen):
    """One propagation sample. Returns:
      X_aug (N, 2+D_SIG+1) -- [position(2), seed signal (zeros off seeds), is_seed(1)]
      target (N, D_SIG)    -- the harmonic/diffusion extension of the seed signals
      gdist (N,N) graph distances, seed_idx (K,).
    The seed signal MUST be injected into the input: it is random per sample, so the
    model can only produce the right output by PROPAGATING it across the local graph.
    The graph itself is built from positions only (pos_dim=2), not the signal."""
    X = torch.rand(N_POINTS, 2, generator=gen) * BOX
    d = torch.cdist(X, X)
    adj = (d < TASK_RADIUS).float()
    gdist = geodesic_distances(adj)

    seed_idx = torch.randperm(N_POINTS, generator=gen)[:K_SEEDS]
    seed_sig = torch.randn(K_SEEDS, D_SIG, generator=gen)

    # Target = harmonic / diffusion extension: clamp seeds at their signal, set every
    # other node to the mean of its (self + neighbours), iterate to steady state.
    # This is the canonical "propagate to equilibrium" field -- genuinely multi-hop
    # (a far node's value is relayed across many edges) and exactly the kind of fixed
    # point a local mean-pool equilibrium can represent.
    self_loop = adj.clone()
    self_loop.fill_diagonal_(1.0)
    A_norm = self_loop / self_loop.sum(dim=1, keepdim=True).clamp(min=1.0)
    f = torch.zeros(N_POINTS, D_SIG)
    f[seed_idx] = seed_sig
    for _ in range(300):
        f = A_norm @ f
        f[seed_idx] = seed_sig
    target = f                                   # (N, D_SIG)

    sig_channel = torch.zeros(N_POINTS, D_SIG)
    sig_channel[seed_idx] = seed_sig
    is_seed = torch.zeros(N_POINTS, 1)
    is_seed[seed_idx] = 1.0
    X_aug = torch.cat([X, sig_channel, is_seed], dim=-1)  # (N, 2+D_SIG+1)
    return X_aug, target, gdist, seed_idx


def make_dataset(n, seed):
    gen = torch.Generator().manual_seed(seed)
    Xs, Ts, Gs = [], [], []
    for _ in range(n):
        X, T, G, _ = make_sample(gen)
        Xs.append(X); Ts.append(T); Gs.append(G)
    return torch.stack(Xs), torch.stack(Ts), Gs


# ---------- training ----------

D_IN = 2 + D_SIG + 1  # position + injected seed signal + seed indicator


def train_model(radius, Xtr, Ttr, seed):
    torch.manual_seed(seed)
    model = SetDEQ(d_in=D_IN, d_latent=D_LATENT, hidden=HIDDEN, update="local_mean",
                   n_classes=D_SIG, max_iter=MAX_ITER, tol=TOL,
                   radius=radius, graph_source="input", node_readout=True,
                   pos_dim=2).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr, Ttr = Xtr.to(DEV), Ttr.to(DEV)
    bs = 64
    model.max_iter = TRAIN_ITER  # cheaper solves while learning; restored after
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for s in range(0, len(Xtr), bs):
            idx = perm[s:s + bs]
            pred, _ = model(Xtr[idx])
            loss = F.mse_loss(pred, Ttr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"   r={radius:<5} epoch {ep:2d}  mse {tot/len(Xtr):.4f}")
    model.max_iter = MAX_ITER  # restore full budget for eval / edit-locality
    return model


@torch.no_grad()
def test_mse(model, Xte, Tte):
    model.eval()
    Xte, Tte = Xte.to(DEV), Tte.to(DEV)
    pred, _ = model(Xte)
    return F.mse_loss(pred, Tte).item()


# ---------- contraction ----------

def certify_contraction(model, Xte, n_sets=20):
    """rho(J_f) at the reached fixed point. rho is a property of whatever point the
    solver lands on, so we compute it on every non-diverged solve (not only those
    meeting the strict tol), and separately report what fraction met tol."""
    model.eval()
    rhos, n_conv = [], 0
    for i in range(min(n_sets, len(Xte))):
        x = Xte[i:i + 1].to(DEV)
        z_star, info = model.solve(x, max_iter=MAX_ITER)
        if info.get("diverged") or not torch.isfinite(z_star).all():
            continue
        n_conv += int(info["converged"])
        rho, _ = spectral_radius(model, x, z_star, n_iter=40, seed=0)
        rhos.append(rho)
    if not rhos:
        return float("nan"), float("nan"), 0.0, 0.0
    rhos = np.array(rhos)
    return rhos.mean(), rhos.max(), (rhos < 1.0).mean(), n_conv / min(n_sets, len(Xte))


# ---------- edit-locality ----------

@torch.no_grad()
def edit_locality(model, Xte, Gte, n_sets=60, max_dist=8):
    """Delete one node per set, re-solve (warm), measure ||Delta Z*_i|| binned by
    graph distance(i, deleted).

    Two correctness guards on the decay measurement:
      - a deletion is only counted if BOTH the full solve and the warm re-solve
        converged, so non-fixed-point iterates never enter the decay bins;
      - SEED (pivotal -- removes an injected source) and NON-SEED (redundant)
        deletions are binned SEPARATELY, since they have categorically different
        influence; the non-seed table is the clean edit-locality result.

    Returns (decay_nonseed, counts_nonseed, decay_seed, counts_seed,
             mean_warm_iters, mean_cold_iters)."""
    model.eval()
    bins = {"nonseed": {d: [] for d in range(1, max_dist + 1)},
            "seed":    {d: [] for d in range(1, max_dist + 1)}}
    warm_iters, cold_iters = [], []

    for s in range(min(n_sets, len(Xte))):
        x = Xte[s:s + 1].to(DEV)
        gdist = Gte[s]
        z_star, info = model.solve(x, max_iter=MAX_ITER)
        if not info["converged"]:
            continue

        is_seed = x[0, :, -1] > 0.5           # last input channel flags seeds
        j = int(torch.randint(0, N_POINTS, (1,)).item())
        kind = "seed" if bool(is_seed[j]) else "nonseed"

        keep = [k for k in range(N_POINTS) if k != j]
        keep_t = torch.tensor(keep, device=DEV)
        x_minus = x[:, keep_t, :]
        z_warm = z_star[:, keep_t, :]

        z_new, info_w = model.solve(x_minus, z0=z_warm, max_iter=MAX_ITER)
        _, info_c = model.solve(x_minus, max_iter=MAX_ITER)
        if info_w["converged"]:
            warm_iters.append(info_w["n_iter"])
        if info_c["converged"]:
            cold_iters.append(info_c["n_iter"])
        if not info_w["converged"]:
            continue  # a non-converged warm iterate would pollute the decay bins

        # relative shift per surviving node vs its pre-deletion equilibrium
        shift = (z_new - z_warm).norm(dim=-1) / (z_warm.norm(dim=-1) + 1e-8)  # (1, N-1)
        shift = shift.squeeze(0).cpu()
        for local_i, orig_i in enumerate(keep):
            gd = gdist[orig_i, j].item()
            if np.isfinite(gd) and 1 <= gd <= max_dist:
                bins[kind][int(gd)].append(shift[local_i].item())

    def summarize(b):
        return ({d: (np.mean(v) if v else float("nan")) for d, v in b.items()},
                {d: len(v) for d, v in b.items()})

    decay_ns, counts_ns = summarize(bins["nonseed"])
    decay_s, counts_s = summarize(bins["seed"])
    return (decay_ns, counts_ns, decay_s, counts_s,
            np.mean(warm_iters or [np.nan]), np.mean(cold_iters or [np.nan]))


# ---------- driver ----------

def main():
    print("Building dataset...")
    Xtr, Ttr, _ = make_dataset(N_TRAIN, SEED)
    Xte, Tte, Gte = make_dataset(N_TEST, SEED + 999)

    rows = []
    for radius in RADII:
        print(f"\n=== radius {radius} ===")
        model = train_model(radius, Xtr, Ttr, SEED)
        mse = test_mse(model, Xte, Tte)
        rho_mean, rho_max, frac_contract, conv_frac = certify_contraction(model, Xte)
        decay_ns, counts_ns, decay_s, counts_s, warm, cold = edit_locality(
            model, Xte, Gte)
        rows.append(dict(radius=radius, mse=mse, rho_mean=rho_mean, rho_max=rho_max,
                         frac_contract=frac_contract, conv_frac=conv_frac,
                         warm=warm, cold=cold, decay_ns=decay_ns, counts_ns=counts_ns,
                         decay_s=decay_s, counts_s=counts_s))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print(f"{'radius':>7} {'test_mse':>9} {'rho_mean':>9} {'rho_max':>8} "
          f"{'contract%':>10} {'conv%':>6} {'warm_it':>8} {'cold_it':>8}")
    for r in rows:
        print(f"{r['radius']:>7} {r['mse']:>9.4f} {r['rho_mean']:>9.3f} "
              f"{r['rho_max']:>8.3f} {r['frac_contract']:>10.0%} "
              f"{r['conv_frac']:>6.0%} {r['warm']:>8.0f} {r['cold']:>8.0f}")

    dists = list(range(1, 9))
    header = "radius  " + "".join(f"  d={d:<5}" for d in dists)

    def decay_table(title, decay_key, counts_key):
        print(f"\n{title}: mean relative ||Delta Z*|| by graph distance to deletion")
        print(header)
        for r in rows:
            cells = "".join(f"  {r[decay_key].get(d, float('nan')):<5.3f}"
                            for d in dists)
            print(f"{r['radius']:>6}  {cells}")
        r0 = rows[0]
        print("  (n per bin, smallest radius: "
              + " ".join(f"d={d}:{r0[counts_key].get(d, 0)}" for d in dists) + ")")

    decay_table("EDIT-LOCALITY [NON-SEED / redundant deletions]", "decay_ns", "counts_ns")
    decay_table("EDIT-LOCALITY [SEED / pivotal deletions]", "decay_s", "counts_s")

    print("\nReading: if non-seed decay falls across d for local (r=1.5) and stays flat")
    print("for global (r=100), edit-locality is real and scales with radius.")


if __name__ == "__main__":
    main()
