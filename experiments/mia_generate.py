"""MIA Phase 1 -- generate & cache the attack dataset (batched, GPU).

Paired membership construction (perfect unlearning => AUC 0.5 by construction):
  base B = 23 survivor points; candidates c, d drawn from B's own GMM clusters.
  MEMBER (label 1):  full = B u {c}; solve Z*(full); warm-delete c -> Z_post; cand=c
  NON-MEMBER (lbl 0): full = B u {d}; solve Z*(full); warm-delete d -> Z_post; cand=c
The deletable point is always placed LAST (index 23), so "delete it" = rows[:23] and
the warm init = Z*(full)[:, :23, :]. Same B, same presented candidate c; the only
difference is whether c or d passed through -- i.e. the warm-start history signal.

Caches per (config, seed): Z_post (2T,23,32) fp16, cand (2T,2), label, conv (did the
solve producing Z_post converge), sens (||Z*(Bu{c})[surv]-Z*(B)||, finite-diff effect
of c), nn_dist (c to nearest survivor). Phase 2 (mia_attack.py) trains the attacker.

Run:  & "D:\\deq-venv\\Scripts\\python.exe" -m experiments.mia_generate
"""

import os
import time

import numpy as np
import torch

from src.data import sample_gmm_set, GMMSetDataset
from src.model import SetDEQ
from src.train import train

DEV = "cuda" if torch.cuda.is_available() else "cpu"
K_RANGE = (1, 4)
N = 24
SURV = N - 1          # 23 survivors; deletable point at index SURV
D = 2
SEP, STD = 4.0, 1.0
TOL = 1e-5
PROBE_ITERS = 200
SOLVE_BATCH = 512

# --- full run (smoke was seeds=[0,1,2], trials=1000) ---
SEEDS = [0, 1, 2, 3, 4]
N_TRIALS = 3000

CONFIGS = {
    "attn_baseline": dict(update="attn", pi_train=False),
    "attn_pi":       dict(update="attn", pi_train=True),
    "normdeepsets":  dict(update="normdeepsets", pi_train=False),
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "mia_cache")


def make_trials(n_trials, seed):
    """B (n,23,2), c (n,2), d (n,2): 25 points per GMM config, split 23/1/1."""
    g = torch.Generator().manual_seed(10_000 + seed)
    Bs, cs, ds = [], [], []
    for _ in range(n_trials):
        k = int(torch.randint(K_RANGE[0], K_RANGE[1] + 1, (1,), generator=g))
        X, _ = sample_gmm_set(k, N + 1, D, SEP, STD, g)  # 25 points
        Bs.append(X[:SURV]); cs.append(X[SURV]); ds.append(X[SURV + 1])
    return torch.stack(Bs), torch.stack(cs), torch.stack(ds)


def train_target(cfg, seed, train_ds):
    torch.manual_seed(seed)
    model = SetDEQ(d_in=D, d_latent=32, hidden=64, update=cfg["update"],
                   n_classes=train_ds.n_classes, max_iter=150, tol=TOL,
                   pi_train=cfg["pi_train"], pi_min_iter=10)
    # Train ON THE GPU with a larger batch -- the DEQ solve is the cost, so keep it
    # on-device (previously CPU = the bottleneck). batch 256 -> good GPU utilization.
    train(model, train_ds, epochs=15, batch_size=256, lr=1e-3, seed=seed,
          log_every=0, device=DEV)
    return model.eval()


@torch.no_grad()
def _solve(model, x, z0=None):
    return model.solve(x, z0=z0, max_iter=PROBE_ITERS)[0]


@torch.no_grad()
def _converged(model, z_star, x):
    fz = model.update(z_star, x)
    r = (fz - z_star).flatten(1).norm(dim=1) / (z_star.flatten(1).norm(dim=1) + 1e-8)
    return (r < TOL).cpu()


@torch.no_grad()
def generate(model, B, c, d):
    """Return dicts of per-sample arrays for member and non-member halves."""
    n = B.shape[0]
    Zp_m, Zp_n, conv_m, conv_n, sens, nn, gap = [], [], [], [], [], [], []
    for s in range(0, n, SOLVE_BATCH):
        Bb = B[s:s + SOLVE_BATCH].to(DEV)
        cb = c[s:s + SOLVE_BATCH].to(DEV)
        db = d[s:s + SOLVE_BATCH].to(DEV)
        full_c = torch.cat([Bb, cb.unsqueeze(1)], dim=1)   # (b,24,2), c last
        full_d = torch.cat([Bb, db.unsqueeze(1)], dim=1)
        Zc = _solve(model, full_c)                          # (b,24,32)
        Zd = _solve(model, full_d)
        Zb = _solve(model, Bb)                              # cold, for sensitivity
        Zpm = _solve(model, Bb, z0=Zc[:, :SURV, :].contiguous())  # warm, member
        Zpn = _solve(model, Bb, z0=Zd[:, :SURV, :].contiguous())  # warm, non-member
        Zp_m.append(Zpm.half().cpu()); Zp_n.append(Zpn.half().cpu())
        conv_m.append(_converged(model, Zpm, Bb)); conv_n.append(_converged(model, Zpn, Bb))
        sens.append((Zc[:, :SURV, :] - Zb).flatten(1).norm(dim=1).cpu())
        nn.append(torch.cdist(cb.unsqueeze(1), Bb).squeeze(1).min(dim=1).values.cpu())
        # direct per-trial membership signal: warm-from-c vs warm-from-d equilibria.
        gap.append(((Zpm - Zpn).flatten(1).norm(dim=1) /
                    (Zpn.flatten(1).norm(dim=1) + 1e-8)).cpu())
    return {
        "Zp_m": torch.cat(Zp_m), "Zp_n": torch.cat(Zp_n),
        "conv_m": torch.cat(conv_m), "conv_n": torch.cat(conv_n),
        "sens": torch.cat(sens), "nn": torch.cat(nn), "gap": torch.cat(gap),
    }


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"device={DEV}  configs={list(CONFIGS)}  seeds={SEEDS}  trials={N_TRIALS}")
    # build the (identical) training set ONCE -- it was being rebuilt per model.
    train_ds = GMMSetDataset(n_samples=2000, k_range=K_RANGE, n_points=N, d=D,
                             sep=SEP, std=STD, seed=1)
    for name, cfg in CONFIGS.items():
        for seed in SEEDS:
            t = time.time()
            model = train_target(cfg, seed, train_ds)
            B, c, d = make_trials(N_TRIALS, seed)
            g = generate(model, B, c, d)
            # assemble paired samples: member (label 1) + non-member (label 0)
            cand = torch.cat([c, c], dim=0).numpy().astype(np.float32)
            Zpost = torch.cat([g["Zp_m"], g["Zp_n"]], dim=0).numpy().astype(np.float16)
            label = np.concatenate([np.ones(N_TRIALS), np.zeros(N_TRIALS)]).astype(np.int64)
            conv = torch.cat([g["conv_m"], g["conv_n"]], dim=0).numpy()
            sens = torch.cat([g["sens"], g["sens"]], dim=0).numpy().astype(np.float32)
            nn = torch.cat([g["nn"], g["nn"]], dim=0).numpy().astype(np.float32)
            gap = torch.cat([g["gap"], g["gap"]], dim=0).numpy().astype(np.float32)
            out = os.path.join(CACHE_DIR, f"{name}_seed{seed}.npz")
            np.savez_compressed(out, Zpost=Zpost, cand=cand, label=label,
                                conv=conv, sens=sens, nn=nn, gap=gap)
            cm = float(g["conv_m"].float().mean())
            print(f"  {name:<14} seed{seed}  conv_member={cm:.2f}  "
                  f"{time.time()-t:.1f}s  -> {os.path.basename(out)}")
    print("done.")


if __name__ == "__main__":
    main()
