"""C5 per-WINDOW residual early-stop -- TEST 1 v2 (retrospective; isolates the claim, no masked solver).

IDEA (ZJ's, principled): during a post-edit re-solve, freeze a window once its CERTIFIED per-window error
bound drops below tol -> the active set shrinks toward the edit, saving compute. Bound = reach envelope dotted
with the current residual field:  ||e_i|| = ||[M^-1 res]_i|| <= sum_j ||R[i,j]|| ||res_j||  (R=(I-J)^-1 at z*).
For a converged window res_j~0, so the bound is dominated by still-ACTIVE windows weighted by transport
distance -- it AUTOMATICALLY keeps window i active until its moving neighbours stop propagating in.

v1 was UNSAFE (rampant false-freezes). Fixes here:
  (1) CORRECT TARGET: these near-singular ckpts are near-multistable, so the WARM re-solve lands on a different
      branch than a COLD solve. Score against the branch the warm trajectory actually reaches: Newton-polish the
      trajectory END -> z*_warm, and build R at z*_warm (not the cold dense_resolvent equilibrium).
  (2) LINEAR-REGIME GATE: the bound uses M=I-J(z*); far from z* the true error follows the segment-averaged A
      (Kantorovich), so the bound only holds near convergence. Allow freezing only once global rel-resid < GATE.
  (3) PATIENCE: Anderson's residual is non-monotone -> require bound<tol for PATIENCE consecutive steps.
  (4) CAUSAL envelope (tier-2): R is block-LOWER-triangular (causal), so R[i,j>i]=0; fit the decay on the
      backward blocks and structure the envelope causally (0 for j>i).
TIER-1 (oracle) = true ||R[i,j]||; TIER-2 (envelope) = C_hat*r_hat^{i-j} (causal), a cheap proxy.

TEST 1 is RETROSPECTIVE: run the full warm solve, capture every iterate, ask per window the first (gated,
patient) iteration its bound<tol. SAFETY=#false-freezes->0 (froze while true err still>tol); COMPUTE=frac of
full window-iters (optimistic; trajectory not re-simulated with freezing -- that's Test 2).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.c5_window_earlystop
"""
import glob
import os

import numpy as np
import torch

import experiments.sliding_window_reach as sw
from experiments.c2_edit_locality import load_checkpoint, make_ff, apply_edit
from experiments.c2d_directional import dense_resolvent

CKPT_DIR = "checkpoints"
N_SEQS = 2
EDITS_PER_MODE = 4
TOL_REL = 1e-2           # window "converged" when error < 1% of its own state norm
SOLVE_TOL = 1e-4         # Anderson rel-resid stop (achievable; long trajectory)
GATE_RESID = 3e-2        # freezing allowed only once global rel-resid < GATE (linear regime, A~=M)
PATIENCE = 3             # bound must stay < tol for this many consecutive iters (robust to Anderson wiggle)
sw.H, sw.dh = 4, sw.d // 4


@torch.no_grad()
def anderson_traj(f, x0, m=5, max_iter=150, tol=SOLVE_TOL, lam=1e-4):
    """Anderson acceleration capturing (x_k, residual g_k=f(x_k)-x_k) at EVERY iterate. Handles rho>1."""
    shape = x0.shape
    D = x0.numel()
    dev, dt = x0.device, x0.dtype
    X = torch.zeros(max_iter, D, device=dev, dtype=dt)
    Fx = torch.zeros(max_iter, D, device=dev, dtype=dt)
    X[0] = x0.reshape(-1)
    Fx[0] = f(X[0].view(shape)).reshape(-1)
    X[1] = Fx[0]
    Fx[1] = f(X[1].view(shape)).reshape(-1)
    traj = [(X[0].view(shape).clone(), (Fx[0] - X[0]).view(shape).clone()),
            (X[1].view(shape).clone(), (Fx[1] - X[1]).view(shape).clone())]
    H = torch.zeros(m + 1, m + 1, device=dev, dtype=dt)
    H[0, 1:] = H[1:, 0] = 1.0
    y = torch.zeros(m + 1, device=dev, dtype=dt)
    y[0] = 1.0
    for k in range(2, max_iter):
        n = min(k, m)
        G = Fx[k - n:k] - X[k - n:k]
        H[1:n + 1, 1:n + 1] = G @ G.t() + lam * torch.eye(n, device=dev, dtype=dt)
        alpha = torch.linalg.solve(H[:n + 1, :n + 1], y[:n + 1])[1:n + 1]
        X[k] = alpha @ Fx[k - n:k]
        Fx[k] = f(X[k].view(shape)).reshape(-1)
        g = Fx[k] - X[k]
        traj.append((X[k].view(shape).clone(), g.view(shape).clone()))
        if g.norm() / (Fx[k].norm() + 1e-9) < tol:
            break
    return traj


def newton_resolvent(ff, z, iters=5):
    """Newton-polish z to the fixed point on ITS branch, then return (z*, R=(I-J)^-1) at that z* (fp64)."""
    z = z.clone().detach()
    ffl = lambda zv: ff(zv.view(z.shape)).reshape(-1)
    for _ in range(iters):
        r = (ff(z) - z).detach()
        if (r.norm() / (z.norm() + 1e-9)).item() < 1e-8:
            break
        zf = z.reshape(-1).detach()
        J = torch.func.jacrev(ffl)(zf).detach().double()
        ImJ = torch.eye(zf.numel(), device=J.device, dtype=torch.float64) - J
        z = (zf + torch.linalg.solve(ImJ, r.reshape(-1).double()).float()).view(z.shape).detach()
    zf = z.reshape(-1).detach()
    J = torch.func.jacrev(ffl)(zf).detach().double()
    R = torch.linalg.inv(torch.eye(zf.numel(), device=J.device, dtype=torch.float64) - J)
    return z.detach(), R


def block_norms(R, bounds, d):
    nb = len(bounds)
    B = torch.zeros(nb, nb, dtype=torch.float64, device=R.device)
    idx = [torch.arange(a * d, b * d, device=R.device) for a, b in bounds]
    for i in range(nb):
        for j in range(nb):
            B[i, j] = torch.linalg.matrix_norm(R[idx[i]][:, idx[j]], ord=2)
    return B


def analyze(traj, z_star, bounds, d, Wmat):
    """Per-window freeze iteration under the gated+patient rule, plus false-freeze count vs the TRUE error."""
    nb, K = len(bounds), len(traj)
    zt = torch.stack([t[0][0] for t in traj])
    rt = torch.stack([t[1][0] for t in traj])
    resnorm = torch.stack([rt[:, a:b, :].reshape(K, -1).norm(dim=1) for a, b in bounds], dim=1)
    znorm = torch.tensor([z_star[0, a:b, :].norm().item() for a, b in bounds], device=zt.device)
    tol_w = (TOL_REL * znorm).cpu().numpy()
    local_rr = (resnorm / (znorm[None, :] + 1e-9)).cpu().numpy()   # PER-WINDOW local rel-resid = the gate:
    #   window i may freeze only once IT is locally near convergence (its local linearization valid), so an
    #   already-settled far window freezes early instead of waiting for the ball (the global gate's flaw).
    true_err = torch.stack([(zt[:, a:b, :] - z_star[:, a:b, :]).reshape(K, -1).norm(dim=1)
                            for a, b in bounds], dim=1).cpu().numpy()
    bound = (resnorm.double() @ Wmat.t().double()).cpu().numpy()               # (K, nb)
    freeze_iter = np.full(nb, K - 1)
    false_freeze = 0
    frozen = 0
    for i in range(nb):
        cnt = 0
        for k in range(K):
            ok = (local_rr[k, i] < GATE_RESID) and (bound[k, i] < tol_w[i])
            cnt = cnt + 1 if ok else 0
            if cnt >= PATIENCE:
                freeze_iter[i] = k
                frozen += 1
                if true_err[k, i] >= tol_w[i]:
                    false_freeze += 1
                break
    compute = float(freeze_iter.sum()) / (nb * (K - 1) + 1e-9)
    return dict(K=K, nb=nb, frozen=frozen, false_freeze=false_freeze, compute=compute,
                final_err=float(true_err[-1].max()))


def main():
    ckpts = [c for c in sorted(glob.glob(os.path.join(CKPT_DIR, "curr*.pt")))
             if any(g in os.path.basename(c) for g in ["16", "24", "40"])]
    print(f"device={sw.DEV}  C5 per-WINDOW early-stop TEST 1 v2. Freeze window i when its bound "
          f"sum_j||R[i,j]|| ||res_j|| < {TOL_REL:g}*||z*_i||,\n  PER-WINDOW gated on LOCAL rel-resid<{GATE_RESID:g} "
          f"(window i's own linear regime) + {PATIENCE} consecutive. Target = WARM branch fixed point.\n"
          f"  SAFETY=false-freezes->0; COMPUTE=frac of full window-iters.\n", flush=True)
    for path in ckpts:
        m, ck = load_checkpoint(path)
        tag = os.path.basename(path).replace(".pt", "")
        gen = torch.Generator().manual_seed(11)
        rows = {"filler": [], "irrelevant": [], "relevant": []}
        for _ in range(N_SEQS):
            toks, _, _ = sw.gen_mqar(1, ck["stage_gap"], gen)
            L = toks.shape[1]
            bounds = [(s, min(s + sw.W, L)) for s in range(0, L, sw.W)]
            if len(bounds) < 3:
                continue
            nb = len(bounds)
            z_old, _, _, _ = dense_resolvent(m, toks)
            for mode in ["filler", "irrelevant", "relevant"]:
                for _ in range(EDITS_PER_MODE):
                    out = apply_edit(toks, gen, mode)
                    if out[0] is None:
                        continue
                    toks2, vpos = out
                    ff2, _ = make_ff(m, toks2)
                    traj = anderson_traj(ff2, z_old.clone())
                    if len(traj) < PATIENCE + 2:
                        continue
                    z_star, R = newton_resolvent(ff2, traj[-1][0])     # WARM-branch fixed point + resolvent
                    Rblk = block_norms(R, bounds, sw.d)
                    # tier-2 CAUSAL geometric envelope: backward ratios, structured 0 for j>i
                    back = [(Rblk[i, i - 1] / (Rblk[i, i] + 1e-30)).item() for i in range(1, nb)]
                    r_hat = min(0.99, float(np.median(back)) if back else 0.5)
                    C_hat = float(torch.diag(Rblk).max().item())
                    ii = torch.arange(nb)[:, None]
                    jj = torch.arange(nb)[None, :]
                    Wenv = torch.where(jj <= ii, C_hat * (r_hat ** (ii - jj).clamp(min=0).double()),
                                       torch.zeros(1, dtype=torch.float64)).to(Rblk.device)
                    a1 = analyze(traj, z_star, bounds, sw.d, Rblk)
                    a2 = analyze(traj, z_star, bounds, sw.d, Wenv)
                    rows[mode].append((a1, a2, r_hat))
        if not any(rows.values()):
            print(f"[{tag}] no usable edits\n"); continue
        print(f"[{tag}] gap={ck['stage_gap']} recall={ck['recall']:.3f} sigma_min={ck['sigma_min']:.3f}",
              flush=True)
        for mode in ["filler", "irrelevant", "relevant"]:
            v = rows[mode]
            if not v:
                continue
            for tier, idx in [("tier1-oracle", 0), ("tier2-envelope", 1)]:
                comp = np.mean([a[idx]["compute"] for a in v]) * 100
                ff = sum(a[idx]["false_freeze"] for a in v)
                tot = sum(a[idx]["frozen"] for a in v)
                kf = np.mean([a[idx]["K"] for a in v])
                fe = np.mean([a[idx]["final_err"] for a in v])
                print(f"    {mode:>10} {tier:>15}: compute={comp:4.0f}% of full  "
                      f"FALSE-FREEZES={ff}/{tot} window-freezes  (K_full~{kf:.0f}  final_err~{fe:.1e})",
                      flush=True)
        print(flush=True)
    print("READ: FALSE-FREEZES must be 0/N (froze while true err > tol = bound under-covered). compute% = "
          "potential\nactive-window-iters vs full (lower=more saving; optimistic). tier-1=oracle R (aggressive), "
          "tier-2=cheap\nenvelope (conservative). final_err~tol confirms the warm-branch target is right "
          "(v1 bug: scored vs cold branch).", flush=True)


if __name__ == "__main__":
    main()
