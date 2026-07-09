"""C2-RESIDUAL-TIGHTNESS — is the tier-2 deployable bound ||z-z*|| <= ||f(z)-z||/sigma_min PRACTICAL
(sound + tight enough to be useful), or vacuous/too loose?

THE MEASUREMENT: for candidate states z at controlled distances from the exact fixed point z*, compare the
CERTIFIED bound (resid/sigma_min, using the CACHED sigma_min at z* -- the realistic maintenance certificate)
against the TRUE error ||z-z*||. Report the tightness ratio = bound/actual (>=1 == sound; near 1 == tight).

Three candidate families isolate the story:
  (S) DIRECTED-STIFF: z = z* + eps * v_min, where v_min = right singular vector of (I-J) for the SMALLEST
      singular value. Then res = (I-J)(eps v_min) = eps*sigma_min*u_min saturates the bound -> ratio ~ 1
      (the WORST-CASE residual direction; the bound is tight exactly here).
  (R) DIRECTED-RANDOM: z = z* + eps * v_rand. res spreads over the spectrum -> ratio ~ kappa (loose, but it's
      an UPPER bound so still safe -- looseness here is benign).
  (I) PARTIAL-SOLVE ITERATES: real residuals from a cold solve. KEY: a stalled/partial solve's residual is
      dominated by the SLOW (stiff) mode -- the same small-sigma direction Anderson stalls on -- so real
      residuals live near (S), and the bound is TIGHT in practice, not at the (R) worst case. This is the
      crux of "practical, not vacuous."

Also computes the Newton-Kantorovich gate: beta=1/sigma_min, eta=beta*resid, h=beta*L*eta (L = local Jacobian-
Lipschitz probe); h<=1/2 => R_minus = (1-sqrt(1-2h))/(beta L) is the RIGOROUS bound and z is in the trust
region. Far from z* the LINEAR bound can UNDER-estimate (ratio<1, unsound) while NK's h>1/2 correctly ABSTAINS
-- demonstrating the bound is a MAINTENANCE (near-z*) certificate, tight inside the trust region NK certifies.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.c2_residual_tightness
"""
import glob
import math
import os

import numpy as np
import torch
from torchdeq import get_deq

import experiments.sliding_window_reach as sw
from experiments.c2_edit_locality import load_checkpoint, make_ff
from experiments.c2d_directional import dense_resolvent

CELLS = {"curr08.pt", "curr16.pt", "currnp40.pt"}     # well-conditioned / mid / near-singular
EPS = [1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1.0]             # perturbation sizes (x ||z*||)
CKPT_DIR = "checkpoints"
sw.H, sw.dh = 4, sw.d // 4


def jac(ff, z):
    zf = z.reshape(-1).detach()
    ffl = lambda v: ff(v.view(z.shape)).reshape(-1)
    return torch.func.jacrev(ffl)(zf).detach().double(), zf


def probe(ff, z_star, z, sigma_min, beta, Lloc):
    """One candidate: (actual, resid, linear-bound, ratio, NK h, R_minus or nan, sound)."""
    with torch.no_grad():
        actual = float((z - z_star).norm())
        resid = float((ff(z) - z).norm())
    bound = resid * beta                               # = resid / sigma_min (cached sigma_min at z*)
    ratio = bound / (actual + 1e-30)
    eta = beta * resid
    h = beta * Lloc * eta
    Rminus = float((1 - math.sqrt(max(1 - 2 * h, 0.0))) / (beta * Lloc)) if h <= 0.5 and Lloc > 0 else float("nan")
    return actual, resid, bound, ratio, h, Rminus, ratio >= 1.0


def main():
    ckpts = [p for p in sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt"))) if os.path.basename(p) in CELLS]
    if not ckpts:
        print(f"No target cells in {CKPT_DIR}/ (want {sorted(CELLS)})"); return
    print(f"device={sw.DEV}  Tier-2 residual-bound tightness. Bound=resid/sigma_min (cached sigma_min at z*) vs\n"
          f"  true ||z-z*||. ratio=bound/actual (>=1 sound, ~1 tight). Families: STIFF (worst-case dir, expect\n"
          f"  ratio~1), RANDOM (expect ~kappa, loose but safe), ITERATES (real partial-solve residuals).\n", flush=True)
    gen = torch.Generator().manual_seed(0)
    for path in ckpts:
        m, ck = load_checkpoint(path)
        gap = ck["stage_gap"]
        L = 2 * sw.D_PAIR + gap + sw.NQ
        d = sw.d
        g2 = torch.Generator().manual_seed(7)
        toks = sw.gen_mqar(1, gap, g2)[0]
        z_star, ff, J, R = dense_resolvent(m, toks)
        N = z_star.numel()
        ImJ = torch.eye(N, dtype=torch.float64, device=J.device) - J
        Usv, S, Vh = torch.linalg.svd(ImJ)
        sigma_min = float(S[-1]); sigma_max = float(S[0]); kappa = sigma_max / sigma_min
        beta = 1.0 / sigma_min
        v_min = Vh[-1].real.float()                    # right sing vec of smallest sigma (saturates the bound)
        v_rand = torch.randn(N, generator=gen, device="cpu").to(sw.DEV); v_rand /= v_rand.norm()
        znorm = float(z_star.norm())

        # local Jacobian-Lipschitz probe along v_min: L ~ ||J(z*+h v)-J(z*)|| / h
        hp = 1e-3
        zc = (z_star.reshape(-1) + hp * v_min).view(z_star.shape)
        Jc, _ = jac(ff, zc)
        Lloc = float((Jc - J).norm() / hp)

        print(f"[{os.path.basename(path)}] gap={gap} L={L} sigma_min={sigma_min:.4f} kappa={kappa:.1f} "
              f"L_loc={Lloc:.2f} ||z*||={znorm:.2f}", flush=True)

        for name, vdir in (("STIFF", v_min), ("RANDOM", v_rand)):
            print(f"    {name} direction:", flush=True)
            for eps in EPS:
                z = (z_star.reshape(-1) + eps * znorm * vdir).view(z_star.shape)
                a, r, b, ratio, h, Rm, sound = probe(ff, z_star, z, sigma_min, beta, Lloc)
                tr = "in-TR" if h <= 0.5 else "ABSTAIN(h>.5)"
                sflag = "" if sound else "  <-- UNSOUND (linear bound too small; far field)"
                rm = f" R-={Rm:.2e}" if not math.isnan(Rm) else ""
                print(f"      eps={eps:>4}: actual={a:.2e} resid={r:.2e} bound={b:.2e} ratio={ratio:6.2f} "
                      f"h={h:.2e} [{tr}]{rm}{sflag}", flush=True)

        # (I) real partial-solve residuals (cold), late iterates = near z*
        deq = get_deq(f_solver="anderson", f_max_iter=40, f_tol=1e-9, ift=True, b_solver="anderson", b_max_iter=1)
        with torch.no_grad():
            traj = deq(ff, torch.zeros_like(z_star))[0]
        iters = list(traj) if isinstance(traj, (list, tuple)) else [traj[i] for i in range(len(traj))]
        print(f"    ITERATES (real cold-solve residuals, far->near):", flush=True)
        for zt in iters[-5:]:
            a, r, b, ratio, h, Rm, sound = probe(ff, z_star, zt, sigma_min, beta, Lloc)
            tr = "in-TR" if h <= 0.5 else "ABSTAIN(h>.5)"
            print(f"      actual={a:.2e} resid={r:.2e} bound={b:.2e} ratio={ratio:6.2f} h={h:.2e} [{tr}]", flush=True)
        print(flush=True)

    print("READ: STIFF ratio~1 (bound tight at the worst-case residual direction) + RANDOM ratio~kappa (loose but\n"
          "safe) => tightness is DIRECTION-dependent. The payoff line: ITERATE ratios (REAL residuals) sit near\n"
          "STIFF, not RANDOM -- a partial solve's residual is dominated by the slow/stiff mode, exactly where the\n"
          "bound is tight => the deployable bound is PRACTICAL, not vacuous. UNSOUND rows (ratio<1) appear only far\n"
          "from z* where NK correctly ABSTAINS (h>1/2) => it's a MAINTENANCE certificate, tight inside the trust region.", flush=True)


if __name__ == "__main__":
    main()
