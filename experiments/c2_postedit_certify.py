"""C2-POSTEDIT-CERTIFY — the DEPLOYABLE tier-2 question: for REAL edits, is the residual certificate tight,
does the real residual live in the stiff mode (which is WHY it would be tight), and does the Woodbury prior
land you inside the Newton-Kantorovich trust region where the RIGOROUS bound fires?

Fixes the prior tightness probe (c2_residual_tightness), which used synthetic perturbations + cold iterates.
Here the candidate is the ACTUAL maintenance residual: after an edit, the warm re-solve starts from z*_old, so
its initial residual is r = f_new(z*_old) - z*_old. We measure, per edit class:

  (1) TIGHTNESS   ratio_lin = (||r||/sigma_min) / ||z*_old - z*_new||   (cached sigma_min at z*_old; z*_new =
                  the WARM-BRANCH fixed point = Anderson-warm+Newton from z*_old, the maintenance target, so
                  multistable cells are scored on the branch the re-solve actually reaches).
  (2) WHY         cos_stiff = ||U_k^T r|| / ||r||, U_k = top-k LEFT singular vecs of (I-J) (smallest sigma) =
                  the directions M^{-1} amplifies. High cos => r lives in the stiff mode => bound is tight
                  BECAUSE the real residual is stiff-aligned (the mechanism, measured not assumed).
  (3) RIGOR/TR    h = beta^2 L ||r||  (beta=1/sigma_min, L=local Jacobian-Lipschitz probe); frac with h<=1/2 =
                  fraction of real edits for which the NK certificate is rigorously in-trust-region. R_minus =
                  the rigorous bound there.
  (4) PRIOR->TR   same for the WOODBURY init z*_old + R@delta_h: does predicting the move pull more edits into
                  the trust region (widen where the rigorous cert fires)? This is the efficiency<->rigor synergy:
                  the prior's job is to land you where the cheap certificate is a theorem.

Cells: curr08/curr16 (well-cond), currnp16/currnp40 (near-singular, single-branch per the solver check). curr40
is EXCLUDED (multistable -> ambiguous target).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.c2_postedit_certify
"""
import glob
import math
import os

import numpy as np
import torch

import experiments.sliding_window_reach as sw
from experiments.c2_edit_locality import load_checkpoint, make_ff, apply_edit, counted_solve
from experiments.c2d_directional import dense_resolvent

CELLS = {"curr08.pt", "curr16.pt", "currnp16.pt", "currnp40.pt"}
EDITS_PER_CLASS = 6
KSTIFF = 8                                            # stiff-subspace dim (~ the carry rank, C2d-V4)
CKPT_DIR = "checkpoints"
sw.H, sw.dh = 4, sw.d // 4


def jac(ff, z):
    zf = z.reshape(-1).detach()
    ffl = lambda v: ff(v.view(z.shape)).reshape(-1)
    return torch.func.jacrev(ffl)(zf).detach().double(), zf


def resid_vec(ff, z):
    with torch.no_grad():
        return (ff(z) - z).reshape(-1).detach().double()


def nk(beta, Lloc, rnorm):
    """(h, R_minus or nan). h<=1/2 => in trust region, R_minus rigorous."""
    h = beta * Lloc * (beta * rnorm)
    Rm = float((1 - math.sqrt(max(1 - 2 * h, 0.0))) / (beta * Lloc)) if h <= 0.5 and Lloc > 0 else float("nan")
    return h, Rm


def main():
    ckpts = [p for p in sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt"))) if os.path.basename(p) in CELLS]
    if not ckpts:
        print(f"No target cells in {CKPT_DIR}/ (want {sorted(CELLS)})"); return
    print(f"device={sw.DEV}  Tier-2 POST-EDIT certificate. Candidate = the real warm residual r=f_new(z*_old)-z*_old.\n"
          f"  ratio_lin=(||r||/sigma_min)/||Δz*|| (tightness); cos_stiff=alignment of r with the top-{KSTIFF} stiff\n"
          f"  left-singular dirs (the WHY); frac in-TR (h<=1/2) for warm vs Woodbury init (rigor + prior synergy).\n"
          f"  {EDITS_PER_CLASS}/class.\n", flush=True)
    for path in ckpts:
        m, ck = load_checkpoint(path)
        gap = ck["stage_gap"]
        L = 2 * sw.D_PAIR + gap + sw.NQ
        d = sw.d
        g2 = torch.Generator().manual_seed(7)
        toks = sw.gen_mqar(1, gap, g2)[0]
        z_old, ff, J, R = dense_resolvent(m, toks)
        N = z_old.numel()
        ImJ = torch.eye(N, dtype=torch.float64, device=J.device) - J
        Usv, S, Vh = torch.linalg.svd(ImJ)
        sigma_min = float(S[-1]); kappa = float(S[0] / S[-1]); beta = 1.0 / sigma_min
        Uk = Usv[:, -KSTIFF:].real                    # top-k LEFT singular vecs (smallest sigma) = stiff dirs
        v_min = Vh[-1].real.float()
        hp = 1e-3
        Jc, _ = jac(ff, (z_old.reshape(-1) + hp * v_min).view(z_old.shape))
        Lloc = float((Jc - J).norm() / hp)

        print(f"[{os.path.basename(path)}] gap={gap} L={L} sigma_min={sigma_min:.4f} kappa={kappa:.0f} "
              f"L_loc={Lloc:.2f} beta={beta:.1f}", flush=True)
        classes = ["filler", "irrelevant", "relevant"] if gap > 0 else ["irrelevant", "relevant"]
        for cls in classes:
            rows = []
            for _ in range(EDITS_PER_CLASS):
                toks2, vpos = apply_edit(toks, g2, cls)
                if toks2 is None:
                    continue
                ff2, _ = make_ff(m, toks2)
                with torch.no_grad():
                    dh = (m.h0(toks2) - m.h0(toks)).reshape(-1).double()
                # warm-branch new equilibrium = the maintenance target
                z_new, _ = counted_solve(m, ff2, z_old.clone())
                actual = float((z_old - z_new).norm())
                if actual < 1e-6:
                    continue                          # edit didn't move the eq (no readers) — skip, undefined ratio
                r = resid_vec(ff2, z_old)
                rn = float(r.norm())
                ratio_lin = (rn / sigma_min) / actual
                dir_est = float((R @ r).norm())          # directional: ||R r|| ~ ||z*_old - z*_new|| (1st-order)
                ratio_dir = dir_est / actual             # ~1 if linearization + cached resolvent hold
                cos_stiff = float((Uk.T @ (r / (rn + 1e-30))).norm())
                h_w, Rm_w = nk(beta, Lloc, rn)
                # Woodbury init residual
                init_w = z_old + (R @ dh).view(z_old.shape).float()
                rw = resid_vec(ff2, init_w); rwn = float(rw.norm())
                actual_w = float((init_w - z_new).norm())
                h_o, _ = nk(beta, Lloc, rwn)
                rows.append((actual, rn, ratio_lin, cos_stiff, h_w, Rm_w, rwn, actual_w, h_o, ratio_dir))
            if not rows:
                print(f"    {cls:>10}: (no usable edits)", flush=True); continue
            A = np.array([r[:10] for r in rows], dtype=float)
            frac_tr_w = float(np.mean(A[:, 4] <= 0.5))
            frac_tr_o = float(np.mean(A[:, 8] <= 0.5))
            rnk = A[~np.isnan(A[:, 5])]
            ratio_nk = float(np.mean(rnk[:, 5] / rnk[:, 0])) if len(rnk) else float("nan")
            print(f"    {cls:>10}: n={len(rows)}  ||Δz*||={A[:,0].mean():.2e}  ratio_lin(scalar)={A[:,2].mean():6.2f}  "
                  f"ratio_dir(||R r||)={A[:,9].mean():5.2f}  cos_stiff={A[:,3].mean():.2f}  "
                  f"in-TR warm={frac_tr_w:.0%}->Wood={frac_tr_o:.0%}  ratio_NK={ratio_nk:.2f}", flush=True)
        print(flush=True)

    print("READ: cos_stiff near 1 => real edit residuals DO live in the stiff mode => ratio_lin near 1 (tight)\n"
          "BECAUSE of alignment, not by luck. in-TR warm = fraction of edits the RIGOROUS NK cert already covers;\n"
          "->Woodbury = after the low-rank prior (||r_wood|| << ||r|| and higher in-TR => the prior pulls edits into\n"
          "the trust region, the efficiency<->rigor synergy). Low cos_stiff / low in-TR = report straight (bound\n"
          "loose in practice / rigorous cert needs more solver work first).", flush=True)


if __name__ == "__main__":
    main()
