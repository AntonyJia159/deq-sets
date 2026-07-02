"""C2 — edit-locality: the sigma_min LAW, sequence version.  == ZJ's scaffold: core measurement is YOURS ==

THE CLAIM (the paper's centerpiece): perturb ONE token of a solved sequence; the equilibrium's response
|dz_i| decays exponentially with distance from the edit, with screening length xi bounded by the Faber/DMS
prediction computed from kappa(I-J). Measured across the curriculum checkpoints (sigma_min 0.15 -> 0.02,
rho 0.27 -> 1.19), this gives:
  (1) the ENVELOPE:      xi_measured <= xi_faber(kappa)     [soundness — the certificate never under-covers]
  (2) the TREND:         xi grows as sigma_min shrinks      [conditioning governs reach]
  (3) the DISSOCIATION:  at the rho=1.19 checkpoint, edits are STILL local (sigma_min>0 governs, not rho)
(3) is the money shot: a linear model cannot even exist at rho>1; we're local there.

WHAT'S WIRED FOR YOU: checkpoint loading, a tight fixed-point solver (the near-rho=1 solves need care),
edit application (substitute one value token), the kappa/Faber helpers, and the report table at the end.

WHAT'S YOURS (four steps, marked TODO(ZJ), each a few lines):
  [A] solve the edited sequence from a WARM START and verify it converged
  [B] measure the per-position response |dz_i| and organize it by distance from the edit
  [C] fit the screening length xi from the decay profile
  [D] compute the Faber prediction and assemble the comparison row

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.c2_edit_locality
"""
import glob
import os

import numpy as np
import torch
from torchdeq import get_deq

import experiments.sliding_window_reach as sw

CKPT_DIR = "checkpoints"
N_EDITS = 16            # average the decay profile over this many random edits (variance is real)
sw.H, sw.dh = 4, sw.d // 4


# ----------------------------------------------------------------------------- wired: infrastructure

def load_checkpoint(path):
    ck = torch.load(path, map_location=sw.DEV, weights_only=False)
    m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    # TIGHT solver for measurement: the training tol (1e-4) is too loose to fit a decay length against.
    m.deq = get_deq(f_solver="broyden", f_max_iter=250, f_tol=1e-7,
                    ift=True, b_solver="broyden", b_max_iter=40)
    return m, ck


def tight_solve(m, toks):
    """Solve to the tight fixed point; returns (z*, h0, mask, residual). Residual should be <~1e-5;
    if it isn't, the checkpoint's fixed point can't support a decay fit — report and skip."""
    h0 = m.h0(toks)
    mask = sw.band_causal_mask(toks.shape[1], toks.device)
    with torch.no_grad():
        z = m.solve(h0, mask)
        maskp = m._maskp(mask)
        resid = ((m.f(z, h0, m.wn(), maskp) - z).norm() / (z.norm() + 1e-9)).item()
    return z, h0, mask, resid


def apply_edit(toks, gen):
    """Substitute ONE value token (an odd position in the pair region) with a different random value id.
    Returns (edited_toks, edit_position). Value ids live in [NKEY, NKEY+NVAL)."""
    toks2 = toks.clone()
    vpos = 1 + 2 * torch.randint(sw.D_PAIR, (1,), generator=gen).item()     # odd slots hold values
    old = toks2[0, vpos].item()
    new = sw.NKEY + torch.randint(sw.NVAL, (1,), generator=gen).item()
    while new == old:
        new = sw.NKEY + torch.randint(sw.NVAL, (1,), generator=gen).item()
    toks2[0, vpos] = new
    return toks2, vpos


def kappa_of(m, toks):
    """kappa_2(I-J) at the (tight) fixed point of this example — reuses the wired spectrum()."""
    rho, smin, _ = m.spectrum(toks)
    # sigma_max(I-J) <= 1 + sigma_max(J); cheap exact route: redo the SVD here if you prefer.
    h0 = m.h0(toks); mask = sw.band_causal_mask(toks.shape[1], toks.device)
    maskp = m._maskp(mask); wn = m.wn()
    with torch.no_grad():
        z = m.solve(h0, mask)
    zf = z.reshape(-1).detach()
    ff = lambda zv: m.f(zv.view(z.shape), h0, wn, maskp).reshape(-1)
    J = torch.func.jacrev(ff)(zf)
    sv = torch.linalg.svdvals(torch.eye(zf.numel(), device=J.device) - J)
    return (sv.max() / sv.min()).item(), rho, smin


def faber_xi(kappa):
    """Faber/DMS screening length (in HOPS of the attention graph) from kappa(I-J):
    decay rate r = (sqrt(k)-1)/(sqrt(k)+1);  xi = -1/ln(r).  One hop = one window width W."""
    rk = (np.sqrt(kappa) - 1.0) / (np.sqrt(kappa) + 1.0)
    return np.inf if rk >= 1.0 else -1.0 / np.log(rk)


# ----------------------------------------------------------------------------- TODO(ZJ): the measurement

def edit_response_profile(m, toks, gen):
    """Return (dists, dz) — per-position response to one edit, as numpy arrays.

    [A] TODO(ZJ) — WARM-START RE-SOLVE:
        z_old, h0_old, mask, resid_old = tight_solve(m, toks)
        toks2, vpos = apply_edit(toks, gen)
        Solve the EDITED sequence starting FROM z_old (this is the warm start — the maintenance
        operation itself). Hints: build h0_new = m.h0(toks2); reuse m.deq with z_old as the init
        (m.deq(ff, z_old)); verify the new residual is as tight as the old one.

    [B] TODO(ZJ) — RESPONSE VS DISTANCE:
        dz_i = ||z_new[0, i] - z_old[0, i]||_2 for every position i  (a length-L vector).
        distance = i - vpos (causal model: only i >= vpos can respond; check that i < vpos gives ~0
        — that's a free sanity check of causality). Return the (distance, dz) pairs for i >= vpos.
    """
    raise NotImplementedError("ZJ: steps [A] and [B] go here")


def fit_xi(dists, dz, floor=1e-9):
    """[C] TODO(ZJ) — fit the screening length xi from the averaged decay profile.

    The model: dz(d) ~ dz(0) * exp(-d / xi).  So a linear fit of log(dz) against d gives slope -1/xi.
    Hints: average dz over edits at each distance first; drop points below `floor` (converged-to-noise
    tail poisons the fit — same lesson as the graph version); use np.polyfit(d, log(dz), 1).
    Return xi in POSITIONS; divide by sw.W to get hops (Faber's unit) when comparing.
    """
    raise NotImplementedError("ZJ: step [C] goes here")


# ----------------------------------------------------------------------------- wired: the report

def main():
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "curr*.pt")))
    if not ckpts:
        print(f"No checkpoints in {CKPT_DIR}/ — run experiments.curriculum_checkpoints first."); return
    print(f"device={sw.DEV}  C2 edit-locality across {len(ckpts)} checkpoints; {N_EDITS} edits each\n",
          flush=True)
    print(f"{'ckpt':>8} {'recall':>7} {'rho':>6} {'smin':>6} {'kappa':>7} | "
          f"{'xi_meas(hops)':>13} {'xi_faber':>9} {'sound?':>7}", flush=True)
    for path in ckpts:
        m, ck = load_checkpoint(path)
        gen = torch.Generator().manual_seed(7)
        toks = sw.gen_mqar(1, ck["stage_gap"], gen)[0]
        kappa, rho, smin = kappa_of(m, toks)
        # ---- [D] TODO(ZJ): loop N_EDITS calls of edit_response_profile, pool (dists, dz),
        #      xi_meas = fit_xi(...) / sw.W ;  xi_pred = faber_xi(kappa) ;  sound = xi_meas <= xi_pred
        xi_meas, xi_pred, sound = float("nan"), faber_xi(kappa), "?"
        print(f"{os.path.basename(path):>8} {ck['recall']:>7.3f} {rho:>6.3f} {smin:>6.3f} {kappa:>7.1f} | "
              f"{xi_meas:>13.2f} {xi_pred:>9.2f} {sound:>7}", flush=True)
    print("\nREAD: sound? must be YES on every row (envelope). xi_meas should GROW as smin shrinks (trend)."
          "\nThe rho>1 row (curr24/40) staying local = the dissociation, the paper's money shot.", flush=True)


if __name__ == "__main__":
    main()
