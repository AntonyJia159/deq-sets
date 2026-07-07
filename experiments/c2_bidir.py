"""C2-BIDIRECTIONAL — the Faber-face edit-locality measurement (the regime the abstract headlines).

Same claim as c2_edit_locality (edit one token; |dz| decays with distance; xi_measured <= xi_faber(kappa))
but on the BIDIRECTIONAL-band checkpoints, where the kappa->xi Chebyshev/Faber formula is the
theoretically PROPER certificate (J banded, not triangular; near-normality measured, not assumed).
On the causal checkpoints that formula was a category error — product-Lyapunov owned that face.

THREE STRUCTURAL CHANGES vs the causal C2:
  (1) TWO-SIDED RESPONSE. dz spreads BOTH ways from the edit; distance = |i - vpos|, and we keep the
      signed profile for the paper plot (left/right mass should be roughly balanced — vs causal, where
      the left was mathematically zero).
  (2) NOISE FLOOR REBUILT. The causal script used pre-edit positions (mathematically identical fixed
      points) as a free per-edit noise floor. Bidirectionally EVERY position may respond, so that
      instrument is dead. Substitute: solve the UNEDITED sequence twice from different inits; the
      per-position |za - zb| is pure solver noise IF the fixed point is unique (div ~ resid). If
      div >> resid the sequence is multistable -> excluded from envelope fits (flagged, not hidden).
  (3) NORMALITY CHECK. The Faber envelope is honest only if J is near-normal there; we MEASURE the
      departure nu = ||J J^T - J^T J||_F / ||J||_F^2 and print it next to the causal checkpoint's nu
      at the same gap — the "two faces" claim, as one number per face.

DISCRIMINATING PREDICTION (must-carry should shrink): a CAUSAL relay cannot see future queries, so it
transports ALL bindings (curr: irrelevant-edit xi 0.75->1.82, far/near up to 0.068). A BIDIRECTIONAL
relay can propagate query identity BACKWARD — it may learn to carry only what will be asked. If
irrelevant-edit far/near drops below the causal values, that's architecture-governed transport, exactly
the two-regime story.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.c2_bidir
"""
import glob
import os

import numpy as np
import torch
from torchdeq import get_deq

import experiments.sliding_window_reach as sw
# reuse the measurement-grade infra (this import also forces tf32 OFF — required here too)
from experiments.c2_edit_locality import make_ff, residual_of, apply_edit, faber_xi, fit_xi

sw.BIDIR = True
CKPT_DIR = "checkpoints"
N_EDITS = 16
N_SEQS = 4
TOL_COUNT = 1e-3        # solver-agnostic cost threshold (see counted_solve annotation)
sw.H, sw.dh = 4, sw.d // 4


def load_ckpt(path):
    ck = torch.load(path, map_location=sw.DEV, weights_only=False)
    m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    # f_tol 1e-4 not 1e-6: Anderson stalls ~1e-4 anyway (v4 finding) and, from WARM inits, its mixing
    # history is built from tiny residuals -> degenerate least-squares -> it burns the full cap without
    # ever satisfying 1e-6 (measured: warm-near-solution 151 evals vs cold 28). Newton polish closes
    # the gap to ~1e-7 either way, so nothing is lost and warm solves stop wasting the cap.
    m.deq = get_deq(f_solver="anderson", f_max_iter=80, f_tol=1e-4,
                    ift=True, b_solver="anderson", b_max_iter=40)
    return m, ck


def counted_solve(m, ff, z0):
    """Solve + Newton polish (same protocol as causal C2), but the COST metric is rebuilt:
    returns (z, k_tol) where k_tol = index of the first function evaluation whose true relative
    residual ||f(z_k)-z_k||/||z_k|| < TOL_COUNT. Anderson's own stop criterion is unusable as a cost
    proxy from warm inits (stall, above), so we read the residual at every eval (one extra norm, no
    extra evals) and count to a threshold Anderson reliably crosses. Warm/cold ratios of k_tol are
    the architecture-internal maintenance-cost claim."""
    rec = {"n": 0, "k": None}

    def ffc(z):
        out = ff(z)
        rec["n"] += 1
        if rec["k"] is None and (out - z).norm().item() < TOL_COUNT * (z.norm().item() + 1e-9):
            rec["k"] = rec["n"]
        return out
    with torch.no_grad():
        z = m.deq(ffc, z0)[0][-1]
    # Newton polish to measurement grade (fp64 linear solve; fp32 f-eval floors rel-resid ~1e-7)
    for _ in range(3):
        r = (ff(z) - z).detach()
        if (r.norm() / (z.norm() + 1e-9)).item() < 1e-7:
            break
        zf = z.reshape(-1).detach()
        ffl = lambda zv: ff(zv.view(z.shape)).reshape(-1)
        J = torch.func.jacrev(ffl)(zf)
        ImJ = torch.eye(zf.numel(), device=J.device, dtype=torch.float64) - J.double()
        step = torch.linalg.solve(ImJ, r.reshape(-1).double())
        z = (zf + step.float()).view(z.shape).detach()
    return z, (rec["k"] if rec["k"] is not None else rec["n"])


def kappa_nu_of(m, toks):
    """kappa_2(I-J), rho, sigma_min AND the normality departure nu = ||JJ^T-J^TJ||_F/||J||_F^2
    at the tight fixed point. nu ~ 0 justifies the Faber-on-[sigma interval] reading; nu O(1)
    (causal/triangular J is maximally non-normal) is where the formula was a category error."""
    ff, _ = make_ff(m, toks)
    z, _ = counted_solve(m, ff, torch.zeros(toks.shape[0], toks.shape[1], sw.d, device=sw.DEV))
    zf = z.reshape(-1).detach()
    ffl = lambda zv: ff(zv.view(z.shape)).reshape(-1)
    J = torch.func.jacrev(ffl)(zf)
    sv = torch.linalg.svdvals(torch.eye(zf.numel(), device=J.device) - J)
    rho = torch.linalg.eigvals(J).abs().max().item()
    nu = (torch.linalg.matrix_norm(J @ J.T - J.T @ J) / (torch.linalg.matrix_norm(J) ** 2 + 1e-12)).item()
    return (sv.max() / sv.min()).item(), rho, sv.min().item(), nu


def repeat_solve_noise(m, toks):
    """Noise instrument (replaces the causal pre-edit trick): solve the SAME unedited sequence from
    zeros and from small-noise init. Returns (per-position |za-zb| profile, relative div, resids).
    Unique fixed point -> the profile is the per-position solver noise floor; div >> resid -> multistable."""
    ff, _ = make_ff(m, toks)
    z0 = torch.zeros(toks.shape[0], toks.shape[1], sw.d, device=sw.DEV)
    torch.manual_seed(1234)
    za, _ = counted_solve(m, ff, z0)
    zb, _ = counted_solve(m, ff, 0.05 * torch.randn_like(z0))
    prof = (za - zb)[0].norm(dim=-1).cpu().numpy()
    div = (za - zb).norm().item() / (zb.norm().item() + 1e-9)
    return prof, div, residual_of(ff, za), residual_of(ff, zb)


def edit_response_profile(m, toks, gen, mode):
    """One edit -> (signed dists, dz over ALL positions, it_warm, it_cold, ok, warm_eq_cold), or None.
    Same warm/cold protocol as causal C2; the profile just isn't truncated to i >= vpos anymore."""
    ff_old, _ = make_ff(m, toks)
    z_old, _ = counted_solve(m, ff_old, torch.zeros(toks.shape[0], toks.shape[1], sw.d, device=sw.DEV))

    toks2, vpos = apply_edit(toks, gen, mode)
    if toks2 is None:
        return None
    ff_new, _ = make_ff(m, toks2)
    z_new, it_warm = counted_solve(m, ff_new, z_old.clone())
    z_cold, it_cold = counted_solve(m, ff_new, torch.zeros_like(z_old))

    r_old, r_new = residual_of(ff_old, z_old), residual_of(ff_new, z_new)
    warm_eq_cold = (z_new - z_cold).norm().item() / (z_cold.norm().item() + 1e-9)
    ok = (r_old < 1e-3) and (r_new < 1e-3)

    dz_all = (z_new - z_old)[0].norm(dim=-1).cpu().numpy()
    dists = np.arange(len(dz_all)) - vpos                     # SIGNED (left = upstream of the edit)
    return dists, dz_all, it_warm, it_cold, ok, warm_eq_cold


def main():
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "bidir*.pt")))
    if not ckpts:
        print(f"No bidir checkpoints in {CKPT_DIR}/ — run experiments.curriculum_bidir first."); return
    print(f"device={sw.DEV}  C2-BIDIR: Faber-face edit-locality; {N_EDITS} edits/mode, {N_SEQS} base seqs\n"
          f"  filler     : fair envelope witness -> xi_meas <= xi_faber(kappa) is THE claim here\n"
          f"  irrelevant : causal face showed MUST-CARRY; bidirectional relay can be query-aware ->\n"
          f"               prediction: far/near drops vs causal (0.068 at curr40)\n"
          f"  relevant   : transport to the dependent query = recall working (no xi claim)\n", flush=True)
    profiles = {}
    for path in ckpts:
        m, ck = load_ckpt(path)
        gen = torch.Generator().manual_seed(7)
        seqs = [sw.gen_mqar(1, ck["stage_gap"], gen)[0] for _ in range(N_SEQS)]
        kappa, rho, smin, nu = kappa_nu_of(m, seqs[0])

        # causal-face contrast at the same gap, if the causal checkpoint exists (one number per face)
        nu_causal = None
        cpath = os.path.join(CKPT_DIR, f"curr{ck['stage_gap']:02d}.pt")
        if os.path.exists(cpath):
            sw.BIDIR = False
            mc, _ = load_ckpt(cpath)
            _, _, _, nu_causal = kappa_nu_of(mc, seqs[0])
            del mc
            sw.BIDIR = True

        # per-seq noise floors + multistability screen (change (2))
        noise_profs, uniq = [], []
        divs = []
        for sq in seqs:
            prof, div, ra, rb = repeat_solve_noise(m, sq)
            multistable = div > 100 * max(ra, rb, 1e-9)
            noise_profs.append(prof); uniq.append(not multistable); divs.append(div)
        print(f"[{os.path.basename(path)}] recall={ck['recall']:.3f} rho={rho:.3f} smin={smin:.3f} "
              f"kappa={kappa:.1f} xi_faber={faber_xi(kappa):.2f} hops | nu_bidir={nu:.3f}"
              + (f" vs nu_causal={nu_causal:.3f}" if nu_causal is not None else "")
              + " | repeat-solve divs: " + "  ".join(f"{dv:.1e}{'' if u else '(MULTI)'}"
                                                     for dv, u in zip(divs, uniq)), flush=True)

        modes = ["irrelevant", "filler", "relevant"] if ck["stage_gap"] > 0 else ["irrelevant", "relevant"]
        for mode in modes:
            D, Z, IW, IC, wc, noises = [], [], [], [], [], []
            for sq, nprof, u in zip(seqs, noise_profs, uniq):
                if not u:
                    continue                                    # multistable seq: no envelope claims on it
                for _ in range(max(1, N_EDITS // N_SEQS)):
                    out = edit_response_profile(m, sq, gen, mode)
                    if out is None:
                        continue
                    d_, z_, iw, ic, ok, weqc = out
                    if ok:
                        D.append(d_); Z.append(z_); IW.append(iw); IC.append(ic)
                        wc.append(weqc); noises.append(float(nprof.max()))
            if not D:
                print(f"    {mode:>10}: no usable edits (no slot / loose fp / all seqs multistable)", flush=True)
                continue
            Dc, Zc = np.concatenate(D), np.concatenate(Z)
            noise_max = float(np.max(noises))
            peak = float(Zc.max())
            absd = np.abs(Dc)
            near = Zc[absd <= sw.W].max() if (absd <= sw.W).any() else np.nan
            far = Zc[absd > 2 * sw.W].mean() if (absd > 2 * sw.W).any() else 0.0
            farfrac = far / (near + 1e-12)
            lmass = Zc[Dc < 0].sum(); rmass = Zc[Dc > 0].sum()
            lfrac = lmass / (lmass + rmass + 1e-12)             # causal face: 0 by construction
            if peak < 10 * noise_max:
                print(f"    {mode:>10}: peak|dz|={peak:.1e} < 10x noise({noise_max:.1e})  "
                      f"wc==cc={np.mean(wc):.1e}  -> CONTAINED (below noise)", flush=True)
                continue
            xi_meas, npts = fit_xi(absd, Zc, noise=noise_max)   # hop-binned on |distance|
            if mode == "filler":
                if np.isnan(xi_meas):
                    verdict = "contained within ~1 window (too short for a hop-granularity fit)"
                else:
                    verdict = "ENVELOPE " + ("OK" if xi_meas <= faber_xi(kappa) else "VIOLATED")
            elif mode == "irrelevant":
                verdict = "query-aware? (compare far/near vs causal must-carry)"
            else:
                verdict = "transport (xi not meaningful)"
            profiles[f"{os.path.basename(path)}_{mode}"] = dict(dists=Dc, dz=Zc)
            print(f"    {mode:>10}: xi_meas={xi_meas:>6.2f} hops (n={npts})  far/near={farfrac:.3f}  "
                  f"left-mass={lfrac:.2f}  peak={peak:.1e}  warm/cold iters={np.mean(IW):.0f}/{np.mean(IC):.0f}  "
                  f"wc==cc={np.mean(wc):.1e}  -> {verdict}", flush=True)
        print(flush=True)

    np.savez(os.path.join(CKPT_DIR, "c2_bidir_profiles.npz"),
             **{k + "_" + f: v[f] for k, v in profiles.items() for f in ("dists", "dz")})
    print(f"(signed decay profiles saved to {CKPT_DIR}/c2_bidir_profiles.npz — the two-sided paper plot)", flush=True)
    print("\nREAD: filler rows = the headline Faber-face envelope (this is the regime where kappa->xi is"
          "\nproper). left-mass ~ 0.5 = genuinely two-sided response (vs 0 causal). irrelevant far/near vs"
          "\nthe causal 0.068 tests the must-carry prediction. nu_bidir << nu_causal justifies per-face"
          "\nmachinery. Multistable seqs are excluded from envelopes, not hidden.", flush=True)


if __name__ == "__main__":
    main()
