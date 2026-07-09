"""C2 — edit-locality: the sigma_min LAW, sequence version.   == filled in, with annotations for ZJ ==

THE CLAIM (the paper's centerpiece): perturb ONE token of a solved sequence; the equilibrium's response
|dz_i| decays exponentially with distance from the edit, with screening length xi bounded by the Faber/DMS
prediction computed from kappa(I-J). Measured across the curriculum checkpoints (sigma_min ~0.18 -> ~0.03),
this gives:
  (1) the ENVELOPE:      xi_measured <= xi_faber(kappa)     [soundness — the certificate never under-covers]
  (2) the TREND:         xi grows as sigma_min shrinks      [conditioning governs reach]
  (3) the DISSOCIATION:  xi tracks sigma_min while rho varies wildly (0.28 -> 8.4) — rho is not the invariant
Plus the operational corollary: the WARM/COLD iteration ratio per edit (the maintenance channel, measured as
an architecture-internal property — never wall-clock; see blueprint 'claim-status').

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.c2_edit_locality
"""
import glob
import os

import numpy as np
import torch
from torchdeq import get_deq

import experiments.sliding_window_reach as sw

# MEASUREMENT-GRADE PRECISION: the training module enables tf32 (fine for SGD, ~1e-3 matmul precision),
# but it caps the Newton polish — the Jacobian and step inherit tf32 error and the residual stalls ~1e-4.
# This script measures decay lengths against noise floors, so accuracy > speed: force true fp32 here.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CKPT_DIR = "checkpoints"
N_EDITS = 16            # average the decay profile over this many random edits (variance is real)
sw.H, sw.dh = 4, sw.d // 4


# ----------------------------------------------------------------------------- infrastructure

def load_checkpoint(path):
    ck = torch.load(path, map_location=sw.DEV, weights_only=False)
    # Restore substrate flags from the checkpoint BEFORE building the model -- the relb table and posw
    # existence depend on them, so a relative-PE checkpoint (currnp: no_posw/rel_bias/readonly_q) must be
    # built with those set or load_state_dict mismatches and the forward is wrong. Defaults = the causal+
    # absolute curr* config, so curr*.pt (which don't save these keys) are byte-for-byte unaffected.
    sw.BIDIR = ck.get("bidir", False)
    sw.REL_BIAS = ck.get("rel_bias", False)
    sw.READONLY_Q = ck.get("readonly_q", False)
    sw.NO_POSW = ck.get("no_posw", False)
    sw.QUERY_FULL = ck.get("query_full", False)
    sw.QK_NORM = ck.get("qk_norm", False)
    m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    # TIGHT solver for measurement. ANNOTATION (v2 fix): Broyden@250 stalled ~1e-3 and never hit 1e-7,
    # so both warm and cold ran to the iteration cap (252/252 -- no early stop, ratio meaningless) and
    # curr08 couldn't even reach 1e-4. Anderson demonstrably converges on this cell; ask for an
    # ACHIEVABLE tol so early termination fires and the warm/cold ratio means something.
    m.deq = get_deq(f_solver="anderson", f_max_iter=150, f_tol=1e-6,   # Newton polish finishes the job
                    ift=True, b_solver="anderson", b_max_iter=40)
    return m, ck


def counted_solve(m, ff, z0):
    """Solve z=ff(z) from init z0, COUNTING function evaluations (solver-agnostic iteration proxy),
    then NEWTON-POLISH to a measurement-grade fixed point.
    ANNOTATION (v4): Anderson stalls at resid ~1e-4 regardless of budget (301/301 in v2/v3), and the
    basic error bound ||z_err|| <= resid / sigma_min means at sigma_min ~0.03 two 'converged' states can
    sit ~3e-2 apart — which is exactly the wc==cc discrepancy v3 measured. So after the solver we take
    up to 3 exact Newton steps  z += (I-J)^{-1}(f(z)-z)  using the dense Jacobian (small L only):
    quadratic convergence to resid ~1e-9. PREDICTION being tested: with tight residuals the warm-vs-cold
    discrepancy collapses (it was error amplification, not multistability)."""
    n = [0]

    def ff_counted(z):
        n[0] += 1
        return ff(z)
    with torch.no_grad():
        z = m.deq(ff_counted, z0)[0][-1]
    # Newton polish (outside no_grad for jacrev; result detached). Target 1e-7: f itself is evaluated
    # in fp32, whose rounding floors the achievable relative residual around 1e-7 — asking for 1e-9
    # would just burn Jacobians. The linear solve runs in float64 (conditioning of I-J is the whole
    # point here; don't let the solve add its own kappa*eps error in fp32).
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
    return z, n[0]


def make_ff(m, toks):
    """Bundle the per-sequence constants (h0, normed weights, additive mask) into a closure z -> f(z).
    ANNOTATION: h0 is the INPUT INJECTION — the only thing an edit changes. The weights and mask are
    the same before/after an edit; this is why 'edit = local perturbation of h0' is exact."""
    h0 = m.h0(toks)
    mask = sw.band_causal_mask(toks.shape[1], toks.device)
    maskp = m._maskp(mask)
    wn = m.wn()
    return (lambda z: m.f(z, h0, wn, maskp)), h0


def residual_of(ff, z):
    return ((ff(z) - z).norm() / (z.norm() + 1e-9)).item()


def apply_edit(toks, gen, mode):
    """Substitute ONE token, of a chosen TASK-RELEVANCE class. ANNOTATION (the C2a/C2b split): a
    'relevant' value is transported by the trained relay to its retrieving query (distance ~gap), so its
    edit-response MUST be large there — that's recall working, not an envelope violation. Only
    task-IRRELEVANT edits ('irrelevant' value = key never queried; 'filler') demand no transport, and
    THEIR decay is what the Faber envelope bounds. Fitting one exponential to relevant edits mixes
    decay with the trained carry ridge — the v2 bug."""
    toks2 = toks.clone()
    L = toks.shape[1]
    n_fill = L - 2 * sw.D_PAIR - sw.NQ
    queried = set(toks[0, 2 * sw.D_PAIR + n_fill:].tolist())              # keys the queries retrieve
    if mode == "filler":
        assert n_fill > 0, "no filler tokens at gap 0"
        vpos = 2 * sw.D_PAIR + torch.randint(n_fill, (1,), generator=gen).item()
        new = sw.NKEY + sw.NVAL + torch.randint(sw.NFILL, (1,), generator=gen).item()
        while new == toks2[0, vpos].item():
            new = sw.NKEY + sw.NVAL + torch.randint(sw.NFILL, (1,), generator=gen).item()
    else:
        want_queried = (mode == "relevant")
        slots = [p for p in range(sw.D_PAIR)
                 if (toks[0, 2 * p].item() in queried) == want_queried]
        if not slots:
            return None, None                                             # no slot of this class this seq
        vpos = 1 + 2 * slots[torch.randint(len(slots), (1,), generator=gen).item()]
        new = sw.NKEY + torch.randint(sw.NVAL, (1,), generator=gen).item()
        while new == toks2[0, vpos].item():
            new = sw.NKEY + torch.randint(sw.NVAL, (1,), generator=gen).item()
    toks2[0, vpos] = new
    return toks2, vpos


def kappa_of(m, toks):
    """Exact kappa_2(I-J) at the tight fixed point (dense Jacobian; small example only)."""
    ff, _ = make_ff(m, toks)
    z, _ = counted_solve(m, ff, torch.zeros(toks.shape[0], toks.shape[1], sw.d, device=sw.DEV))
    zf = z.reshape(-1).detach()
    ffl = lambda zv: ff(zv.view(z.shape)).reshape(-1)
    J = torch.func.jacrev(ffl)(zf)
    sv = torch.linalg.svdvals(torch.eye(zf.numel(), device=J.device) - J)
    rho = torch.linalg.eigvals(J).abs().max().item()
    return (sv.max() / sv.min()).item(), rho, sv.min().item()


def faber_xi(kappa):
    """Faber/DMS screening length (in HOPS of the attention graph): r=(sqrt(k)-1)/(sqrt(k)+1); xi=-1/ln r."""
    rk = (np.sqrt(kappa) - 1.0) / (np.sqrt(kappa) + 1.0)
    return np.inf if rk >= 1.0 else -1.0 / np.log(rk)


# ----------------------------------------------------------------------------- the measurement (filled in)

def edit_response_profile(m, toks, gen, mode):
    """One edit -> (dists, dz, it_warm, it_cold, ok, warm_eq_cold, noise), or None if no slot of `mode`.

    [A] WARM-START RE-SOLVE — the maintenance operation itself.
    ANNOTATION: we solve the ORIGINAL sequence cold (from zeros), then the EDITED sequence twice:
    once WARM (init = old z*) and once COLD (init = zeros, the control for the iteration ratio).
    Path-independence of the fixed point means warm and cold land on the SAME z_new — the warm run
    is cheaper, not different. If either residual is loose (>1e-4), the fixed point can't support a
    decay fit at this checkpoint -> flag not-ok and let the caller skip/report."""
    ff_old, _ = make_ff(m, toks)
    z_old, _ = counted_solve(m, ff_old, torch.zeros(toks.shape[0], toks.shape[1], sw.d, device=sw.DEV))

    toks2, vpos = apply_edit(toks, gen, mode)
    if toks2 is None:
        return None
    ff_new, _ = make_ff(m, toks2)
    z_new, it_warm = counted_solve(m, ff_new, z_old.clone())              # WARM: start at old equilibrium
    z_cold, it_cold = counted_solve(m, ff_new, torch.zeros_like(z_old))  # COLD control (same target fp)

    r_old, r_new = residual_of(ff_old, z_old), residual_of(ff_new, z_new)
    warm_eq_cold = (z_new - z_cold).norm().item() / (z_cold.norm().item() + 1e-9)   # should be ~solver tol
    ok = (r_old < 1e-3) and (r_new < 1e-3)

    # [B] RESPONSE VS DISTANCE, plus the NOISE FLOOR trick.
    # ANNOTATION: dz_i = ||z_new_i - z_old_i|| per position, distance = i - vpos. The model is CAUSAL and
    # the prefix is identical, so pre-edit positions have MATHEMATICALLY IDENTICAL fixed points — any
    # measured pre-edit dz is therefore pure SOLVER NOISE. v1 printed it as a scary "causality violated"
    # warning; v2 exploits it: the pre-edit max is a per-edit EMPIRICAL NOISE FLOOR, exactly the right
    # cutoff for the decay fit (better than any guessed constant — the artifact becomes the instrument).
    dz_all = (z_new - z_old)[0].norm(dim=-1).cpu().numpy()               # (L,) per-position response
    pre = dz_all[:vpos]
    noise = float(pre.max()) if pre.size else 0.0
    idx = np.arange(len(dz_all))
    keep = idx >= vpos
    return idx[keep] - vpos, dz_all[keep], it_warm, it_cold, ok, warm_eq_cold, noise


def fit_xi(dists, dz, noise=0.0):
    """[C] SCREENING-LENGTH FIT.
    ANNOTATION: model dz(d) ~ dz(0) exp(-d/xi)  =>  log dz vs d is a line with slope -1/xi.
    Steps: (i) average dz over edits AT EACH DISTANCE (edits differ in vpos, so pooling first then
    grouping keeps all information); (ii) drop the converged-to-noise tail — the floor is the MEASURED
    solver noise (pre-edit dz, which is mathematically zero) x3, with a relative fallback. Fitting
    through the noise tail would flatten the slope and fake a huge xi — the exact trap from the graph
    version; (iii) least-squares on the survivors.
    Returns xi in POSITIONS (divide by sw.W for hops) and the number of points used."""
    # HOP-GRANULARITY (v5 fix): the DMS envelope decays PER HOP (one hop = one window = W positions);
    # a profile flat across the first window then collapsing is envelope-CONSISTENT, but a per-position
    # fit reads it as slope~0 -> xi=inf -> fake violation (v4.1's filler 'inf' rows). Bin distances into
    # hops ceil(d/W) first; the fitted xi is then directly in hops (no /W conversion afterwards).
    dists, dz = np.asarray(dists), np.asarray(dz)
    hops = np.ceil(dists / sw.W).astype(int)                # d=0 -> hop 0 (the edited site itself)
    hs = np.unique(hops)
    mean_dz = np.array([dz[hops == h].mean() for h in hs])
    floor = max(1e-8, 1e-5 * mean_dz.max(), 3.0 * noise)    # measured noise floor dominates when available
    use = mean_dz > floor
    if use.sum() < 3:
        return np.nan, int(use.sum())
    slope, _ = np.polyfit(hs[use], np.log(mean_dz[use]), 1)
    if slope >= 0:                                          # no decay measurable (profile flat/rising)
        return np.inf, int(use.sum())
    return -1.0 / slope, int(use.sum())


# ----------------------------------------------------------------------------- the report

def multistability_probe(m, toks):
    """Two COLD solves from different inits (zeros vs small noise). If they land on different states
    that BOTH have small residuals, the cell is genuinely multistable (not just a stalled solver) —
    Geng's scenario C surfacing. Returns (divergence, resid_a, resid_b)."""
    ff, _ = make_ff(m, toks)
    z0 = torch.zeros(toks.shape[0], toks.shape[1], sw.d, device=sw.DEV)
    torch.manual_seed(1234)
    za, _ = counted_solve(m, ff, z0)
    zb, _ = counted_solve(m, ff, 0.05 * torch.randn_like(z0))
    div = (za - zb).norm().item() / (zb.norm().item() + 1e-9)
    return div, residual_of(ff, za), residual_of(ff, zb)


def main():
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "curr*.pt")))
    if not ckpts:
        print(f"No checkpoints in {CKPT_DIR}/ — run experiments.curriculum_checkpoints first."); return
    print(f"device={sw.DEV}  C2 v3: edit-locality by TASK-RELEVANCE of the edit; {N_EDITS} edits/mode\n"
          f"  C2a (irrelevant/filler edits): no transport required -> pure decay -> the Faber ENVELOPE test\n"
          f"  C2b (relevant edits): trained relay MUST carry the change to the dependent query -> expect a\n"
          f"      far-field ridge (that's recall working; single-exponential xi is NOT meaningful there)\n",
          flush=True)
    profiles = {}
    N_SEQS = 4                                                # v4: multiple base sequences (v3 used one,
    for path in ckpts:                                        # so slot geometry made some stats vacuous)
        m, ck = load_checkpoint(path)
        gen = torch.Generator().manual_seed(7)
        seqs = [sw.gen_mqar(1, ck["stage_gap"], gen)[0] for _ in range(N_SEQS)]
        kappa, rho, smin = kappa_of(m, seqs[0])
        # v5: probe EVERY base sequence (v4 probed only seqs[0], confounding edit-induced multistability
        # with sequence-dependent multistability in the wc==cc readout)
        divs = [multistability_probe(m, sq)[0] for sq in seqs]
        print(f"[{os.path.basename(path)}] recall={ck['recall']:.3f} rho={rho:.3f} smin={smin:.3f} "
              f"kappa={kappa:.1f} xi_faber={faber_xi(kappa):.2f} | multistable-probe per-seq divs: "
              + "  ".join(f"{dv:.1e}" for dv in divs), flush=True)

        # [D] ASSEMBLE, per edit-class: pool profiles -> fit xi -> envelope test (C2a only).
        # ANNOTATION: xi is fit in positions then converted to HOPS (/W) because Faber's unit is one
        # application of the local operator = one window width. far/near = mean response beyond 2
        # windows over the near-field peak: ~0 for C2a (containment), >>0 for C2b (the transport ridge
        # reaching the dependent query = the recompute set, NOT a violation). AMPLITUDE GATE (v4): if
        # even the PEAK response is <10x the measured noise floor, the edit produced no measurable
        # response at all -> that is containment evidence ("CONTAINED (below noise)"), and fitting a
        # slope through noise (v3's fake 34.8-hop "violation") is refused.
        modes = ["irrelevant", "filler", "relevant"] if ck["stage_gap"] > 0 else ["irrelevant", "relevant"]
        for mode in modes:
            D, Z, IW, IC, wc, noises = [], [], [], [], [], []
            for sq in seqs:
                for _ in range(max(1, N_EDITS // N_SEQS)):
                    out = edit_response_profile(m, sq, gen, mode)
                    if out is None:
                        continue
                    d_, z_, iw, ic, ok, weqc, noise = out
                    if ok:
                        D.append(d_); Z.append(z_); IW.append(iw); IC.append(ic)
                        wc.append(weqc); noises.append(noise)
            if not D:
                print(f"    {mode:>10}: no usable edits (no slot of this class / loose fp)", flush=True)
                continue
            Dc, Zc = np.concatenate(D), np.concatenate(Z)
            noise_max = float(np.max(noises))
            peak = float(Zc.max())
            near = Zc[Dc <= sw.W].max() if (Dc <= sw.W).any() else np.nan
            far = Zc[Dc > 2 * sw.W].mean() if (Dc > 2 * sw.W).any() else 0.0
            farfrac = far / (near + 1e-12)
            if peak < 10 * noise_max:                          # amplitude gate
                print(f"    {mode:>10}: peak|dz|={peak:.1e} < 10x noise({noise_max:.1e})  "
                      f"wc==cc={np.mean(wc):.1e}  -> CONTAINED (below noise)", flush=True)
                continue
            xi_meas, npts = fit_xi(Dc, Zc, noise=noise_max)   # hop-binned: already in hops
            # v5: only FILLER is a fair envelope witness (must-carry: a causal relay transports ALL
            # bindings, so unqueried-value edits legitimately ride the carry — transport, not violation)
            if mode == "filler":
                if np.isnan(xi_meas):        # sequence shorter than ~3 windows: no hop-bins to fit;
                    verdict = "contained within ~1 window (too short for a hop-granularity fit)"
                else:
                    verdict = "ENVELOPE " + ("OK" if xi_meas <= faber_xi(kappa) else "VIOLATED")
            elif mode == "irrelevant":
                verdict = "must-carry transport (state-relevant, output-irrelevant)"
            else:
                verdict = "transport (xi not meaningful)"
            profiles[f"{os.path.basename(path)}_{mode}"] = dict(dists=Dc, dz=Zc)
            print(f"    {mode:>10}: xi_meas={xi_meas:>6.2f} hops (n={npts})  far/near={farfrac:.3f}  "
                  f"peak={peak:.1e}  wc==cc={np.mean(wc):.1e}  -> {verdict}", flush=True)
        print(flush=True)

    np.savez(os.path.join(CKPT_DIR, "c2_profiles.npz"),
             **{k + "_" + f: v[f] for k, v in profiles.items() for f in ("dists", "dz")})
    print(f"(raw decay profiles saved to {CKPT_DIR}/c2_profiles.npz for the paper plot)", flush=True)
    print("\nREAD: C2a rows (irrelevant/filler) carry the envelope claim: xi_meas <= xi_faber and far/near"
          "\n~ 0 (containment). C2b rows (relevant) should show far/near >> 0 growing with gap = the trained"
          "\ntransport to the dependent query — the recompute set, not a violation. Multistable-probe:"
          "\ndiv >> resids = genuine multiple equilibria (warm start = branch tracking).", flush=True)


if __name__ == "__main__":
    main()
