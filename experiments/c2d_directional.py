"""C2d — the DIRECTIONAL certificate (causal claw-back, tier 1).

THE PROBLEM IT SOLVES: on the causal face the scalar xi-ball is sound but VACUOUS — sigma_min reports
the carry direction, so for any carry-exciting edit the ball ~ the whole suffix. But must-carry is
LOW-RANK: the relay keeps a few state directions at gain ~1 (the carry subspace) and screens the rest.
A per-edit DIRECTIONAL certificate exploits this: the edit's input perturbation delta_h is known BEFORE
solving (it is the embedding delta at the edit site), so project it through the FAR-REACH MAP and get an
a-priori, per-edit transport prediction — containment verdicts for transverse edits, rank-r carry
updates for the rest.

OBJECTS (per base sequence, exact-oracle level):
  R   = (I - J)^{-1} at the tight fixed point (dense, fp64 — toy scale affords the oracle)
  F_p = R[far rows, block column p]  — the far-reach map of edit site p (far = distance > 2w)
        SVD(F_p): top directions = the carry subspace AT THE SOURCE; sigma_1 >> sigma_k = low-rank carry
  pred_far(edit) = ||F_p @ delta_h_p||   (a-priori, pre-solve, O(n_far d^2) with the oracle;
                                          O(r d) once the carry basis is cached)

VALIDATIONS (the experiment):
  V1 PROFILE:    linear-response prediction (R @ delta_h) vs measured |dz| per position — does
                 first-order reasoning survive FINITE token substitutions? (log-log corr above noise)
  V2 TAXONOMY:   mean pred_far by edit class should reproduce filler < irrelevant < relevant — the
                 measured 3-tier table, now predicted a-priori from delta_h alone
  V3 SOUNDNESS:  measured_far <= slack * pred_far; count first-order violations (meas > 2x pred, above
                 noise). FALSE CONTAINMENT (pred below noise but meas transports) = the safety-critical
                 failure; count must be 0.
  V4 LOW-RANK:   singular spectrum of F_p — effective rank (participation ratio) and sigma_1/sigma_5
  V5 PRODUCT FORM (discharges the old debt): re-block into w-windows -> block-bidiagonal; coarse
                 transfer T_k = (I - D_k)^{-1} N_k; the T-product must RECONSTRUCT the exact R block
                 (nilpotent resummation is exact — identity check), and the scalar norm-product bound's
                 slack vs the directional product = what the scalar certificate loses.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.c2d_directional
"""
import glob
import math
import os

import numpy as np
import torch

import experiments.sliding_window_reach as sw
from experiments.c2_edit_locality import load_checkpoint, counted_solve, make_ff, apply_edit

CKPT_DIR = "checkpoints"
N_SEQS = 3
EDITS_PER_MODE_SEQ = 4
FAR_HOPS = 2                      # far = distance > FAR_HOPS * W (matches C2's far/near split)
sw.H, sw.dh = 4, sw.d // 4


def dense_resolvent(m, toks):
    """Tight fixed point, dense J, and the exact resolvent R = (I-J)^{-1} in fp64 (oracle; small L)."""
    ff, _ = make_ff(m, toks)
    z, _ = counted_solve(m, ff, torch.zeros(toks.shape[0], toks.shape[1], sw.d, device=sw.DEV))
    zf = z.reshape(-1).detach()
    ffl = lambda zv: ff(zv.view(z.shape)).reshape(-1)
    J = torch.func.jacrev(ffl)(zf).detach().double()
    N = zf.numel()
    R = torch.linalg.inv(torch.eye(N, device=J.device, dtype=torch.float64) - J)
    return z, ff, J, R


def far_reach_map(R, vpos, L, d):
    """F_p = R[far rows, block col vpos]; far = positions i with i - vpos > FAR_HOPS*W."""
    far_idx = [i for i in range(L) if i - vpos > FAR_HOPS * sw.W]
    if not far_idx:
        return None, far_idx
    rows = torch.cat([torch.arange(i * d, (i + 1) * d, device=R.device) for i in far_idx])
    return R[rows][:, vpos * d:(vpos + 1) * d], far_idx


def product_form_check(J, R, L, d):
    """V5: re-block into w-windows (block-bidiagonal by the re-blocking fact), verify the coarse
    T-product reconstructs the exact far R block (identity), and report the scalar norm-product
    slack vs the directional product per hop count."""
    w = sw.W
    nwin = math.ceil(L / w)
    if nwin < 3:
        return None
    bounds = [(k * w, min((k + 1) * w, L)) for k in range(nwin)]
    idx = lambda a, b: torch.arange(a * d, b * d, device=J.device)
    # band check: nothing below the previous window
    for k in range(2, nwin):
        a, b = bounds[k]
        pa, _ = bounds[k - 1]
        if J[idx(a, b)][:, :pa * d].abs().max().item() > 1e-9:
            return dict(error="band violation: J reaches beyond previous window")
    T = {}
    for k in range(1, nwin):
        a, b = bounds[k]
        pa, pb = bounds[k - 1]
        Dk = J[idx(a, b)][:, idx(a, b)]
        Nk = J[idx(a, b)][:, idx(pa, pb)]
        Ik = torch.eye((b - a) * d, device=J.device, dtype=torch.float64)
        T[k] = torch.linalg.solve(Ik - Dk, Nk)
    # identity: R[last window, first window] == T_{nwin-1}...T_1 (I - D_0)^{-1}
    a0, b0 = bounds[0]
    D0 = J[idx(a0, b0)][:, idx(a0, b0)]
    M = torch.linalg.inv(torch.eye((b0 - a0) * d, device=J.device, dtype=torch.float64) - D0)
    prod = M
    slacks = []
    for k in range(1, nwin):
        prod = T[k] @ prod
        prod_norm = torch.linalg.matrix_norm(
            torch.linalg.multi_dot([T[j] for j in range(k, 0, -1)]) if k > 1 else T[1], ord=2).item()
        scalar_bound = float(np.prod([torch.linalg.matrix_norm(T[j], ord=2).item()
                                      for j in range(1, k + 1)]))
        slacks.append((k, scalar_bound / max(prod_norm, 1e-30)))
    aL, bL = bounds[-1]
    R_block = R[idx(aL, bL)][:, idx(a0, b0)]
    relerr = (torch.linalg.matrix_norm(prod - R_block) /
              (torch.linalg.matrix_norm(R_block) + 1e-30)).item()
    hop_norms = [torch.linalg.matrix_norm(T[k], ord=2).item() for k in range(1, nwin)]
    return dict(relerr=relerr, slacks=slacks, hop_norms=hop_norms)


def main():
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "curr*.pt")))
    if not ckpts:
        print(f"No causal checkpoints in {CKPT_DIR}/"); return
    print(f"device={sw.DEV}  C2d directional certificate (causal face): far-reach map F_p = exact "
          f"resolvent oracle;\n  pred_far = ||F_p @ delta_h|| computed PRE-SOLVE; validated against the "
          f"measured nonlinear response.\n  far = distance > {FAR_HOPS}w; {N_SEQS} seqs x "
          f"{EDITS_PER_MODE_SEQ} edits/mode/seq\n", flush=True)
    all_records = []
    for path in ckpts:
        m, ck = load_checkpoint(path)
        gap = ck["stage_gap"]
        L = 2 * sw.D_PAIR + gap + sw.NQ
        gen = torch.Generator().manual_seed(7)
        seqs = [sw.gen_mqar(1, gap, gen)[0] for _ in range(N_SEQS)]
        print(f"[{os.path.basename(path)}] gap={gap} L={L} recall={ck['recall']:.3f} "
              f"sigma_min={ck['sigma_min']:.3f}", flush=True)

        # V5 on the first sequence (identity + scalar slack)
        z0, ff0, J0, R0 = dense_resolvent(m, seqs[0])
        pf = product_form_check(J0, R0, L, sw.d)
        if pf is None:
            print(f"    V5 product-form: skipped (<3 windows at L={L})", flush=True)
        elif "error" in pf:
            print(f"    V5 product-form: {pf['error']}", flush=True)
        else:
            sl = "  ".join(f"k={k}:{s:.1f}x" for k, s in pf["slacks"])
            hn = "  ".join(f"{h:.2f}" for h in pf["hop_norms"])
            print(f"    V5 product-form: coarse T-product reconstructs exact R block, "
                  f"relerr={pf['relerr']:.1e} (identity {'OK' if pf['relerr'] < 1e-6 else 'LOOSE'})\n"
                  f"       per-hop ||T_k||: {hn}\n"
                  f"       scalar norm-product slack vs directional product: {sl}", flush=True)

        if L - 1 <= FAR_HOPS * sw.W + 1:
            print(f"    (sequence too short for a far field at 2w={2*sw.W}; edit validations skipped)\n",
                  flush=True)
            continue

        modes = ["filler", "irrelevant", "relevant"] if gap > 0 else ["irrelevant", "relevant"]
        records = []
        for si, toks in enumerate(seqs):
            z, ff, J, R = (z0, ff0, J0, R0) if si == 0 else dense_resolvent(m, toks)
            for mode in modes:
                for _ in range(EDITS_PER_MODE_SEQ):
                    out = apply_edit(toks, gen, mode)
                    if out[0] is None:
                        continue
                    toks2, vpos = out
                    with torch.no_grad():
                        dh_full = (m.h0(toks2) - m.h0(toks)).reshape(-1).double()
                    dh_p = dh_full[vpos * sw.d:(vpos + 1) * sw.d]
                    # V1: full linear-response profile vs measured
                    pred_prof = (R @ dh_full).view(L, sw.d).norm(dim=-1).cpu().numpy()
                    ff2, _ = make_ff(m, toks2)
                    z2, _ = counted_solve(m, ff2, z.clone())
                    dz = (z2 - z)[0].norm(dim=-1).cpu().numpy()
                    noise = float(dz[:vpos].max()) if vpos > 0 else 1e-8
                    mask = (np.arange(L) >= vpos) & ((pred_prof > 3 * noise) | (dz > 3 * noise))
                    corr = np.nan
                    if mask.sum() >= 4:
                        corr = float(np.corrcoef(np.log10(pred_prof[mask] + 1e-12),
                                                 np.log10(dz[mask] + 1e-12))[0, 1])
                    # V2/V3/V4: far-reach map
                    F, far_idx = far_reach_map(R, vpos, L, sw.d)
                    rec = dict(mode=mode, ckpt=os.path.basename(path), corr=corr, noise=noise)
                    if F is not None:
                        sv = torch.linalg.svdvals(F).cpu().numpy()
                        pred_far = float((F @ dh_p).norm())
                        rows = np.concatenate([np.arange(i, i + 1) for i in far_idx])
                        meas_far = float(np.sqrt((dz[rows] ** 2).sum()))
                        n_far_coord = np.sqrt(len(far_idx))   # per-position noise norm x sqrt(#positions)
                        rec.update(pred_far=pred_far, meas_far=meas_far,
                                   sv1=float(sv[0]), sv5=float(sv[4]) if len(sv) > 4 else float(sv[-1]),
                                   r_eff=float((sv.sum() ** 2) / ((sv ** 2).sum() + 1e-30)),
                                   dh_norm=float(dh_p.norm()),
                                   noise_far=noise * float(n_far_coord))
                    records.append(rec)
        all_records += records

        # ---- per-checkpoint report
        withfar = [r for r in records if "pred_far" in r]
        corrs = [r["corr"] for r in records if np.isfinite(r.get("corr", np.nan))]
        if corrs:
            print(f"    V1 profile: log-log corr(pred, meas) mean={np.mean(corrs):.3f}  "
                  f"min={np.min(corrs):.3f}  (n={len(corrs)})", flush=True)
        if withfar:
            print("    V2 taxonomy (a-priori pred_far by class -> should be monotone):", flush=True)
            for mode in modes:
                sel = [r for r in withfar if r["mode"] == mode]
                if sel:
                    pf_ = np.mean([r["pred_far"] for r in sel])
                    mf_ = np.mean([r["meas_far"] for r in sel])
                    print(f"       {mode:>10}: pred_far={pf_:.3e}  meas_far={mf_:.3e}  (n={len(sel)})",
                          flush=True)
            above = [r for r in withfar if r["meas_far"] > 3 * r["noise_far"]]
            viol = [r for r in above if r["meas_far"] > 2 * r["pred_far"]]
            ratios = [r["meas_far"] / (r["pred_far"] + 1e-30) for r in above]
            contained_pred = [r for r in withfar if r["pred_far"] < 3 * r["noise_far"]]
            false_cont = [r for r in contained_pred if r["meas_far"] > 10 * r["noise_far"]]
            print(f"    V3 soundness: meas/pred ratio median={np.median(ratios) if ratios else np.nan:.2f} "
                  f" max={np.max(ratios) if ratios else np.nan:.2f}  first-order violations "
                  f"(meas>2x pred): {len(viol)}/{len(above)}", flush=True)
            print(f"       containment safety: {len(contained_pred)} edits predicted-contained, "
                  f"FALSE containments (meas>10x noise): {len(false_cont)}"
                  f"{'  <-- CERTIFICATE VIOLATION' if false_cont else '  (0 = sound)'}", flush=True)
            r_effs = [r["r_eff"] for r in withfar]
            rk = np.mean([r["sv1"] / (r["sv5"] + 1e-30) for r in withfar])
            print(f"    V4 low-rank carry: effective rank r_eff mean={np.mean(r_effs):.1f} "
                  f"(of d={sw.d})  sigma1/sigma5 mean={rk:.1f}", flush=True)
        print(flush=True)

    np.savez(os.path.join(CKPT_DIR, "c2d_records.npz"),
             **{f"r{i}_{k}": v for i, r in enumerate(all_records) for k, v in r.items()
                if not isinstance(v, str)},
             modes=np.array([r["mode"] for r in all_records]),
             ckpts=np.array([r["ckpt"] for r in all_records]))
    print(f"(records saved to {CKPT_DIR}/c2d_records.npz for the pred-vs-meas scatter)", flush=True)
    print("\nREAD: V1 high corr = first-order reasoning survives finite token edits. V2 monotone = the\n"
          "3-tier taxonomy is PREDICTED a-priori from delta_h alone. V3 zero false containments = the\n"
          "per-edit verdict is safe. V4 r_eff << d = the carry is low-rank (rank-r update, not full\n"
          "suffix). V5 relerr ~ fp precision = the coarse product form IS the exact resolvent\n"
          "(re-blocking theorem operationalized); slack column = what the scalar bound loses.", flush=True)


if __name__ == "__main__":
    main()
