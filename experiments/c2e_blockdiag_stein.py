"""C2e -- the FREE block-diagonal-Stein directional charge (bidir face).

WHAT THIS TESTS. Fable's observation: the dense Stein Gramian P (c2_weighted_cert) already contains a
SOUND, source-and-direction-anisotropic directional charge, at zero extra solve. If an edit delta_h is
supported in window j, then x*Px = x_j* P_jj x_j EXACTLY (off-diagonal blocks drop, single nonzero block),
so with P >= I the adapted-norm bound gives

    ||G^k u||_2 <= sqrt(u_j* P_jj u_j) * eff_rate^k ,   u = D^{-1} delta_h  (supported in window j),

and summing the tail from the min far block-distance d0 (paths shorter than d0 cannot reach the far window):

    ||(R delta_h)_far||  <=  sqrt(u_j* P_jj u_j) * ||...|| * eff_rate^d0 / (1 - eff_rate)   [SOUND].

This is the CERTIFIED cousin of C2d's directional charge pred_far = ||F_p delta_h|| (exact linear response,
TIGHT but not sound). We compare four quantities per edit class:
    pred_far   = ||F_p delta_h||                          exact linear far-response  (C2d, tight-not-sound)
    sound_far  = sqrt(u_j* P_jj u_j) * eff^d0/(1-eff)      free block-diag Stein       (SOUND + anisotropic)
    global_far = sqrt(lam_max(P)) * ||u_j|| * eff^d0/(1-eff)   isotropic tier-1        (SOUND, whole-P const)
    meas_far   = ||dz_far||                                measured nonlinear response (reality check)
Expected: pred_far <= sound_far <= global_far.  sound/pred = tightness of the sound bound; global/sound =
what the source+direction anisotropy buys over the current isotropic constant. And: does sqrt(u_j*P_jj u_j)
DISCRIMINATE filler < irrelevant < relevant the way the heuristic charge does?

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.c2e_blockdiag_stein
"""
import glob
import os

import numpy as np
import torch

import experiments.sliding_window_reach as sw
from experiments.c2_edit_locality import load_checkpoint, counted_solve, make_ff, apply_edit
from experiments.c2d_directional import dense_resolvent, far_reach_map
from experiments.c2_weighted_cert import power_lammax

CKPT_DIR = "checkpoints"
sw.H, sw.dh = 4, sw.d // 4
N_SEQS = 3
EDITS_PER_MODE_SEQ = 4
FAR_HOPS = 2                      # far = token distance > FAR_HOPS*W (matches C2/C2d)
R_GRID = [0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]


def block_decomp(J, L, d, w):
    """Reblock M=I-J into w-token windows. Return G=-D^{-1}(M-D), the dense block-diagonal Dinv, and the
    per-window (dim_lo, dim_hi) block ranges. fp32 (GPU-fast, memory-light) -- matches c2_weighted_cert."""
    n = J.shape[0]
    M = torch.eye(n, device=J.device, dtype=torch.float64) - J.double()
    idx = [(s, min(s + w, L)) for s in range(0, L, w)]        # window token ranges
    Dinv = torch.zeros_like(M)
    off = M.clone()
    blocks = []
    for (ta, tb) in idx:
        a, b = ta * d, tb * d
        Dinv[a:b, a:b] = torch.linalg.inv(M[a:b, a:b])
        off[a:b, a:b] = 0.0
        blocks.append((a, b))
    G = -(Dinv @ off)
    return G.float(), Dinv, blocks


def build_gramian(G, r, Kmax=800, tail_thresh=0.9):
    """Early-stopped observability Gramian P = sum_j (Gt^T)^j Gt^j, Gt=G/r (c2_weighted_cert construction).
    Returns (P, eff_rate, lam_max) or None if r <= rho(G) (Gramian diverges). P >= I, so it is a valid
    Lyapunov certificate as soon as ||Gt^{k+1}||_F < 1."""
    n = G.shape[0]
    Gt = G / r
    P = torch.eye(n, dtype=G.dtype, device=G.device)
    Gk = Gt.clone()
    for _ in range(1, Kmax):
        P = P + Gk.T @ Gk
        Gk = Gk @ Gt
        tf = torch.linalg.norm(Gk).item()
        if tf < tail_thresh:
            break
        if tf > 1e6:
            return None
    if tf >= 1.0:
        return None
    lam = power_lammax(P)
    cP = np.sqrt(max(1.0 - 1.0 / max(lam, 1.0), 0.0))
    return P, r * cP, lam


def common_rate(Gs):
    """Smallest r on the grid valid for EVERY sequence's G -> a single certified operating point for the
    whole checkpoint (so the per-class aggregates are on one common rate, not pooled across seq-specific r).
    Returns (r, [(P,eff,lam) per seq]) or None."""
    for r in R_GRID:
        outs = [build_gramian(G, r) for G in Gs]
        if all(o is not None for o in outs):
            return r, outs
    return None


def main():
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "bidir*.pt")))
    if not ckpts:
        print(f"No bidir checkpoints in {CKPT_DIR}/"); return
    print(f"device={sw.DEV}  C2e: FREE block-diagonal-Stein directional charge (bidir face)\n"
          f"  sound_far = sqrt(u_j* P_jj u_j)*eff^d0/(1-eff) from the CACHED dense Gramian; u=D^-1 delta_h.\n"
          f"  compare vs pred_far=||F_p dh|| (C2d exact linear, tight-not-sound), global_far (isotropic\n"
          f"  sqrt(lam_max(P))), meas_far (nonlinear). {N_SEQS} seqs x {EDITS_PER_MODE_SEQ} edits/class.\n",
          flush=True)
    for path in ckpts:
        m, ck = load_checkpoint(path)
        if "W" in ck:
            sw.W = ck["W"]
        gap = ck["stage_gap"]
        L = 2 * sw.D_PAIR + gap + sw.NQ
        gen = torch.Generator().manual_seed(7)
        seqs = [sw.gen_mqar(1, gap, gen)[0] for _ in range(N_SEQS)]
        w = sw.W
        print(f"[{os.path.basename(path)}] gap={gap} L={L} W={w} recall={ck['recall']:.3f} "
              f"sigma_min={ck.get('sigma_min', float('nan')):.3f}", flush=True)
        if L - 1 <= FAR_HOPS * w + 1:
            print("    (too short for a far field; skipped)\n", flush=True); continue

        modes = ["filler", "irrelevant", "relevant"] if gap > 0 else ["irrelevant", "relevant"]
        recs = []
        # Pass 1: build the oracle + block operator for every sequence, THEN pick ONE common rate r valid
        # for all of them (so per-class aggregates share a single certified operating point).
        per_seq = []
        for toks in seqs:
            z, ff, J, R = dense_resolvent(m, toks)
            G, Dinv, blocks = block_decomp(J, L, sw.d, w)
            per_seq.append((toks, z, ff, R, G, Dinv, blocks))
        cr = common_rate([d[4] for d in per_seq])
        if cr is None:
            print("    (no single rate valid for all seqs on grid; skipped)\n", flush=True); continue
        r, grams = cr
        # per-window source anisotropy from seq0: sqrt(lam_max(P_jj)) vs the global sqrt(lam_max(P))
        P0, eff0, lam0 = grams[0]
        blocks0 = per_seq[0][6]
        per_win = np.array([np.sqrt(max(power_lammax(P0[a:b, a:b].contiguous()), 1.0)) for (a, b) in blocks0])
        print(f"    r={r:.2f} (common) eff_rate={eff0:.3f} global const sqrt(lam_max P)={np.sqrt(lam0):.1f} | "
              f"per-window sqrt(lam_max P_jj): min={per_win.min():.2f} med={np.median(per_win):.2f} "
              f"max={per_win.max():.2f}  (anisotropy {per_win.max()/max(per_win.min(),1e-9):.1f}x)", flush=True)
        # Pass 2: edits, each seq at the common rate r
        for (toks, z, ff, R, G, Dinv, blocks), (P, eff, lam) in zip(per_seq, grams):
            glob_const = np.sqrt(max(lam, 1.0))
            Pd = P.double()
            for mode in modes:
                for _ in range(EDITS_PER_MODE_SEQ):
                    toks2, vpos = apply_edit(toks, gen, mode)
                    if toks2 is None:
                        continue
                    jwin = vpos // w
                    a, b = blocks[jwin]
                    with torch.no_grad():
                        dh_full = (m.h0(toks2) - m.h0(toks)).reshape(-1).double()
                    dh_j = dh_full[a:b]                               # window block (zero except vpos slot)
                    u_j = Dinv[a:b, a:b] @ dh_j                       # D^-1 delta_h, supported in window j
                    charge_sound = float(torch.sqrt(torch.clamp(u_j @ (Pd[a:b, a:b] @ u_j), min=0.0)))
                    charge_glob = glob_const * float(torch.linalg.norm(u_j))
                    # far region + min block distance d0 (paths shorter than d0 can't reach any far window)
                    far_tok = [p for p in range(L) if p - vpos > FAR_HOPS * w]
                    if not far_tok:
                        continue
                    d0 = min((p // w) - jwin for p in far_tok)
                    d0 = max(d0, 1)
                    tail = eff ** d0 / (1.0 - eff)
                    sound_far = charge_sound * tail
                    global_far = charge_glob * tail
                    # C2d exact linear far-response (single-token column) + measured nonlinear
                    F, far_idx = far_reach_map(R, vpos, L, sw.d)
                    dh_p = dh_full[vpos * sw.d:(vpos + 1) * sw.d]
                    pred_far = float((F @ dh_p).norm()) if F is not None else float("nan")
                    ff2, _ = make_ff(m, toks2)
                    z2, _ = counted_solve(m, ff2, z.clone())
                    dz = (z2 - z)[0].norm(dim=-1).double()
                    meas_far = float(torch.sqrt((dz[far_tok] ** 2).sum()))
                    recs.append(dict(mode=mode, pred=pred_far, sound=sound_far,
                                     glob=global_far, meas=meas_far))

        # ---- per-checkpoint report
        print(f"    {'class':>11} | {'meas_far':>10} {'pred_far':>10} {'sound_far':>10} {'global_far':>11}"
              f" | {'sound/pred':>10} {'glob/sound':>10}", flush=True)
        for mode in modes:
            sel = [x for x in recs if x["mode"] == mode]
            if not sel:
                continue
            gm = lambda k: float(np.exp(np.mean(np.log([max(x[k], 1e-30) for x in sel]))))  # geo-mean
            meas, pred, snd, glb = gm("meas"), gm("pred"), gm("sound"), gm("glob")
            print(f"    {mode:>11} | {meas:10.2e} {pred:10.2e} {snd:10.2e} {glb:11.2e}"
                  f" | {snd/max(pred,1e-30):10.1f} {glb/max(snd,1e-30):10.1f}  (n={len(sel)})", flush=True)
        # soundness: sound_far must dominate the EXACT linear pred_far (both linear -> no nonlinearity alibi)
        viol = [x for x in recs if x["sound"] < 0.999 * x["pred"]]
        vg = [x for x in recs if x["glob"] < 0.999 * x["sound"]]
        print(f"    soundness: sound_far >= pred_far violations {len(viol)}/{len(recs)}"
              f"{'  <-- UNSOUND' if viol else '  (0 = sound)'};  "
              f"global >= sound violations {len(vg)}/{len(recs)}", flush=True)
        # discrimination: is the sound charge monotone across classes (filler<irrelevant<relevant)?
        order = [m2 for m2 in ["filler", "irrelevant", "relevant"] if any(x["mode"] == m2 for x in recs)]
        gmeans = {m2: float(np.exp(np.mean(np.log([max(x["sound"], 1e-30)
                  for x in recs if x["mode"] == m2])))) for m2 in order}
        mono = all(gmeans[order[i]] <= gmeans[order[i + 1]] for i in range(len(order) - 1))
        print(f"    discrimination (sound_far geo-mean): "
              + "  ".join(f"{m2}={gmeans[m2]:.2e}" for m2 in order)
              + f"  -> {'MONOTONE' if mono else 'NOT monotone'}\n", flush=True)

    print("READ: sound_far>=pred_far with 0 violations = the free block-diagonal charge is a genuine SOUND\n"
          "upper bound on the exact directional response (tight tier-1 becomes sound at zero solve cost).\n"
          "glob/sound = anisotropy gain over the current isotropic constant. MONOTONE = v*P_jj v recovers\n"
          "the filler<irrelevant<relevant taxonomy -- the sound charge discriminates edit classes.", flush=True)


if __name__ == "__main__":
    main()
