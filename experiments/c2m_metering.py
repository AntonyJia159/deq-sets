"""C2m — emergent metering: does warm-start cost meter the OUTPUT (realized movement ||dz*||),
not the INPUT (||delta_h||), and not the problem size?

WHAT SURVIVED C2t: cost ~ realized movement (tier-3 of the claw-back ladder) was only ever measured as
a coarse 3-class ordering. This experiment tests it as a LAW, with the double dissociation that makes
it non-trivial (any solver "works harder on harder problems"; the claim is WHICH quantity is metered):

  (a) matched ||delta_h||, different movement -> cost differs   (cost is not metering the input)
  (b) matched movement, different ||delta_h|| -> cost same      (cost is metering the output)

Token edits can't control this (their delta_h is whatever the embedding table gives), so we add
SYNTHETIC h0-perturbations along the top/bottom right-singular directions of the resolvent's source
column (carry vs transverse, from the C2d machinery) at several magnitudes eps: same eps, carry vs
transverse = dissociation (a); small-eps-carry vs large-eps-transverse landing at similar ||dz|| =
dissociation (b). Real token edits (3 classes) stay in for ecological validity.

BASELINE (the flat toll): the COLD solve of the same edited problem — recompute-from-scratch pays a
roughly constant bill regardless of how little actually moved (the in-framework analog of a
feedforward net's fixed L-layer toll). Prediction: n_cold flat in ||dz||; n_warm ~ A*log10(||dz||)+B.

FORECAST TIE (C2d): pred = ||R @ delta_h|| is the a-priori bill forecast (computable pre-solve);
corr(log pred, log ||dz||) ~ 1 closes the loop certificate -> forecast -> meter.

FACES: causal (curr) + bidirectional (bidir) on the same axes — metering is face-agnostic as a
phenomenon; it is LOAD-BEARING causally (where the scalar certificate is vacuous) and a bonus
bidirectionally (where the Faber envelope already forecasts a priori). No eager/lazy language — that
framing died in C2t.

HONEST RISK, pre-registered: the few-eval floor may dominate small edits -> metering could come out
COARSE (a contained-vs-transporting step, not a smooth law). Report whichever it is.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.c2m_metering
"""
import glob
import os

import numpy as np
import torch

import experiments.sliding_window_reach as sw
from experiments.c2_bidir import load_ckpt, counted_solve
from experiments.c2_edit_locality import make_ff, apply_edit
from experiments.c2d_directional import dense_resolvent

CKPT_DIR = "checkpoints"
N_SEQS = 2
REAL_PER_MODE = 2
EPS_SET = [0.05, 0.2, 1.0, 3.0]
MIN_GAP = 16
PATTERNS = ["curr*.pt", "bidir1*.pt", "bidir2*.pt", "bidir4*.pt"]   # both faces; skip qv (weak substrate)
sw.H, sw.dh = 4, sw.d // 4


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4:
        return np.nan
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_pearson(n, a, b):
    """corr(n, a | b) — is input-norm (a) informative about cost (n) once movement (b) is known?"""
    n, a, b = map(lambda v: np.asarray(v, float), (n, a, b))
    if len(n) < 5:
        return np.nan
    r_na, r_nb, r_ab = (np.corrcoef(n, a)[0, 1], np.corrcoef(n, b)[0, 1], np.corrcoef(a, b)[0, 1])
    den = np.sqrt((1 - r_nb ** 2) * (1 - r_ab ** 2))
    return float((r_na - r_nb * r_ab) / den) if den > 1e-9 else np.nan


def make_ff_dh(m, toks, dh_full):
    """f with a SYNTHETIC input perturbation: h0 -> h0 + dh (dh in fp32, shaped (1,L,d))."""
    h0 = m.h0(toks) + dh_full
    mask = sw.band_causal_mask(toks.shape[1], toks.device)
    maskp = m._maskp(mask)
    wn = m.wn()
    return lambda z: m.f(z, h0, wn, maskp)


def main():
    ckpts = []
    for pat in PATTERNS:
        ckpts += sorted(glob.glob(os.path.join(CKPT_DIR, pat)))
    ckpts = [p for p in dict.fromkeys(ckpts) if "np" not in os.path.basename(p)
             and "qv" not in os.path.basename(p)]
    print(f"device={sw.DEV}  C2m emergent metering: n_warm vs realized ||dz|| (cold = flat-toll baseline,"
          f"\n  ||R dh|| = a-priori forecast); real edits x{REAL_PER_MODE}/mode + synthetic carry/transverse"
          f" x eps{EPS_SET}; {N_SEQS} seqs\n", flush=True)
    summary = []
    all_recs = []
    for path in ckpts:
        m, ck = load_ckpt(path)
        gap = ck["stage_gap"]
        if gap < MIN_GAP:
            continue
        name = os.path.basename(path)
        face = "bidir" if ck.get("bidir") else "causal"
        gen = torch.Generator().manual_seed(7)
        seqs = [sw.gen_mqar(1, gap, gen)[0] for _ in range(N_SEQS)]
        recs = []
        for toks in seqs:
            z0, ff0, J, R = dense_resolvent(m, toks)
            L = toks.shape[1]
            perts = []                                    # (label, toks2 or None, dh_full fp64 flat)
            for mode in (["filler", "irrelevant", "relevant"] if gap > 0 else []):
                for _ in range(REAL_PER_MODE):
                    out = apply_edit(toks, gen, mode)
                    if out[0] is None:
                        continue
                    with torch.no_grad():
                        dh = (m.h0(out[0]) - m.h0(toks)).detach()
                    perts.append((mode, out[0], dh))
            p_src = 1                                     # a value position in the binding region
            G = R[:, p_src * sw.d:(p_src + 1) * sw.d]
            _, _, Vh = torch.linalg.svd(G, full_matrices=False)
            v_top, v_bot = Vh[0], Vh[-1]                  # carry vs transverse input directions at p_src
            for eps in EPS_SET:
                for tag, v in (("syn-carry", v_top), ("syn-trans", v_bot)):
                    dh = torch.zeros(1, L, sw.d, device=sw.DEV)
                    dh[0, p_src] = (eps * v).float()
                    perts.append((tag, None, dh))
            for label, toks2, dh in perts:
                ff_new = make_ff(m, toks2)[0] if toks2 is not None else make_ff_dh(m, toks, dh)
                z_new, n_warm = counted_solve(m, ff_new, z0.clone())
                _, n_cold = counted_solve(m, ff_new, torch.zeros_like(z0))
                dz = (z_new - z0).norm().item()
                dh_flat = dh.reshape(-1).double()
                pred = (R @ dh_flat).norm().item()
                recs.append(dict(label=label, dh=float(dh_flat.norm()), dz=dz, pred=pred,
                                 nw=n_warm, nc=n_cold))
        if not recs:
            continue
        all_recs += [dict(r, ckpt=name, face=face) for r in recs]
        dz = np.array([r["dz"] for r in recs]); dh = np.array([r["dh"] for r in recs])
        nw = np.array([r["nw"] for r in recs]); nc = np.array([r["nc"] for r in recs])
        pred = np.array([r["pred"] for r in recs])
        use = dz > 1e-3                                   # above measurement noise
        floor_n = nw.min()
        floor_frac = float((nw <= floor_n + 1).mean())
        A, B = (np.polyfit(np.log10(dz[use]), nw[use], 1) if use.sum() >= 5 else (np.nan, np.nan))
        print(f"[{name}] face={face} gap={gap} smin={ck['sigma_min']:.3f} (n={len(recs)})\n"
              f"    LAW    : n_warm = {A:.2f}*log10||dz|| + {B:.2f}   "
              f"Spearman(n_warm, ||dz||)={spearman(nw[use], dz[use]):.2f}\n"
              f"    DISSOC : Spearman(n_warm, ||dh||)={spearman(nw[use], dh[use]):.2f}   "
              f"partial corr(n_warm, log||dh|| | log||dz||)={partial_pearson(nw[use], np.log10(dh[use] + 1e-12), np.log10(dz[use])):.2f}"
              f"   (metering OUTPUT iff first tracks ||dz|| and partial ~ 0)\n"
              f"    TOLL   : n_cold mean={nc.mean():.1f}  Spearman(n_cold, ||dz||)={spearman(nc[use], dz[use]):.2f}"
              f"   (flat = recompute charges regardless of movement)\n"
              f"    FORECAST: Spearman(pred, ||dz||)={spearman(pred[use], dz[use]):.2f}   "
              f"(||R dh|| computed pre-solve)\n"
              f"    FLOOR  : {floor_frac:.0%} of points at <= {floor_n + 1} evals "
              f"(coarseness check)\n", flush=True)
        summary.append((name, face, ck["sigma_min"], A))
    print("SLOPE vs CONDITIONING (law: slope ~ 1/log(1/rate), grows as smin falls):", flush=True)
    for name, face, smin, A in summary:
        print(f"    {name:<14} {face:<7} smin={smin:.3f}  slope={A:.2f}", flush=True)
    np.savez(os.path.join(CKPT_DIR, "c2m_records.npz"),
             labels=np.array([r["label"] for r in all_recs]),
             ckpts=np.array([r["ckpt"] for r in all_recs]),
             faces=np.array([r["face"] for r in all_recs]),
             dh=np.array([r["dh"] for r in all_recs]), dz=np.array([r["dz"] for r in all_recs]),
             pred=np.array([r["pred"] for r in all_recs]),
             nw=np.array([r["nw"] for r in all_recs]), nc=np.array([r["nc"] for r in all_recs]))
    print(f"\n(records -> {CKPT_DIR}/c2m_records.npz : n_warm-vs-||dz|| scatter w/ class+face markers, "
          f"cold flat line = the paper figure)", flush=True)
    print("\nREAD: metering holds if n_warm tracks ||dz|| (law), ||dh|| adds nothing given ||dz||\n"
          "(dissociation), n_cold is flat (toll), and ||R dh|| forecasts the bill (pre-solve). If the\n"
          "floor dominates, report metering as COARSE — a step, not a smooth law. Same shape on both\n"
          "faces = face-agnostic phenomenon; causally load-bearing, bidirectionally a bonus.", flush=True)


if __name__ == "__main__":
    main()
