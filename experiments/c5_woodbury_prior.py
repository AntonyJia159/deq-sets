"""C5-WOODBURY-PRIOR (smoke test) — is the low-rank resolvent prediction a "warmer-than-warm" start?

THE ONE CLAIM, ISOLATED (residual only, NO timing): after a content edit, does initializing the re-solve at
    z_pred = z*_old + (I-J)^{-1} delta_h        (the linear-response / Woodbury prediction of the NEW equilibrium)
land CLOSER to the new fixed point than plain warm-start (z*_old) — measured purely by the residual
||f(z)-z|| at each init? If yes, resid(Woodbury) < resid(warm) < resid(cold). This is a pure accuracy
statement about the PRIOR; efficiency (iterations, wall-clock) is downstream and NOT the point here. It can
fail informatively: a nonlinear-enough edit can make the linear prediction OVERSHOOT -> worse than warm.

Two levels:
  L1 (premise):  full-resolvent prediction z*_old + R @ delta_h  (oracle predictor). Tests "is linear response
                 a good init at all?" (C2d-V1 already showed the response PROFILE matches; this asks whether it
                 is close enough to be a useful START.)
  L2 (the cheap trick): rank-r truncation of that predicted displacement field (top-r SVD of the (L,d) shift).
                 Tests whether the LOW-RANK / Woodbury-cheap version keeps the benefit (C2d-V4: carry ~ rank 8).
                 If resid(rank-r) ~ resid(full) << resid(warm), the deployable cheap predictor is justified.

Reuses the C2d exact-resolvent oracle verbatim. Causal (curr*) checkpoints = the guaranteed path; bidir is a
trivial extension (swap the loader). Secondary (clearly labelled): iters from warm vs Woodbury init, for
context only -- the verdict is the residual.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.c5_woodbury_prior
"""
import glob
import os

import numpy as np
import torch

import experiments.sliding_window_reach as sw
from experiments.c2_edit_locality import load_checkpoint, counted_solve, make_ff, apply_edit
from experiments.c2d_directional import dense_resolvent

CKPT_DIR = "checkpoints"
N_SEQS = 3
EDITS_PER_MODE_SEQ = 4
RANKS = [4, 8, 16]
sw.H, sw.dh = 4, sw.d // 4


def residual(ff, z_init):
    """||f(z)-z|| at a given init state (shape [1,L,d]); one function eval, no solve."""
    with torch.no_grad():
        return float((ff(z_init) - z_init).norm())


def main():
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "curr*.pt")))
    if not ckpts:
        print(f"No causal checkpoints in {CKPT_DIR}/"); return
    print(f"device={sw.DEV}  C5 Woodbury-prior smoke test (residual only): does z*_old + R@delta_h beat plain\n"
          f"  warm-start (z*_old) as an init? verdict = resid(Woodbury) < resid(warm) < resid(cold).\n"
          f"  L1 full-resolvent predictor; L2 rank-r truncation (the cheap version). {N_SEQS} seqs x "
          f"{EDITS_PER_MODE_SEQ} edits/mode.\n", flush=True)
    all_rows = []
    for path in ckpts:
        m, ck = load_checkpoint(path)
        gap = ck["stage_gap"]
        L = 2 * sw.D_PAIR + gap + sw.NQ
        d = sw.d
        gen = torch.Generator().manual_seed(7)
        seqs = [sw.gen_mqar(1, gap, gen)[0] for _ in range(N_SEQS)]
        print(f"[{os.path.basename(path)}] gap={gap} L={L} recall={ck['recall']:.3f} "
              f"sigma_min={ck['sigma_min']:.3f}", flush=True)
        modes = ["filler", "irrelevant", "relevant"] if gap > 0 else ["irrelevant", "relevant"]
        rows = []
        for toks in seqs:
            z_old, ff, J, R = dense_resolvent(m, toks)     # old equilibrium + exact resolvent oracle
            z_cold = torch.zeros(1, L, d, device=sw.DEV)
            for mode in modes:
                for _ in range(EDITS_PER_MODE_SEQ):
                    out = apply_edit(toks, gen, mode)
                    if out[0] is None:
                        continue
                    toks2, vpos = out
                    with torch.no_grad():
                        dh_full = (m.h0(toks2) - m.h0(toks)).reshape(-1).double()
                    ff2, _ = make_ff(m, toks2)
                    dz_pred = (R @ dh_full)                 # [L*d] fp64 — the linear-response prediction

                    r_cold = residual(ff2, z_cold)
                    r_warm = residual(ff2, z_old)
                    init_full = z_old + dz_pred.view(1, L, d).float()
                    r_full = residual(ff2, init_full)

                    # L2: rank-r truncation of the (L,d) predicted displacement field
                    dzp = dz_pred.view(L, d)
                    U, S, Vh = torch.linalg.svd(dzp, full_matrices=False)
                    r_rank = {}
                    for r in RANKS:
                        rr = min(r, S.numel())
                        dzr = (U[:, :rr] * S[:rr]) @ Vh[:rr]
                        init_r = z_old + dzr.view(1, L, d).float()
                        r_rank[r] = residual(ff2, init_r)

                    # secondary (context only): iterations to converge from warm vs Woodbury init
                    _, it_warm = counted_solve(m, ff2, z_old.clone())
                    _, it_wood = counted_solve(m, ff2, init_full.clone())

                    rows.append(dict(mode=mode, r_cold=r_cold, r_warm=r_warm, r_full=r_full,
                                     it_warm=it_warm, it_wood=it_wood,
                                     **{f"r_rank{r}": r_rank[r] for r in RANKS}))
        if not rows:
            print("    (no usable edits)\n", flush=True); continue
        all_rows += [dict(ckpt=os.path.basename(path), **r) for r in rows]

        # ---- per-checkpoint report
        def mean(key, sel):
            return float(np.mean([r[key] for r in sel]))
        for mode in modes:
            sel = [r for r in rows if r["mode"] == mode]
            if not sel:
                continue
            rc, rw, rf = mean("r_cold", sel), mean("r_warm", sel), mean("r_full", sel)
            gain = rw / (rf + 1e-30)                        # >1 => Woodbury prior is closer than warm
            rank_str = "  ".join(f"r{r}:{mean(f'r_rank{r}', sel):.2e}" for r in RANKS)
            iw, iwo = mean("it_warm", sel), mean("it_wood", sel)
            verdict = "OK (closer)" if rf < rw else "NO (overshoot: worse than warm)"
            print(f"    {mode:>10}: resid  cold={rc:.2e}  warm={rw:.2e}  Woodbury(full)={rf:.2e}  "
                  f"[{gain:.2f}x closer than warm -> {verdict}]", flush=True)
            print(f"                rank-trunc: {rank_str}    (retains benefit if ~ Woodbury-full)", flush=True)
            print(f"                iters (context only): warm={iw:.1f}  Woodbury={iwo:.1f}", flush=True)
        print(flush=True)

    if all_rows:
        np.savez(os.path.join(CKPT_DIR, "c5_woodbury_records.npz"),
                 **{k: np.array([r.get(k, np.nan) for r in all_rows])
                    for k in all_rows[0] if k not in ("ckpt", "mode")},
                 ckpt=np.array([r["ckpt"] for r in all_rows]),
                 mode=np.array([r["mode"] for r in all_rows]))
    print("READ: verdict = resid(Woodbury full) < resid(warm) < resid(cold) => the linear-response prediction is\n"
          "a strictly better init (the 'warmer-than-warm' prior holds). rank-trunc ~ Woodbury-full => the CHEAP\n"
          "low-rank version keeps the benefit (deployable). If Woodbury > warm on some class = overshoot, the\n"
          "prediction is too nonlinear there (report straight). iters are context only; efficiency is C5's job.", flush=True)


if __name__ == "__main__":
    main()
