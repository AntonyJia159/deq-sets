"""C5-WOODBURY+BROYDEN (smoke test) — does swapping the SOLVER from Anderson to Broyden let the low-residual
Woodbury init actually pay off in fewer iterations?

BACKGROUND (c5_woodbury_prior finding): the Woodbury cross-edit prior z0 = z*_old + (I-J)^{-1} dh is a strictly
better INIT (1.8-65x lower residual) but ANDERSON does NOT convert that into fewer iters -- its convergence
tracks slow-mode/carry content, not initial residual (curr16 filler: 8x lower residual yet MORE iters). The
diagnosis was: a quasi-Newton solver converges superlinearly near the solution and SHOULD cash a low-residual
start. Broyden is that solver (it builds J^{-1} internally via Sherman-Morrison, Bai 2019). So the ONLY change
from c5_woodbury_prior is: f_solver='broyden' and a TIGHT tol (Broyden can reach 1e-8; Anderson floors ~1e-4).
No custom solver, no seeding of the Jacobian estimate -- off-the-shelf Broyden, Woodbury only as the init.

WHAT'S (tentatively) NOVEL: the cross-EDIT Woodbury prior itself (measure the edit's low-rank footprint ->
predict the NEW equilibrium -> solve from there); Broyden's Sherman-Morrison is standard and solver-INTERNAL.
Not a priority for THIS paper -- a scoped probe for the efficiency spin-off (large practical DEQs). Report straight.

Verdict: iters(Broyden, Woodbury-init) < iters(Broyden, warm) AND < iters(Anderson, Woodbury-init) => the pairing
cashes the prior. If Broyden ~ Anderson, the init advantage still isn't converting; if Broyden diverges on
near-singular curr40, quarantine it (report straight).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.c5_woodbury_broyden
"""
import glob
import os

import numpy as np
import torch
from torchdeq import get_deq

import experiments.sliding_window_reach as sw
from experiments.c2_edit_locality import load_checkpoint, make_ff, apply_edit
from experiments.c2d_directional import dense_resolvent

CKPT_DIR = "checkpoints"
N_SEQS = 2
EDITS_PER_MODE_SEQ = 2
TOL = 1e-4                 # ACHIEVABLE for both (Anderson floors ~1e-4) so the read is speed-to-shared-tol, not
MAXIT = 200                # who-reaches-the-impossible-tol. Confirms the null isn't a capping artifact.
sw.H, sw.dh = 4, sw.d // 4


def iters_to_tol(solver, ff, z_init):
    """f-evals for `solver` (early-stop TOL) from z_init; (n_evals, final_rel_resid, converged). fp32, no seeding."""
    n = [0]

    def fc(z):
        n[0] += 1
        return ff(z)
    deq = get_deq(f_solver=solver, f_max_iter=MAXIT, f_tol=TOL, ift=True, b_solver="anderson", b_max_iter=1)
    with torch.no_grad():
        try:
            z = deq(fc, z_init.clone())[0][-1]
            rel = float((ff(z) - z).norm() / (z.norm() + 1e-9))
        except Exception:
            return n[0], float("nan"), False
    return n[0], rel, np.isfinite(rel) and rel < TOL


def main():
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "curr*.pt")))
    if not ckpts:
        print(f"No causal checkpoints in {CKPT_DIR}/"); return
    print(f"device={sw.DEV}  C5 Woodbury+Broyden smoke test. Does swapping Anderson->Broyden let the low-residual\n"
          f"  Woodbury init (z*_old + R@dh) convert into fewer iters? Off-the-shelf Broyden, Woodbury only as init.\n"
          f"  Metric = f-evals to rel-resid < {TOL:g}. {N_SEQS} seqs x {EDITS_PER_MODE_SEQ} edits/mode.\n",
          flush=True)
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
        modes = ["filler", "relevant"] if gap > 0 else ["relevant"]
        rows = []
        for toks in seqs:
            z_old, ff, J, R = dense_resolvent(m, toks)          # old equilibrium + exact resolvent (for the prior)
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
                    dz_pred = (R @ dh_full)                      # Woodbury linear-response prediction of new eq
                    init_wood = z_old + dz_pred.view(1, L, d).float()

                    # 2 solvers x {warm, Woodbury} init  (+ Broyden cold for context). Anderson-Woodbury is the
                    # incumbent that didn't pay; Broyden-Woodbury is the pairing under test.
                    aw = iters_to_tol("anderson", ff2, z_old)
                    ao = iters_to_tol("anderson", ff2, init_wood)
                    bc = iters_to_tol("broyden", ff2, z_cold)
                    bw = iters_to_tol("broyden", ff2, z_old)
                    bo = iters_to_tol("broyden", ff2, init_wood)

                    rows.append(dict(mode=mode,
                                     aw=aw[0], caw=int(aw[2]), ao=ao[0], cao=int(ao[2]),
                                     bc=bc[0], cbc=int(bc[2]), bw=bw[0], cbw=int(bw[2]),
                                     bo=bo[0], rbo=bo[1], cbo=int(bo[2])))
        if not rows:
            print("    (no usable edits)\n", flush=True); continue
        all_rows += [dict(ckpt=os.path.basename(path), **r) for r in rows]

        def mean(key, sel):
            return float(np.mean([r[key] for r in sel]))
        for mode in modes:
            sel = [r for r in rows if r["mode"] == mode]
            if not sel:
                continue
            print(f"    {mode:>10}: f-evals@{TOL:g} (conv%)  |  Anderson warm={mean('aw', sel):.1f}"
                  f"({mean('caw', sel):.0%}) Woodbury={mean('ao', sel):.1f}({mean('cao', sel):.0%})   ||   "
                  f"Broyden cold={mean('bc', sel):.1f}({mean('cbc', sel):.0%}) warm={mean('bw', sel):.1f}"
                  f"({mean('cbw', sel):.0%}) Woodbury={mean('bo', sel):.1f}({mean('cbo', sel):.0%})", flush=True)
        print(flush=True)

    if all_rows:
        np.savez(os.path.join(CKPT_DIR, "c5_woodbury_broyden_records.npz"),
                 **{k: np.array([r.get(k, np.nan) for r in all_rows])
                    for k in all_rows[0] if k not in ("ckpt", "mode")},
                 ckpt=np.array([r["ckpt"] for r in all_rows]),
                 mode=np.array([r["mode"] for r in all_rows]))
    print("READ: the pairing PAYS iff Broyden-Woodbury has the fewest evals AND beats Anderson-Woodbury (the\n"
          "incumbent that didn't cash the prior). Broyden ~ Anderson => the init advantage still isn't converting;\n"
          "conv% < 100 on near-singular curr40 => Broyden unstable there, quarantine (report straight).", flush=True)


if __name__ == "__main__":
    main()
