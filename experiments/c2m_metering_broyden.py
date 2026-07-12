"""C2m under Broyden — is the causal-face metering weakness an ANDERSON (spectral-gap) artifact?

C2m (Anderson) found: n_warm meters realized ||dz|| as a clean LAW on the BIDIR face (Spearman ~0.90,
partial corr(n,||dh|| | ||dz||)~0) but WEAKLY + mode-confounded on the CAUSAL face (Spearman ~0.67,
NEGATIVE partials: carry-aligned movement = slow modes = disproportionate cost). Its stated mechanism is
exactly the spectral gap: near-normal J -> uniform mode rates -> magnitude metering; non-normal causal J ->
wildly varying mode rates -> raw ||dz|| under-determines cost.

But that mechanism is FIXED-POINT-ITERATION-specific. Anderson is spectral-gap-limited; Broyden (quasi-Newton)
is affine-invariant -> local convergence ~conditioning-independent (the c5_woodbury_broyden correction: the
"init doesn't help" null was an Anderson artifact). So the PREDICTION: under Broyden the causal-face metering
should CLEAN UP -- Spearman(n_warm, ||dz||) rises toward the bidir number and the negative partial -> ~0 --
because Broyden takes the slow (carry) mode in stride, so cost tracks geometric movement, not which modes moved.
If so, "metering is a bidirectional property" is the next Anderson artifact; if the causal face stays weak under
Broyden too, the face asymmetry is real (about the operator, not the solver).

Controlled: identical edits (same seeds as c2m_metering), ||dz|| from a solver-independent Newton-polished
solve, n_warm/n_cold counted under BOTH solvers by the SAME crossing-of-TOL_COUNT protocol. Only the solver varies.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.c2m_metering_broyden
"""
import glob
import os

import numpy as np
import torch
from torchdeq import get_deq

import experiments.sliding_window_reach as sw
from experiments.c2_bidir import load_ckpt, TOL_COUNT
from experiments.c2_edit_locality import make_ff, apply_edit
from experiments.c2d_directional import dense_resolvent
from experiments.c2m_metering import make_ff_dh, spearman, partial_pearson

CKPT_DIR = "checkpoints"
N_SEQS = 3
REAL_PER_MODE = 2
EPS_SET = [0.05, 0.2, 1.0, 3.0]
MIN_GAP = 16
PATTERNS = ["curr*.pt", "bidir1*.pt", "bidir2*.pt", "bidir4*.pt"]   # both faces; skip np/qv
sw.H, sw.dh = 4, sw.d // 4


def make_deq(solver):
    return get_deq(f_solver=solver, f_max_iter=80, f_tol=1e-4, ift=True, b_solver="anderson", b_max_iter=1)


def count_solve(deq, ff, z0, polish=False):
    """f-evals under `deq` to the first crossing of rel-resid < TOL_COUNT (solver-agnostic cost, same as
    c2_bidir.counted_solve). If polish, Newton-refine to measurement grade for a solver-independent z."""
    rec = {"n": 0, "k": None}

    def ffc(z):
        out = ff(z)
        rec["n"] += 1
        if rec["k"] is None and (out - z).norm().item() < TOL_COUNT * (z.norm().item() + 1e-9):
            rec["k"] = rec["n"]
        return out
    with torch.no_grad():
        try:
            z = deq(ffc, z0.clone())[0][-1]
        except Exception:
            z = z0.clone()
    if polish:
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


def face_report(recs, face, solver_key):
    """Aggregate metering stats for one (face, solver) over all its records."""
    sel = [r for r in recs if r["face"] == face]
    if len(sel) < 5:
        return None
    dz = np.array([r["dz"] for r in sel])
    dh = np.array([r["dh"] for r in sel])
    nw = np.array([r[solver_key + "_nw"] for r in sel])
    nc = np.array([r[solver_key + "_nc"] for r in sel])
    use = dz > 1e-3
    return dict(
        n=int(use.sum()),
        sp_law=spearman(nw[use], dz[use]),
        sp_dh=spearman(nw[use], dh[use]),
        partial=partial_pearson(nw[use], np.log10(dh[use] + 1e-12), np.log10(dz[use])),
        sp_cold=spearman(nc[use], dz[use]),
        nc_mean=float(nc[use].mean()),
    )


def main():
    ckpts = []
    for pat in PATTERNS:
        ckpts += sorted(glob.glob(os.path.join(CKPT_DIR, pat)))
    ckpts = [p for p in dict.fromkeys(ckpts) if "np" not in os.path.basename(p)
             and "qv" not in os.path.basename(p)]
    anderson, broyden = make_deq("anderson"), make_deq("broyden")
    print(f"device={sw.DEV}  C2m under BOTH solvers: does causal-face metering clean up under Broyden?\n"
          f"  identical edits; ||dz|| Newton-polished (solver-free); n_warm/n_cold counted to TOL_COUNT={TOL_COUNT:g}"
          f" for Anderson AND Broyden.\n", flush=True)
    all_recs = []
    per_ckpt = []
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
            perts = []
            for mode in ["filler", "irrelevant", "relevant"]:
                for _ in range(REAL_PER_MODE):
                    out = apply_edit(toks, gen, mode)
                    if out[0] is None:
                        continue
                    with torch.no_grad():
                        dh = (m.h0(out[0]) - m.h0(toks)).detach()
                    perts.append((mode, out[0], dh))
            p_src = 1
            G = R[:, p_src * sw.d:(p_src + 1) * sw.d]
            _, _, Vh = torch.linalg.svd(G, full_matrices=False)
            v_top, v_bot = Vh[0], Vh[-1]
            for eps in EPS_SET:
                for tag, v in (("syn-carry", v_top), ("syn-trans", v_bot)):
                    dh = torch.zeros(1, L, sw.d, device=sw.DEV)
                    dh[0, p_src] = (eps * v).float()
                    perts.append((tag, None, dh))
            for label, toks2, dh in perts:
                ff_new = make_ff(m, toks2)[0] if toks2 is not None else make_ff_dh(m, toks, dh)
                z_ref = torch.zeros_like(z0)
                # measurement-grade z_new (solver-independent) + Broyden warm cost
                z_new, bro_nw = count_solve(broyden, ff_new, z0.clone(), polish=True)
                _, and_nw = count_solve(anderson, ff_new, z0.clone())
                _, bro_nc = count_solve(broyden, ff_new, z_ref)
                _, and_nc = count_solve(anderson, ff_new, z_ref)
                dz = (z_new - z0).norm().item()
                pred = (R @ dh.reshape(-1).double()).norm().item()
                recs.append(dict(label=label, face=face, ckpt=name, dh=float(dh.reshape(-1).norm()),
                                 dz=dz, pred=pred, and_nw=and_nw, and_nc=and_nc,
                                 bro_nw=bro_nw, bro_nc=bro_nc))
        all_recs += recs
        # per-ckpt Anderson-vs-Broyden metering law
        a = face_report(recs, face, "and")
        b = face_report(recs, face, "bro")
        if a and b:
            per_ckpt.append((name, face, ck["sigma_min"], a, b))
            print(f"[{name}] face={face} gap={gap} smin={ck['sigma_min']:.3f} (n={a['n']})\n"
                  f"    ANDERSON: Spearman(n_warm,||dz||)={a['sp_law']:+.2f}  partial(n,||dh|| | ||dz||)={a['partial']:+.2f}"
                  f"  cold_flat_Sp={a['sp_cold']:+.2f}\n"
                  f"    BROYDEN : Spearman(n_warm,||dz||)={b['sp_law']:+.2f}  partial(n,||dh|| | ||dz||)={b['partial']:+.2f}"
                  f"  cold_flat_Sp={b['sp_cold']:+.2f}\n", flush=True)

    print("=" * 78)
    print("AGGREGATE BY FACE (the headline — pooled over ckpts for statistical power):", flush=True)
    for face in ["causal", "bidir"]:
        a = face_report(all_recs, face, "and")
        b = face_report(all_recs, face, "bro")
        if not (a and b):
            continue
        print(f"\n  {face.upper()} face (n={a['n']}):")
        print(f"    Anderson: Spearman(n_warm,||dz||)={a['sp_law']:+.2f}  partial={a['partial']:+.2f}  "
              f"cold_Sp={a['sp_cold']:+.2f} (nc~{a['nc_mean']:.0f})", flush=True)
        print(f"    Broyden : Spearman(n_warm,||dz||)={b['sp_law']:+.2f}  partial={b['partial']:+.2f}  "
              f"cold_Sp={b['sp_cold']:+.2f} (nc~{b['nc_mean']:.0f})", flush=True)

    np.savez(os.path.join(CKPT_DIR, "c2m_broyden_records.npz"),
             **{k: np.array([r[k] for r in all_recs]) for k in
                ("label", "face", "ckpt", "dh", "dz", "pred", "and_nw", "and_nc", "bro_nw", "bro_nc")})
    print(f"\n(records -> {CKPT_DIR}/c2m_broyden_records.npz)", flush=True)
    print("\nREAD: if the CAUSAL face's Spearman(n_warm,||dz||) rises toward the bidir number AND its negative\n"
          "partial -> ~0 under Broyden, then 'metering is a bidirectional property' was an ANDERSON artifact\n"
          "(the face asymmetry was the spectral gap). If causal stays weak/mode-confounded under Broyden too,\n"
          "the asymmetry is a real property of the (non-normal) operator, solver-independent.", flush=True)


if __name__ == "__main__":
    main()
