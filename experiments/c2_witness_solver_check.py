"""C2-WITNESS-SOLVER-CHECK — is the noncontractive-but-local witness (curr40 rho(J)=8.37; currnp rho>1)
an artifact of the SOLVER, or a property of the fixed point?

WHY: the characterization (sigma_min, rho(J), the far-field resolvent decay = the "envelope holds" witness) is
computed at z* from counted_solve, which is Anderson(150) THEN up-to-3 exact Newton steps -> resid ~1e-7. So z*
is already measurement-grade, NOT a stalled Anderson iterate. This script MEASURES that claim instead of asserting
it: solve each witness cell a SECOND way (Broyden, independent solver) from the same init, Newton-polish both to
the exact fixed point, and confirm the two paths agree on z*, sigma_min, rho(J), and the far-field block-norm
profile of R=(I-J)^{-1}. Agreement => the witness is solver-independent (answers "is rho=8.37 a solver artifact?").
Disagreement => the cell is genuinely multistable and the two solvers found different branches (a real finding).

Reports: raw Broyden residual (did Broyden reach the fixed point on its OWN, no polish?), ||z_A - z_B|| after
polishing both, and side-by-side (sigma_min, rho, max far-field block norm) + the max relative block-norm
disagreement of R across all position pairs (the envelope is derived from these, so if they match the envelope is
identical). Causal curr40 = the poster child; currnp near-singular gaps = the rho>1 witnesses.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.c2_witness_solver_check
"""
import glob
import os

import numpy as np
import torch
from torchdeq import get_deq

import experiments.sliding_window_reach as sw
from experiments.c2_edit_locality import load_checkpoint, make_ff, counted_solve

# stiff / noncontractive-witness cells (by checkpoint basename); others skipped to keep it cheap
WITNESS = {"curr40.pt", "currnp16.pt", "currnp24.pt", "currnp40.pt", "currnpqk40.pt"}
CKPT_DIR = "checkpoints"
sw.H, sw.dh = 4, sw.d // 4


def newton_polish(ff, z, steps=3, tol=1e-9):
    """Up to `steps` exact Newton steps z += (I-J)^{-1}(f(z)-z) (fp64 solve) -> the exact fixed point."""
    for _ in range(steps):
        r = (ff(z) - z).detach()
        if (r.norm() / (z.norm() + 1e-9)).item() < tol:
            break
        zf = z.reshape(-1).detach()
        ffl = lambda zv: ff(zv.view(z.shape)).reshape(-1)
        J = torch.func.jacrev(ffl)(zf)
        ImJ = torch.eye(zf.numel(), device=J.device, dtype=torch.float64) - J.double()
        step = torch.linalg.solve(ImJ, r.reshape(-1).double())
        z = (zf + step.float()).view(z.shape).detach()
    return z


def analyze(ff, z, L, d):
    """(sigma_min, rho(J), R, resid) at z, plus the far-field block-norm decay of R."""
    zf = z.reshape(-1).detach()
    ffl = lambda zv: ff(zv.view(z.shape)).reshape(-1)
    J = torch.func.jacrev(ffl)(zf).detach().double()
    N = zf.numel()
    ImJ = torch.eye(N, dtype=torch.float64, device=J.device) - J
    sv = torch.linalg.svdvals(ImJ)
    rho = torch.linalg.eigvals(J).abs().max().item()
    R = torch.linalg.inv(ImJ)
    resid = ((ff(z) - z).norm() / (z.norm() + 1e-9)).item()
    return dict(sigma_min=sv.min().item(), rho=rho, R=R, resid=resid, zf=zf)


def block_norms(R, L, d):
    """(L,L) matrix of ||R[i-block, j-block]||_2 -- the object the far-field envelope bounds."""
    B = torch.zeros(L, L)
    for i in range(L):
        for j in range(L):
            B[i, j] = torch.linalg.matrix_norm(R[i * d:(i + 1) * d, j * d:(j + 1) * d], ord=2)
    return B


def farfield_max(B, L):
    """max block norm at each backward hop distance i-j>0 (the causal far field)."""
    prof = {}
    for i in range(L):
        for j in range(i):
            dist = i - j
            prof[dist] = max(prof.get(dist, 0.0), B[i, j].item())
    return prof


def main():
    ckpts = [p for p in sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
             if os.path.basename(p) in WITNESS]
    if not ckpts:
        print(f"No witness checkpoints found in {CKPT_DIR}/ (looking for {sorted(WITNESS)})"); return
    print(f"device={sw.DEV}  Witness solver-independence check. Solve each stiff cell TWO ways (Anderson+Newton\n"
          f"  vs Broyden+Newton) from the same init; confirm z*, sigma_min, rho(J), and the far-field R decay\n"
          f"  agree => the noncontractive witness is NOT a solver artifact. {len(ckpts)} cells.\n", flush=True)
    for path in ckpts:
        m, ck = load_checkpoint(path)
        gap = ck["stage_gap"]
        L = 2 * sw.D_PAIR + gap + sw.NQ
        d = sw.d
        gen = torch.Generator().manual_seed(7)
        toks = sw.gen_mqar(1, gap, gen)[0]
        ff, _ = make_ff(m, toks)
        z0 = torch.zeros(1, L, d, device=sw.DEV)

        # path A: the incumbent analysis path (Anderson then exact Newton polish)
        zA, _ = counted_solve(m, ff, z0.clone())

        # path B: independent Broyden, record its RAW residual, then polish to the exact fixed point
        deqB = get_deq(f_solver="broyden", f_max_iter=300, f_tol=1e-9, ift=True, b_solver="anderson", b_max_iter=1)
        with torch.no_grad():
            zB_raw = deqB(ff, z0.clone())[0][-1]
        rawB = ((ff(zB_raw) - zB_raw).norm() / (zB_raw.norm() + 1e-9)).item()
        zB = newton_polish(ff, zB_raw)

        aA = analyze(ff, zA, L, d)
        aB = analyze(ff, zB, L, d)
        dz = ((aA["zf"] - aB["zf"]).norm() / (aA["zf"].norm() + 1e-9)).item()

        BA, BB = block_norms(aA["R"], L, d), block_norms(aB["R"], L, d)
        mask = BA > 1e-6                                   # relative disagreement where the block is non-trivial
        rel_R = ((BA - BB).abs()[mask] / BA[mask]).max().item() if mask.any() else 0.0
        pA, pB = farfield_max(BA, L), farfield_max(BB, L)
        far = sorted(pA)[len(pA) // 2:]                    # the actual far field (larger hop distances)
        ff_max = max(pA[dd] for dd in far) if far else 0.0
        ff_max_B = max(pB[dd] for dd in far) if far else 0.0

        branch = "SAME fixed point" if dz < 1e-4 else f"** DIFFERENT branch (dz={dz:.1e}) — MULTISTABLE **"
        print(f"[{os.path.basename(path)}] gap={gap} L={L} recall={ck['recall']:.3f}", flush=True)
        print(f"    Broyden raw resid (no polish) = {rawB:.1e}   ||zA - zB|| after polish = {dz:.1e}  -> {branch}",
              flush=True)
        print(f"    sigma_min:  A={aA['sigma_min']:.5f}  B={aB['sigma_min']:.5f}   |   "
              f"rho(J):  A={aA['rho']:.3f}  B={aB['rho']:.3f}"
              f"{'   <- NONCONTRACTIVE' if max(aA['rho'], aB['rho']) > 1 else ''}", flush=True)
        agree = "B MATCHES" if rel_R < 1e-3 else "** B DIFFERS **"
        print(f"    far-field max ||R-block||:  A={ff_max:.3e}  B={ff_max_B:.3e}   max rel disagreement of R "
              f"blocks = {rel_R:.1e}  [{agree}]   (envelope is solver-independent if small)\n", flush=True)

    print("READ: ||zA-zB|| ~ 1e-7 => both solvers found the SAME fixed point; sigma_min/rho/far-field agree to\n"
          "the polish floor => the noncontractive witness (rho>1 yet edit-local) is a property of the operator,\n"
          "NOT the solver. Any 'DIFFERENT branch' line = a genuinely multistable cell (report straight).", flush=True)


if __name__ == "__main__":
    main()
