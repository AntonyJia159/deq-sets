"""C5 reader-set reach — goal-oriented (adjoint / DWR) recompute ball vs the plain forward reach.

THE DEPLOYMENT QUESTION. Tier-1's forward reach circles every position an edit MOVES. But at serving
time we do not care about the whole state — we care about the OUTPUT at reader positions (the tokens a
known prompt will actually read). On the causal face the forward ball is ~the whole suffix (dual to
forgetting: a model that remembers moves everything), so plain reach is close to vacuous. The reader-set
restriction is what could rescue it: recompute only positions that are BOTH reachable from the edit AND
influential on a reader.

TWO REACHES, intersected:
  forward_i  = || [R @ dh]_i ||                         (edit -> position i; R=(I-J)^{-1}, the resolvent)
  adjoint_i  = || H_read . R[reader_rows, i_block] ||_F (position i -> reader LOGITS; the goal-adjoint /
               dual-weighted-residual influence of i on the outputs we read. H_read = the head applied at
               each reader position. This row-block of R is exactly (I-J)^{-T} seen from the readers.)
  forward ball F        = { i : forward_i  > tau * peak }      (tier-1 tight reach)
  reader-set ball F n A  = { i in F : adjoint_i > tau * peak } (forward reach INTERSECT reader influence)

VALIDATION (the money check, nonlinear): recompute ONLY the reader-set ball with a MASKED fixed-point
solve (freeze the complement at z*_old — a Gauss-Seidel/block-coordinate solve; also discharges the owed
"does masked iteration reach the same fixed point" debt), and confirm the READER OUTPUTS match the full
re-solve (logit error small, argmax preserved) while |F n A| << |F|. The unqueried-edit case is the
tell: forward reach is nonzero (must-carry transports the binding) but NO reader reads it, so the
reader-set ball is ~empty and the reader output is correctly unchanged — the reader-set principle made
quantitative.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.c5_readerset_reach
"""
import os

import numpy as np
import torch

import experiments.sliding_window_reach as sw
from experiments.c2_edit_locality import load_checkpoint, counted_solve, make_ff, apply_edit
from experiments.c2d_directional import dense_resolvent

CKPT_DIR = "checkpoints"
GAPS = [8, 16, 24, 40]                    # gap0 has no corridor between edit and readers -> skip
N_SEQS = 3
EDITS_PER_MODE_SEQ = 4
TAU = 1e-2                                 # ball threshold: fraction of the peak of each reach quantity


def goal_adjoint_map(m, R, reader_pos, L, d):
    """H_read . R[reader_rows, :]  -> (n_read*NVAL, L*d): the map from every position's input to the
    reader LOGITS. Column-block i, Frobenius-normed, is adjoint_i (position i's influence on the outputs
    we read). Built once per sequence from the dense resolvent."""
    reader_rows = torch.cat([torch.arange(r * d, (r + 1) * d, device=R.device) for r in reader_pos])
    R_read = R[reader_rows, :]                                    # (n_read*d, L*d)
    Hw = m.head.weight.detach().double()                         # (NVAL, d)
    nr = len(reader_pos)
    R_read = R_read.view(nr, d, L * d)                           # per-reader d-block rows
    G = torch.einsum("vd,rdc->rvc", Hw, R_read).reshape(nr * Hw.shape[0], L * d)   # (nr*NVAL, L*d)
    return G


def masked_solve(ff, z_old, ball_mask, iters=15, tol=1e-9):
    """Recompute ONLY the ball (freeze the complement at z*_old), solving z_ball = f(z)_ball with the
    complement pinned. Plain Picard/Gauss-Seidel DIVERGES here (rho(J) can exceed 1 — noncontractive) and
    a fixed-Jacobian chord diverges on large edits (the attention re-routes, so J at z*_old is a bad
    model), so we take FULL Newton steps on the ball coordinates with backtracking:
    z_ball += alpha (I-J(z))_bb^{-1} (f(z)-z)_ball, alpha halved until the ball residual decreases. This
    is the honest 'block-coordinate recompute reaches the right fixed point' check. An EMPTY ball (a
    truly reader-irrelevant edit) recomputes nothing — z stays z*_old."""
    d = z_old.shape[2]
    ball_pos = torch.nonzero(ball_mask.to(z_old.device), as_tuple=False).flatten().tolist()
    if not ball_pos:
        return z_old.clone(), 0.0
    coords = torch.cat([torch.arange(p * d, (p + 1) * d, device=z_old.device) for p in ball_pos])
    ffl = lambda zv: ff(zv.view(z_old.shape)).reshape(-1)
    z = z_old.reshape(-1).detach().clone()
    res = float("nan")
    for _ in range(iters):
        r_b = (ffl(z) - z).detach()[coords].double()
        rn = r_b.norm().item()
        res = rn / (z[coords].norm().item() + 1e-9)
        if res < tol:
            break
        J = torch.func.jacrev(ffl)(z).detach().double()
        ImJ_bb = torch.eye(coords.numel(), device=z.device, dtype=torch.float64) - J[coords][:, coords]
        step = torch.linalg.solve(ImJ_bb, r_b)
        alpha = 1.0                                                        # backtracking line-search
        for _bt in range(25):
            z_try = z.clone(); z_try[coords] += (alpha * step).to(z.dtype)
            if (ffl(z_try) - z_try).detach()[coords].double().norm().item() < rn:
                break
            alpha *= 0.5
        z = z.clone(); z[coords] += (alpha * step).to(z.dtype)
    return z.view(z_old.shape), res


def reader_logits(m, z, reader_pos):
    return m.head(z[0, reader_pos])                              # (n_read, NVAL)


def main():
    print(f"device={sw.DEV}  C5 reader-set reach: forward reach INTERSECT adjoint(reader) reach.\n"
          f"  Plain forward ball = every position the edit moves (tier-1). Reader-set ball = only those\n"
          f"  that move a READER LOGIT. Validated by a masked re-solve of the reader-set ball alone.\n"
          f"  tau={TAU} of peak; readers = the query positions.\n", flush=True)
    paths = [os.path.join(CKPT_DIR, f"currnp{g:02d}.pt") for g in GAPS]
    paths += [os.path.join(CKPT_DIR, f"bidir{g:02d}.pt") for g in GAPS]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print(f"No currnp*/bidir* checkpoints in {CKPT_DIR}/"); return

    all_rows = []
    for path in paths:
        m, ck = load_checkpoint(path)
        gap = ck["stage_gap"]
        L = 2 * sw.D_PAIR + gap + sw.NQ
        reader_pos = list(range(L - sw.NQ, L))
        gen = torch.Generator().manual_seed(11)
        print(f"[{os.path.basename(path)}] gap={gap} L={L} recall={ck['recall']:.3f} "
              f"sigma_min={ck['sigma_min']:.3f}  readers@{reader_pos}", flush=True)

        modes = ["filler", "irrelevant", "relevant"]
        recs = []
        for si in range(N_SEQS):
            toks = sw.gen_mqar(1, gap, gen)[0]
            z, ff, J, R = dense_resolvent(m, toks)
            G = goal_adjoint_map(m, R, reader_pos, L, sw.d)
            adjoint = G.view(G.shape[0], L, sw.d).norm(dim=(0, 2)).cpu().numpy()   # adjoint_i, per position
            o_base = reader_logits(m, z, reader_pos)
            for mode in modes:
                for _ in range(EDITS_PER_MODE_SEQ):
                    toks2, vpos = apply_edit(toks, gen, mode)
                    if toks2 is None:
                        continue
                    with torch.no_grad():
                        dh_full = (m.h0(toks2) - m.h0(toks)).reshape(-1).double()
                    forward = (R @ dh_full).view(L, sw.d).norm(dim=-1).cpu().numpy()   # forward_i

                    fpk, apk = forward.max(), adjoint.max()
                    causal = not ck.get("bidir", False)
                    dnstream = np.arange(L) >= vpos if causal else np.ones(L, bool)
                    F = (forward > TAU * fpk) & dnstream                              # tier-1 forward ball
                    A = adjoint > TAU * apk                                           # reader-influential
                    RB = F & A                                                        # reader-set ball
                    RB[reader_pos] = RB[reader_pos] | (forward[reader_pos] > TAU * fpk)  # keep moved readers
                    if F.sum() == 0:
                        continue

                    # --- nonlinear validation: full solve vs masked (forward ball) vs masked (reader-set)
                    ff2, _ = make_ff(m, toks2)
                    z_full, _ = counted_solve(m, ff2, z.clone())
                    o_full = reader_logits(m, z_full, reader_pos)
                    z_rb, res_rb = masked_solve(ff2, z, torch.from_numpy(RB))
                    o_rb = reader_logits(m, z_rb, reader_pos)
                    z_fb, res_fb = masked_solve(ff2, z, torch.from_numpy(F))
                    o_fb = reader_logits(m, z_fb, reader_pos)

                    o_change = (o_full - o_base).norm().item()                        # did the edit move readers?
                    err_rb = (o_rb - o_full).norm().item()
                    err_fb = (o_fb - o_full).norm().item()
                    err_none = (o_base - o_full).norm().item()                        # skip everything (stale)
                    argmax_ok = bool((o_rb.argmax(-1) == o_full.argmax(-1)).all().item())
                    recs.append(dict(mode=mode, F=int(F.sum()), RB=int(RB.sum()),
                                     o_change=o_change, err_rb=err_rb, err_fb=err_fb, err_none=err_none,
                                     argmax_ok=argmax_ok, res_rb=res_rb))
        all_rows += [dict(ckpt=os.path.basename(path), **r) for r in recs]

        # -------- per-checkpoint report
        for mode in modes:
            sel = [r for r in recs if r["mode"] == mode]
            if not sel:
                continue
            F_ = np.mean([r["F"] for r in sel]); RB_ = np.mean([r["RB"] for r in sel])
            red = RB_ / max(F_, 1e-9)
            oc = np.mean([r["o_change"] for r in sel])
            e_rb = np.mean([r["err_rb"] for r in sel]); e_none = np.mean([r["err_none"] for r in sel])
            amax = np.mean([r["argmax_ok"] for r in sel])
            print(f"    {mode:>10}: forward ball={F_:4.1f} pos  reader-set ball={RB_:4.1f} pos "
                  f"({red*100:4.0f}% of forward)  | reader-out change={oc:.2e}  err(reader-set "
                  f"recompute)={e_rb:.2e}  err(skip all)={e_none:.2e}  argmax-preserved={amax*100:.0f}%",
                  flush=True)
        print(flush=True)

    # -------- headline aggregate
    print("=" * 96)
    q = [r for r in all_rows if r["mode"] in ("filler", "irrelevant")]     # reader-IRRELEVANT edits
    rel = [r for r in all_rows if r["mode"] == "relevant"]                 # reader-RELEVANT edits
    if q:
        red_q = np.mean([r["RB"] / max(r["F"], 1e-9) for r in q])
        print(f"READER-IRRELEVANT edits (filler/unqueried): reader-set ball = {red_q*100:.0f}% of the "
              f"forward ball on average\n  (must-carry moves the state, but no reader reads it -> "
              f"goal-oriented correctly skips it). n={len(q)}", flush=True)
    if rel:
        red_r = np.mean([r["RB"] / max(r["F"], 1e-9) for r in rel])
        amax_r = np.mean([r["argmax_ok"] for r in rel])
        e_ok = np.mean([r["err_rb"] < 0.05 * max(r["o_change"], 1e-9) for r in rel])
        print(f"READER-RELEVANT edits: reader-set ball = {red_r*100:.0f}% of the forward ball; masked "
              f"reader-set recompute preserves argmax {amax_r*100:.0f}% and stays <5% of the reader-output "
              f"change {e_ok*100:.0f}% of the time. n={len(rel)}", flush=True)
    print("\nREAD: reader-set ball << forward ball with reader OUTPUTS preserved = goal-oriented recompute\n"
          "beats state-space recompute; the gap is the reader-set principle's deployment payoff. Masked\n"
          "solve converging (res_rb small) = block-coordinate recompute reaches the right fixed point.",
          flush=True)


if __name__ == "__main__":
    main()
