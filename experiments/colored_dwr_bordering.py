"""SCHUR-CORRECTED CHEAP ADJOINT for inserts -- the bordering fix that makes the cached DWR adjoint rigorous
and recovers flip discrimination WITHOUT a re-solve (follow-up to colored_dwr_insert / exp #3, and exp #4).

THE PROBLEM (from exp #3). The NAIVE cheap adjoint reuses the cached PRE-insert reader row of R and has NO
column at the inserted slot -> it is blind to the new item, so flip discrimination collapses (0.9x on recency).

THE FIX (bordering / Woodbury). An insert is a LOCAL operator change confined to a window S around the cut
(pure-relative substrate: far entries unchanged up to re-indexing). Embed the cached base resolvent as R_emb
(scatter to aligned indices; identity at the new slot). Then M' = M_emb + P Delta P^T with Delta supported on
S x S, and Woodbury gives the NEW reader row EXACTLY from cached blocks + one k x k solve (k=|S| d, INDEPENDENT
of sequence length L):
    R'[q,:] = R_emb[q,:] - (R_emb[q,:] P) (I + Delta (P^T R_emb P))^{-1} Delta (P^T R_emb),
    Delta = (J_emb - J')|_{S x S}.
Using J'(z2*) (exact) validates the algebra + MEASURES the bordering support; using J'(z_warm) (one LOCAL
Jacobian, no fixed-point iteration) is the CHEAP predictor.

MEASURED FINDING (2026-07-13): the Woodbury algebra is EXACT (recon 1e-15 when S spans the support), BUT the
insert's operator change is NON-LOCAL at the converged fixed point -- deltaM = J(z2*)-J(z*) has ~45% of its mass
beyond +/-8 positions and vanishes only when S = the whole sequence (state-dependent J: the equilibrium shifts
everywhere the insert reaches, at rate rho(G)). So EXACT bordering needs S ~ L -> there is NO length-independent
rigorous cheap correction (this REFUTES the earlier "bordering makes the cheap bound rigorous" hope). What the
WARM-LOCAL correction DOES buy: it recovers flip DISCRIMINATION on long-range deps (naive 0x -> ~13-20x, ~oracle)
and stays empirically SOUND, but reconstructs the warm operator (not R2; recon 20-990% on big flips) -> an
approximate cheap ANSWER-METERING signal, not an exact/rigorous resolvent. Recency (latest) stays weak (~1x:
nonlinear selection). CHECKS below report recon rel-err (exact-J and warm-J), naive/schur/oracle discrimination,
soundness, and a wall-time proxy (no clean L-independence since the exact support is global).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.colored_dwr_bordering
"""
import os
import time

import numpy as np
import torch

import experiments.sliding_window_reach as sw
from experiments.c2_edit_locality import load_checkpoint, counted_solve, make_ff
from experiments.c2d_directional import dense_resolvent
from experiments.colored_dwr_insert import make_insert, answer_of, insert_zero_slots, align_map
import experiments.colored_recall as cr

CKPT_DIR = "checkpoints"
N_SEQS = 8
N_ITEMS = 10
MARGIN = 22          # S = new positions within MARGIN of the cut (>= 2W to cover the Jacobian band; W=10)


def blk(idxs, d, device):
    return torch.cat([torch.arange(i * d, (i + 1) * d, device=device) for i in idxs])


def build_emb(Base, L, L2, ins_pos, d, fill_ins_identity):
    """Scatter an (L d, L d) base block-matrix to the aligned (L2 d, L2 d) frame; inserted slots get an
    identity block (fill_ins_identity=True, for R_emb=M_emb^-1) or zero (False, for J_emb)."""
    amap = align_map(L2, ins_pos)                      # new -> base (-1 at inserted)
    old_new = [p for p in range(L2) if amap[p] >= 0]
    old_base = [amap[p] for p in old_new]
    dev = Base.device
    E = torch.zeros(L2 * d, L2 * d, dtype=Base.dtype, device=dev)
    rn, rb = blk(old_new, d, dev), blk(old_base, d, dev)
    E[rn.unsqueeze(1), rn.unsqueeze(0)] = Base[rb.unsqueeze(1), rb.unsqueeze(0)]
    if fill_ins_identity:
        for p in ins_pos:
            ix = blk([p], d, dev)
            E[ix.unsqueeze(1), ix.unsqueeze(0)] = torch.eye(d, dtype=Base.dtype, device=dev)
    return E


def schur_reader_row(R_emb, J_emb, Jloc_SpSp, q2, Sp, d):
    """Woodbury correction of the reader row. Delta = (J_emb - J')|_{SxS}; returns R'[q2 block, :] (d, L2 d)."""
    dev = R_emb.device
    Pc = blk(Sp, d, dev)
    qb = blk([q2], d, dev)
    Delta = J_emb[Pc.unsqueeze(1), Pc.unsqueeze(0)] - Jloc_SpSp        # (k, k)
    RSS = R_emb[Pc.unsqueeze(1), Pc.unsqueeze(0)]                      # (k, k)
    RS = R_emb[Pc, :]                                                  # (k, L2 d)
    g = R_emb[qb][:, Pc]                                              # (d, k)
    k = Pc.numel()
    inner = torch.eye(k, dtype=R_emb.dtype, device=dev) + Delta @ RSS  # (k, k)
    corr = g @ torch.linalg.solve(inner, Delta @ RS)                  # (d, L2 d)
    return R_emb[qb] - corr


def main():
    print(f"device={sw.DEV}  SCHUR-CORRECTED cheap adjoint (bordering/Woodbury) for inserts.\n"
          f"  naive cheap = cached reader row (blind to insert); schur = + local Woodbury correction (k x k,\n"
          f"  L-independent). Validate vs oracle R2; recover flip discrimination; quantify efficiency.\n"
          f"  N_ITEMS={N_ITEMS} MARGIN={MARGIN} (S = cut +/- MARGIN).\n", flush=True)
    for mode, fname, fills in [("latest", "colored_recall.pt", [16, 28]),
                               ("earliest", "colored_recall_earliest.pt", [16, 28])]:
        path = os.path.join(CKPT_DIR, fname)
        if not os.path.exists(path):
            print(f"[{mode}] {fname} missing\n"); continue
        m, ck = load_checkpoint(path)
        Hw = m.head.weight.detach().double()
        d = sw.d
        classes = ["inset", "shadow", "diffcolor", "filler"]
        print(f"===== mode={mode}  d={d}  W={sw.W} =====", flush=True)
        for FILL in fills:
            gen = torch.Generator().manual_seed(20)
            recs = {cl: [] for cl in classes}
            recon_exact, recon_warm, ks, L2s = [], [], [], []
            t_resolve, t_cheap, t_wood = [], [], []
            for _ in range(N_SEQS):
                toks, qmask, targ, deps = cr.gen_colored_recall(1, N_ITEMS, gen, fill=FILL, mode=mode)
                row = toks[0]; L = row.shape[0]; qbase = 2 * N_ITEMS + FILL
                d0 = deps[0][0]; c = d0["color"]; qp = d0["query"]
                z, ff, J, R = dense_resolvent(m, toks)                 # cached base (fp64 R, J)
                o_base = (Hw @ z[0, qp].double()); base_ans = int(o_base.argmax())
                for cl in classes:
                    built = make_insert(row, N_ITEMS, c, mode, cl, gen, qbase)
                    if built is None:
                        continue
                    row2, n2, ins_pos, is_item, v_ins = built
                    toks2 = row2.unsqueeze(0); L2 = row2.shape[0]; shift = len(ins_pos)
                    q2 = qp + shift
                    gt = answer_of(row2, n2, c, mode)
                    if gt is None:
                        continue
                    gt_target, _, gt_vpos = gt
                    gt_flip = (gt_target != base_ans)
                    # ---- ORACLE (the expensive path): re-solve fixed point + resolvent
                    if sw.DEV == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    z2, ff2, J2, R2 = dense_resolvent(m, toks2)
                    if sw.DEV == "cuda":
                        torch.cuda.synchronize()
                    t_resolve.append(time.perf_counter() - t0)
                    qb = blk([q2], d, R2.device)
                    a_oracle = Hw @ R2[qb, :]
                    # ---- warm start + residual (shared)
                    z_warm = insert_zero_slots(z, ins_pos)
                    r = (ff2(z_warm) - z_warm).reshape(-1).double(); rn = r.norm().item()
                    o_warm = (Hw @ z_warm[0, q2].double()); o2 = (Hw @ z2[0, q2].double())
                    actual = (o2 - o_warm).norm().item()
                    # ---- embeddings from cached base
                    R_emb = build_emb(R, L, L2, ins_pos, d, fill_ins_identity=True)
                    J_emb = build_emb(J, L, L2, ins_pos, d, fill_ins_identity=False)
                    cut = min(ins_pos)
                    Sp = [p for p in range(L2) if min(abs(p - ip) for ip in ins_pos) <= MARGIN]
                    Pc = blk(Sp, d, R_emb.device); k = Pc.numel()
                    a_naive = Hw @ R_emb[qb, :]                        # cached reader row, blind to insert
                    # exact-J Woodbury (validate algebra + support): Delta from J2
                    Jexact_SpSp = J2[Pc.unsqueeze(1), Pc.unsqueeze(0)]
                    row_exact = schur_reader_row(R_emb, J_emb, Jexact_SpSp, q2, Sp, d)
                    recon_exact.append((row_exact - R2[qb, :]).norm().item() /
                                       (R2[qb, :].norm().item() + 1e-30))
                    # cheap: local warm Jacobian, only the S output rows (k backward passes)
                    ffl2 = lambda zv: ff2(zv.view(z_warm.shape)).reshape(-1)
                    if sw.DEV == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    Jw_rows = torch.func.jacrev(lambda zv: ffl2(zv)[Pc])(
                        z_warm.reshape(-1).float()).double()          # (k, L2 d)
                    Jw_SpSp = Jw_rows[:, Pc]
                    if sw.DEV == "cuda":
                        torch.cuda.synchronize()
                    t_jac = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    row_schur = schur_reader_row(R_emb, J_emb, Jw_SpSp, q2, Sp, d)   # the k x k Woodbury LA
                    if sw.DEV == "cuda":
                        torch.cuda.synchronize()
                    t_wood.append(time.perf_counter() - t0)
                    t_cheap.append(t_jac)
                    a_schur = Hw @ row_schur
                    recon_warm.append((row_schur - R2[qb, :]).norm().item() /
                                      (R2[qb, :].norm().item() + 1e-30))
                    ks.append(k); L2s.append(L2 * d)

                    def metrics(a):
                        a = a.detach()
                        pr = a.view(Hw.shape[0], L2, d).norm(dim=(0, 2)).cpu().numpy()
                        pk = float(pr.max())
                        return dict(dwr=(a @ r).norm().item(), bound=torch.linalg.matrix_norm(a, ord=2).item() * rn,
                                    pins=float(pr[ins_pos].max()) / max(pk, 1e-30))
                    recs[cl].append(dict(gt_flip=gt_flip, actual=actual,
                                         naive=metrics(a_naive), schur=metrics(a_schur), oracle=metrics(a_oracle)))

            # ---- report (per fill)
            print(f"  --- FILL={FILL}  L2 d={int(np.mean(L2s))}  k=|S| d={int(np.mean(ks))} "
                  f"(k/L2d={np.mean(ks)/np.mean(L2s):.2f})  recon rel-err: exact-J={np.mean(recon_exact):.1e} "
                  f"warm-J={np.mean(recon_warm):.1e} ---", flush=True)
            for cl in classes:
                s = recs[cl]
                if not s:
                    continue
                gm = lambda sel, kk: float(np.exp(np.mean(np.log([max(x[sel][kk], 1e-30) for x in s]))))
                snd = lambda sel: np.mean([x['actual'] <= 1.001 * x[sel]['bound'] for x in s])
                print(f"    {cl:>10} gt_flip={np.mean([x['gt_flip'] for x in s])*100:3.0f}% | prof@ins "
                      f"naive={gm('naive','pins'):.3f} schur={gm('schur','pins'):.3f} oracle={gm('oracle','pins'):.3f}"
                      f" | dwr naive={gm('naive','dwr'):.2e} schur={gm('schur','dwr'):.2e} oracle={gm('oracle','dwr'):.2e}"
                      f" | sound schur={snd('schur')*100:.0f}%", flush=True)
            ins = recs["inset"]; noflip = [x for cl in ["shadow", "diffcolor", "filler"] for x in recs[cl]]
            if ins and noflip:
                def sep(sel):
                    a = np.median([x[sel]['pins'] for x in ins]); b = np.median([x[sel]['pins'] for x in noflip])
                    return a / max(b, 1e-6)
                tr, tj, tw = np.mean(t_resolve), np.mean(t_cheap), np.mean(t_wood)
                Lr = np.mean(L2s) / np.mean(ks)
                print(f"    -> flip-discrimination (in/off prof@ins): naive={sep('naive'):.1f}x  "
                      f"schur={sep('schur'):.1f}x  oracle={sep('oracle'):.1f}x", flush=True)
                print(f"    -> EFFICIENCY: re-solve={tr*1e3:.1f}ms  Woodbury-LA(kxk)={tw*1e3:.2f}ms  "
                      f"localJac(autograd,O(k L2d))={tj*1e3:.1f}ms.  k/L2d={1/Lr:.2f} -> at toy L~2W support "
                      f"fills the seq (no win); asymptotic (W fixed, L grows): k=O(Wd) const, inverse-cost "
                      f"(L2d/k)^3={Lr**3:.0f}x, re-solve iters skipped entirely\n", flush=True)
    print("READ: exact-J recon is NOT ~1e-10 unless S spans the whole seq -> the insert's operator change is\n"
          "NON-LOCAL (state-dependent J; equilibrium shifts globally) -> no length-independent exact bordering.\n"
          "But warm-local schur prof@ins ~ oracle on long-range (earliest) = it RECOVERS flip discrimination the\n"
          "naive cached adjoint lost, still sound -> a cheap APPROXIMATE answer-metering signal, not a rigorous\n"
          "cheap resolvent. Recency (latest) stays weak (nonlinear selection).", flush=True)


if __name__ == "__main__":
    main()
