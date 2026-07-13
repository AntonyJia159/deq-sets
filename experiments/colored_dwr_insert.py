"""COLORED-RECALL DWR / reader-set bound for INSERTS -- the keystone, measured (experiment #3).

WHY. The reader-set / dual-weighted-residual (DWR) bound is the paper's "one principled escape" from the
impossible triangle (note #11 §4, the certificate-map highlighted row) -- but it was cited, never run. The
reviewer's obvious question: "you have R cached and known readers -- why not run the adjoint bound?" This runs
it on the sharpest available ground truth. colored-recall's stripe rule is EXACT: inserting a (color,value)
item flips reader q's answer IFF it is q's color AND becomes the newly-selected write (more-recent for
'latest', earlier for 'earliest'); every other insert (shadowed same-color / different-color / filler) leaves
q INVARIANT. So the goal-adjoint influence  w_reader . source  has an exact binary to be checked sound+tight.

OBJECTS (per inserted sequence, at its own tight fixed point; head H = m.head.weight, NVAL x d):
  a_read = H @ R[q_block, :]                    (NVAL x L2 d) -- the reader's goal-adjoint row of R=(I-J)^-1
  prof(i) = || a_read[:, i_block] ||            per-source influence on reader q's LOGITS (the DWR weight)
  r = f2(z_warm) - z_warm                       residual of the aligned-frame warm start (old z, fresh slot@cut)
  DWR estimate   = || a_read @ r ||             first-order predicted reader-logit change (tight, goal-oriented)
  bound_reader   = || a_read ||_2 . || r ||     SOUND reader-restricted bound
  bound_global   = || H ||_2 . || r || / smin   SOUND global bound (the un-restricted a-posteriori resid/smin)
  actual         = || H z2*[q] - H z_warm[q] || the true reader-logit error of the stale warm state

CHECKS: (1) SOUND -- actual <= bound_reader <= bound_global. (2) TIGHT -- bound_reader << bound_global (the
reader-restriction drops the far-field slack). (3) READER-INVARIANCE / discrimination -- prof at the inserted
slot is LARGE iff the insert is in q's stripe (== ground-truth flip); ~0 (invariance certified) for shadowed /
different-color / filler. (4) prof concentrates on q's stripe (selected write + its color token).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.colored_dwr_insert
"""
import os

import numpy as np
import torch

import experiments.sliding_window_reach as sw
from experiments.c2_edit_locality import load_checkpoint, counted_solve, make_ff
from experiments.c2d_directional import dense_resolvent
import experiments.colored_recall as cr

CKPT_DIR = "checkpoints"
N_SEQS = 16
N_ITEMS = 8
FILL = 6


def answer_of(row, n_items, c, mode):
    """Task oracle on a token row: (target class T[selected value], colorpos, valpos) for color c."""
    colors = row[0:2 * n_items:2]
    values = row[1:2 * n_items:2] - cr.VAL0
    idxs = [i for i in range(n_items) if int(colors[i]) == c]
    if not idxs:
        return None
    t = idxs[-1] if mode == "latest" else idxs[0]
    return int(cr._TPERM[int(values[t])]), 2 * t, 2 * t + 1


def make_insert(row, n_items, c, mode, cls, gen, qbase):
    """Build the inserted token row for a class. Returns (row2, n_items2, ins_positions, is_item, v_ins) or
    None. Item inserts add 2 tokens [color, value] at an even item slot (parity preserved); filler adds 1
    token in the filler region. Slot chosen so the ground-truth flip class is exact."""
    idxs = [i for i in range(n_items) if int(row[2 * i]) == c]
    t_sel = idxs[-1] if mode == "latest" else idxs[0]
    v_cur = int(row[2 * t_sel + 1]) - cr.VAL0
    if cls == "filler":
        fpos = qbase - 1 if qbase - 1 >= 2 * n_items else 2 * n_items          # inside filler region
        fv = cr.FILLER0 + int(torch.randint(sw.NFILL, (1,), generator=gen))
        row2 = torch.cat([row[:fpos], torch.tensor([fv], device=row.device), row[fpos:]])
        return row2, n_items, [fpos], False, None
    # item insert: pick color + a value that differs from the current selection
    if cls == "inset":
        c_ins = c
        item_slot = t_sel + 1 if mode == "latest" else t_sel        # after last (latest) / before first (earliest)
    elif cls == "shadow":
        c_ins = c
        item_slot = 0 if mode == "latest" else n_items              # before all (latest) / after all (earliest)
    elif cls == "diffcolor":
        c_ins = (c + 1) % cr.C
        item_slot = t_sel + 1
    else:
        return None
    v_ins = (v_cur + 1 + int(torch.randint(cr.V - 1, (1,), generator=gen))) % cr.V   # != v_cur
    ipos = 2 * item_slot
    new = torch.tensor([c_ins, cr.VAL0 + v_ins], device=row.device)
    row2 = torch.cat([row[:ipos], new, row[ipos:]])
    return row2, n_items + 1, [ipos, ipos + 1], True, v_ins


def insert_zero_slots(z, positions):
    """Aligned-frame warm start: old state z with fresh zero slots spliced at `positions` (ascending)."""
    parts, prev = [], 0
    for p in sorted(positions):
        parts.append(z[:, prev:p])
        parts.append(torch.zeros(z.shape[0], 1, z.shape[2], device=z.device, dtype=z.dtype))
        prev = p
    parts.append(z[:, prev:])
    return torch.cat(parts, dim=1)


def align_map(L2, ins_pos):
    """new-frame position -> base-frame position (aligned frame); -1 at the inserted slots (no base column)."""
    ins = set(ins_pos)
    return [-1 if p in ins else p - sum(1 for q in ins_pos if q < p) for p in range(L2)]


def main():
    print(f"device={sw.DEV}  COLORED-RECALL DWR/reader-set bound for INSERTS (keystone, exp #3)\n"
          f"  goal-adjoint w_reader.source vs the EXACT stripe flip rule; sound (actual<=bound_reader<="
          f"bound_global),\n  tight (reader-restriction << global), reader-invariant (prof@insert fires IFF "
          f"in-stripe).\n  {N_SEQS} seqs, n_items={N_ITEMS}, fill={FILL}.\n", flush=True)
    for mode, fname in [("latest", "colored_recall.pt"), ("earliest", "colored_recall_earliest.pt")]:
        path = os.path.join(CKPT_DIR, fname)
        if not os.path.exists(path):
            print(f"[{mode}] {fname} missing\n"); continue
        m, ck = load_checkpoint(path)
        Hw = m.head.weight.detach().double()
        Hn = torch.linalg.matrix_norm(Hw, ord=2).item()
        gen = torch.Generator().manual_seed(20)
        classes = ["inset", "shadow", "diffcolor", "filler"]
        recs = {cl: [] for cl in classes}
        print(f"===== mode={mode}  recall_by_len={ck.get('lengen')}  ||H||={Hn:.2f} =====", flush=True)
        for _ in range(N_SEQS):
            toks, qmask, targ, deps = cr.gen_colored_recall(1, N_ITEMS, gen, fill=FILL, mode=mode)
            row = toks[0]
            L = row.shape[0]
            qbase = 2 * N_ITEMS + FILL
            d0 = deps[0][0]; c = d0["color"]; qp = d0["query"]
            z, ff, J, R = dense_resolvent(m, toks)
            o_base = (Hw @ z[0, qp].double())
            base_ans = int(o_base.argmax())
            for cl in classes:
                built = make_insert(row, N_ITEMS, c, mode, cl, gen, qbase)
                if built is None:
                    continue
                row2, n2, ins_pos, is_item, v_ins = built
                toks2 = row2.unsqueeze(0)
                shift = len(ins_pos)
                q2 = qp + shift                                        # reader re-indexed (insert is before it)
                gt = answer_of(row2, n2, c, mode)
                if gt is None:
                    continue
                gt_target, gt_cpos, gt_vpos = gt
                gt_flip = (gt_target != base_ans)
                # inserted-seq fixed point + resolvent
                z2, ff2, J2, R2 = dense_resolvent(m, toks2)
                L2 = row2.shape[0]
                o2 = (Hw @ z2[0, q2].double())
                model_ans = int(o2.argmax())
                actual_flip = (model_ans != base_ans)
                model_correct = (model_ans == gt_target)
                # aligned warm start + residual
                z_warm = insert_zero_slots(z, ins_pos)
                r = (ff2(z_warm) - z_warm).reshape(-1).double()
                rn = r.norm().item()
                o_warm = (Hw @ z_warm[0, q2].double())
                actual = (o2 - o_warm).norm().item()
                # reader goal-adjoint row of R2
                qrows = torch.arange(q2 * sw.d, (q2 + 1) * sw.d, device=R2.device)
                a_read = Hw @ R2[qrows, :].double()                   # (NVAL, L2 d)
                a_norm = torch.linalg.matrix_norm(a_read, ord=2).item()
                dwr_est = (a_read @ r).norm().item()
                smin2 = torch.linalg.svdvals(
                    torch.eye(L2 * sw.d, device=J2.device, dtype=torch.float64) - J2).min().item()
                bound_reader = a_norm * rn
                bound_global = Hn * rn / max(smin2, 1e-12)
                prof = a_read.view(Hw.shape[0], L2, sw.d).norm(dim=(0, 2)).cpu().numpy()
                prof_peak = float(prof.max())
                prof_insert = float(prof[ins_pos].max())
                prof_stripe = float(prof[gt_vpos])                    # influence of the selected write
                # CHEAP: reuse the CACHED pre-insert reader adjoint (base R's reader row), aligned to the new
                # indices; only the residual r is fresh (one f-eval, no R2 re-solve). No column at the inserted
                # slot (base R never saw it) -> the insert's effect must route through r, via the cached adjoint.
                aH_base = Hw @ R[qp * sw.d:(qp + 1) * sw.d, :].double()          # (NVAL, L d) -- cached
                aH_cheap = torch.zeros(Hw.shape[0], L2 * sw.d, dtype=torch.float64, device=R.device)
                for p2, pb in enumerate(align_map(L2, ins_pos)):
                    if pb >= 0:
                        aH_cheap[:, p2 * sw.d:(p2 + 1) * sw.d] = aH_base[:, pb * sw.d:(pb + 1) * sw.d]
                dwr_cheap = (aH_cheap @ r).norm().item()
                b_read_cheap = torch.linalg.matrix_norm(aH_cheap, ord=2).item() * rn
                recs[cl].append(dict(
                    gt_flip=gt_flip, actual_flip=actual_flip, correct=model_correct,
                    actual=actual, dwr_est=dwr_est, b_read=bound_reader, b_glob=bound_global,
                    dwr_cheap=dwr_cheap, b_read_cheap=b_read_cheap,
                    sound_cheap=(actual <= 1.001 * b_read_cheap),
                    rn=rn, prof_insert=prof_insert / max(prof_peak, 1e-30),
                    prof_stripe=prof_stripe / max(prof_peak, 1e-30),
                    sound=(actual <= 1.001 * bound_reader), tight=bound_global / max(bound_reader, 1e-30)))

        # -------- report
        print(f"    {'class':>10} | {'gt_flip':>7} {'mdl_flip':>8} {'correct':>7} | "
              f"{'actual':>9} {'dwr_est':>9} {'b_reader':>9} {'b_global':>9} | "
              f"{'sound%':>6} {'tight(g/r)':>10} | {'prof@ins':>8} {'prof@stripe':>11}", flush=True)
        for cl in classes:
            s = recs[cl]
            if not s:
                continue
            gm = lambda k: float(np.exp(np.mean(np.log([max(x[k], 1e-30) for x in s]))))
            f = lambda k: float(np.mean([x[k] for x in s]))
            print(f"    {cl:>10} | {f('gt_flip')*100:6.0f}% {f('actual_flip')*100:7.0f}% "
                  f"{f('correct')*100:6.0f}% | {gm('actual'):9.2e} {gm('dwr_est'):9.2e} "
                  f"{gm('b_read'):9.2e} {gm('b_glob'):9.2e} | {f('sound')*100:5.0f}% "
                  f"{gm('tight'):10.1f} | {f('prof_insert'):8.3f} {f('prof_stripe'):11.3f}  (n={len(s)})",
                  flush=True)
        # discrimination headline
        ins = recs["inset"]
        noflip = [x for cl in ["shadow", "diffcolor", "filler"] for x in recs[cl]]
        if ins and noflip:
            pi_in = np.median([x["prof_insert"] for x in ins])
            pi_no = np.median([x["prof_insert"] for x in noflip])
            sound_all = np.mean([x["sound"] for cl in classes for x in recs[cl]])
            print(f"    -> DISCRIMINATION prof@insert: in-stripe median={pi_in:.3f} vs off-stripe "
                  f"median={pi_no:.3f} (sep {pi_in/max(pi_no,1e-6):.1f}x);  "
                  f"soundness (actual<=bound_reader) {sound_all*100:.0f}% overall", flush=True)
        # -------- CHEAP: cached base-R reader adjoint + fresh residual (no R2 re-solve)
        print(f"    --- CHEAP (cached pre-insert reader adjoint, only residual recomputed) ---", flush=True)
        print(f"    {'class':>10} | {'actual':>9} {'dwr_oracle':>10} {'dwr_cheap':>9} "
              f"{'bR_cheap':>9} | {'sound_cheap%':>12} {'cheap/oracle':>12}", flush=True)
        for cl in classes:
            s = recs[cl]
            if not s:
                continue
            gm = lambda k: float(np.exp(np.mean(np.log([max(x[k], 1e-30) for x in s]))))
            f = lambda k: float(np.mean([x[k] for x in s]))
            print(f"    {cl:>10} | {gm('actual'):9.2e} {gm('dwr_est'):10.2e} {gm('dwr_cheap'):9.2e} "
                  f"{gm('b_read_cheap'):9.2e} | {f('sound_cheap')*100:11.0f}% "
                  f"{gm('dwr_cheap')/max(gm('dwr_est'),1e-30):12.2f}", flush=True)
        if ins and noflip:
            dc_in = np.median([x["dwr_cheap"] for x in ins])
            dc_no = np.median([x["dwr_cheap"] for x in noflip])
            sc_all = np.mean([x["sound_cheap"] for cl in classes for x in recs[cl]])
            print(f"    -> CHEAP discrimination dwr_cheap: in-stripe median={dc_in:.2e} vs off-stripe "
                  f"median={dc_no:.2e} (sep {dc_in/max(dc_no,1e-30):.1f}x);  cheap soundness "
                  f"{sc_all*100:.0f}%\n", flush=True)
    print("READ: prof@insert large IFF gt_flip = the goal-adjoint w_reader.source recovers the exact stripe\n"
          "(reader-invariance certified for off-stripe inserts); bound_reader sound & << bound_global = the\n"
          "reader-restriction is the tightening. This is the DWR keystone, measured.", flush=True)


if __name__ == "__main__":
    main()
