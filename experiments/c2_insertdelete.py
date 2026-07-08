"""C2-INSERT/DELETE — the structural-edit shadow, and the aligned-frame (relative-PE) reduction.

Substitution (C2/C2-bidir) keeps the sequence LENGTH fixed. The vast majority of real edits — insert a
token, delete a token, splice a span — change the length and RE-INDEX every downstream position. Under an
ABSOLUTE positional embedding that re-indexing is itself a dense edit: every shifted token's h0 changes, so
even a SEMANTICALLY NULL filler insert casts a shadow over the whole downstream (causal) / whole sequence
(bidir). The escape is a PURE-RELATIVE substrate (NO_POSW): a global shift is invisible to relative attention
except at the cut, so insert/delete reduces to a width-w CONTENT substitution at the cut -- the 'aligned-frame'
reduction, whose viability the posw ablation set up.

MONEY CONTRAST (a null filler insert in the filler region):
  - absolute PE (curr = causal+abs, bidir = bidir+abs): far/near HIGH -> DENSE positional shadow.
  - relative PE (bidirnp = bidir+rel):               far/near LOW  -> shadow COLLAPSES to the cut,
    and matches the substitution-at-cut profile (the reduction is real).
Plus: (a) delete is symmetric; (b) the far-field response stays LOW-RANK (the carry-subspace 'highway');
(c) an aligned-frame WARM START (copy old state, splice a fresh slot at the cut) re-solves cheaply on the
relative substrate but not the absolute one -> the maintainability claim for structural edits.

Aligned frame: insert at c maps old i<c <-> new i, old i>=c <-> new i+1 (inserted slot c is new, dropped
from the diff); delete at c maps old i<c <-> new i, old i>c <-> new i-1 (removed slot c dropped).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.c2_insertdelete
"""
import glob
import os

import numpy as np
import torch

import experiments.sliding_window_reach as sw
import experiments.c2_bidir as cb
from experiments.c2_edit_locality import make_ff

sw.H, sw.dh = 4, sw.d // 4
CKPT_DIR = "checkpoints"
N_SEQS = 8


def load(path):
    m, ck = cb.load_ckpt(path)
    sw.NO_POSW = ck.get("no_posw", False)          # cb.load_ckpt does NOT restore this -- critical for bidirnp
    return m, ck


def solve(m, toks, z0=None):
    ff, _ = make_ff(m, toks)
    if z0 is None:
        z0 = torch.zeros(toks.shape[0], toks.shape[1], sw.d, device=sw.DEV)
    z, it = cb.counted_solve(m, ff, z0)
    return z, it


def filler_val(gen):
    return sw.NKEY + sw.NVAL + int(torch.randint(sw.NFILL, (1,), generator=gen).item())


def insert_tok(toks, c, val):
    new = torch.full((toks.shape[0], 1), val, dtype=toks.dtype, device=toks.device)
    return torch.cat([toks[:, :c], new, toks[:, c:]], dim=1)


def delete_tok(toks, c):
    return torch.cat([toks[:, :c], toks[:, c + 1:]], dim=1)


def aligned_dz(z_old, z_new, c, op):
    """Per old-position response in the aligned frame. op in {insert, delete, subst}. Returns
    (signed dist from cut, |dz| per aligned position, the far-field dz vectors for the rank check)."""
    L = z_old.shape[1]
    if op == "insert":                                    # drop the new inserted slot c
        an = torch.cat([z_new[:, :c], z_new[:, c + 1:]], dim=1)
        keep = np.arange(L)
    elif op == "delete":                                  # drop the removed old slot c
        keepidx = [i for i in range(L) if i != c]
        an = z_new
        z_old = z_old[:, keepidx]
        keep = np.array(keepidx)
    else:                                                 # substitution: same length
        an = z_new
        keep = np.arange(L)
    diff = (an - z_old)[0]                                 # (n_kept, d)
    dz = diff.norm(dim=-1).cpu().numpy()
    dist = keep - c
    far = diff[np.abs(dist) > sw.W]                        # far-field vectors for the SVD rank
    return dist, dz, far


def summarize(dist, dz, pos_scale):
    """Normalize the response by the per-position STATE scale (not the near-field, which an insert's local
    disturbance inflates) -> far_rel = downstream far-field |dz| as a fraction of a token's state norm; this
    is comparable across insert (huge local kick) and substitution (tiny). dense = fraction of downstream-far
    positions responding above 10% of the state scale (absolute threshold). up_rel = upstream far-field."""
    absd = np.abs(dist)
    farmask = (dist > 2 * sw.W)                             # downstream far field
    upmask = (dist < -2 * sw.W)
    far_rel = dz[farmask].mean() / pos_scale if farmask.any() else 0.0
    up_rel = dz[upmask].mean() / pos_scale if upmask.any() else 0.0
    dense = float(np.mean(dz[dist > sw.W] > 0.1 * pos_scale)) if (dist > sw.W).any() else 0.0
    return dict(far_rel=far_rel, up_rel=up_rel, dense=dense)


def eff_rank(far_vecs, thresh=0.1):
    if far_vecs.shape[0] < 2:
        return np.nan
    sv = torch.linalg.svdvals(far_vecs.double()).cpu().numpy()
    return int((sv > thresh * sv[0]).sum())


def main():
    fams = [("curr", "causal+abs"), ("bidir", "bidir+abs"), ("bidirnp", "bidir+rel")]
    gaps = [24, 40]
    print(f"device={sw.DEV}  C2-INSERT/DELETE: structural-edit shadow + aligned-frame (relative-PE) reduction\n"
          f"  null filler insert -> far/near HIGH under abs PE (dense positional shadow), LOW under rel PE\n"
          f"  (collapses to the cut, matches substitution). rank = far-field SVD (carry 'highway'). W={sw.W}\n", flush=True)
    for gap in gaps:
        for fam, tag in fams:
            path = os.path.join(CKPT_DIR, f"{fam}{gap:02d}.pt")
            if not os.path.exists(path):
                print(f"[{fam}{gap:02d}] missing"); continue
            m, ck = load(path)
            gen = torch.Generator().manual_seed(11)
            rows = {"insert": [], "subst": [], "delete": []}
            iw_list, ic_list, rank_list = [], [], []
            for _ in range(N_SEQS):
                toks, _, _ = sw.gen_mqar(1, gap, gen)
                L = toks.shape[1]
                n_fill = L - 2 * sw.D_PAIR - sw.NQ
                if n_fill < 4:
                    continue
                c = 2 * sw.D_PAIR + n_fill // 2                # cut in the middle of the filler region
                z_old, _ = solve(m, toks)
                pos_scale = z_old[0].norm(dim=-1).mean().item()   # per-position state norm (the yardstick)

                # INSERT a null filler at the cut
                fv = filler_val(gen)
                t_ins = insert_tok(toks, c, fv)
                z_ins, ic = solve(m, t_ins)                    # cold solve (converged reference)
                z0_warm = torch.cat([z_old[:, :c], torch.zeros(1, 1, sw.d, device=sw.DEV), z_old[:, c:]], dim=1)
                _, iw = solve(m, t_ins, z0_warm)               # aligned-frame warm start
                d, dz, far = aligned_dz(z_old, z_ins, c, "insert")
                rows["insert"].append(summarize(d, dz, pos_scale)); iw_list.append(iw); ic_list.append(ic)
                rank_list.append(eff_rank(far))

                # SUBSTITUTION at the cut (same length) -- the aligned-frame target
                t_sub = toks.clone(); t_sub[0, c] = filler_val(gen)
                z_sub, _ = solve(m, t_sub)
                d2, dz2, _ = aligned_dz(z_old, z_sub, c, "subst")
                rows["subst"].append(summarize(d2, dz2, pos_scale))

                # DELETE the token at the cut
                t_del = delete_tok(toks, c)
                z_del, _ = solve(m, t_del)
                d3, dz3, _ = aligned_dz(z_old, z_del, c, "delete")
                rows["delete"].append(summarize(d3, dz3, pos_scale))

            if not rows["insert"]:
                print(f"[{fam}{gap:02d}] no usable seqs"); continue
            agg = {k: {kk: np.nanmean([r[kk] for r in v]) for kk in v[0]} for k, v in rows.items()}
            print(f"[{fam}{gap:02d}] {tag}  (n={len(rows['insert'])})", flush=True)
            for op in ["insert", "subst", "delete"]:
                a = agg[op]
                print(f"    {op:>7}: far_rel={a['far_rel']:.3f} (frac of state norm)  downstream-dense={a['dense']:.2f}  "
                      f"up_rel={a['up_rel']:.3f}", flush=True)
            # the positional-shadow factor: how much MORE far-field an insert casts than a substitution
            ratio = agg["insert"]["far_rel"] / (agg["subst"]["far_rel"] + 1e-9)
            print(f"    POSITIONAL SHADOW: insert far_rel / subst far_rel = {ratio:.1f}x  "
                  f"(>>1 = abs-PE re-index shadow; ~1 = aligned-frame reduction holds)", flush=True)
            print(f"    insert warm/cold iters={np.mean(iw_list):.0f}/{np.mean(ic_list):.0f}  "
                  f"far-field rank={np.nanmean(rank_list):.1f} (of d={sw.d})\n", flush=True)
    print("READ: far_rel = downstream far-field |dz| as a fraction of a token's state norm (comparable across\n"
          "insert & subst). POSITIONAL SHADOW ratio = insert/subst far_rel: >>1 means the length re-index adds a\n"
          "dense downstream shadow (abs PE); ~1 means insert reduces to substitution-at-cut (aligned-frame, rel\n"
          "PE). Causal (curr): up_rel~0 (downstream only). Low far-field rank = the carry 'highway'.", flush=True)


if __name__ == "__main__":
    main()
