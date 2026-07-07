"""C2t — deferred billing / the eager-vs-lazy iteration ledger.

THE TRICK: retargeting a query token (substituting WHICH KEY it asks about) IS the "reader arrives"
event — just another substitution edit. So the lazy-vs-eager split is pure edit SEQUENCING:

  LAZY  path: edit an unqueried value  ->  it_write_lazy      (bill small if the relay can defer)
              then retarget a query to it -> it_trigger        (the deferred bill arrives)
  EAGER path: retarget the query first  -> it_retarget_pre    (pure reader-arrival baseline)
              then edit the (now queried) value -> it_write_eager  (bill lands at write)

Both paths end at the SAME token sequence -> final fixed points must agree (path-independence check;
disagreement = branch divergence, report not hide). The LEDGER prediction: total(lazy) ~ total(eager),
only the write/trigger SPLIT moves. Where the split lands is the evaluation strategy:

  causal (curr)      : EAGER by necessity — must-carry pre-pays transport at write; trigger ~ free
                       (a query retarget only re-solves the suffix, which is the query itself)
  bidir readonly     : ALSO EAGER — queries invisible to context (the C2-bidir correction), the relay
    (bidir)            carries all bindings tailward at write regardless of readers
  bidir query-visible: LAZY *if* query-awareness formed — cheap write (binding stored locally),
    (bidirqv)          expensive gap-dependent trigger (transport materializes at reader arrival);
                       ALSO re-test must-carry properly here: dz far-field of the write step should
                       drop vs the readonly substrate if the relay is actually query-aware

Per-step response summaries locate WHERE transport materializes: dz at the retargeted query position
(and far-field norm) for the write step vs the trigger step.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.c2t_deferred
"""
import glob
import os

import numpy as np
import torch

import experiments.sliding_window_reach as sw
from experiments.c2_bidir import load_ckpt, counted_solve
from experiments.c2_edit_locality import make_ff, residual_of

CKPT_DIR = "checkpoints"
N_SEQS = 3
N_REP = 3                    # (slot, query, value) draws per seq
MIN_GAP = 16
sw.H, sw.dh = 4, sw.d // 4

PATTERNS = ["curr*.pt", "bidir0*.pt", "bidir1*.pt", "bidir2*.pt", "bidir4*.pt", "bidirqv*.pt"]


def pick_protocol(toks, gen):
    """Choose an UNQUERIED key slot b, a query index qi, and a new value token. None if impossible."""
    L = toks.shape[1]
    queried = set(toks[0, L - sw.NQ:].tolist())
    slots = [b for b in range(sw.D_PAIR) if toks[0, 2 * b].item() not in queried]
    if not slots:
        return None
    b = slots[torch.randint(len(slots), (1,), generator=gen).item()]
    qi = torch.randint(sw.NQ, (1,), generator=gen).item()
    old_val = toks[0, 2 * b + 1].item()
    new_val = sw.NKEY + torch.randint(sw.NVAL, (1,), generator=gen).item()
    while new_val == old_val:
        new_val = sw.NKEY + torch.randint(sw.NVAL, (1,), generator=gen).item()
    return b, qi, new_val


def substitute(toks, pos, token):
    t = toks.clone()
    t[0, pos] = token
    return t


def solve_step(m, toks, z_init):
    """Warm counted solve on toks from z_init; returns (z, iters, resid, dz-profile vs z_init)."""
    ff, _ = make_ff(m, toks)
    z, it = counted_solve(m, ff, z_init.clone())
    dz = (z - z_init)[0].norm(dim=-1).cpu().numpy()
    return z, it, residual_of(ff, z), dz


def main():
    ckpts = []
    for pat in PATTERNS:
        ckpts += sorted(glob.glob(os.path.join(CKPT_DIR, pat)))
    ckpts = [p for p in dict.fromkeys(ckpts)]
    print(f"device={sw.DEV}  C2t deferred-billing ledger; iters = evals to rel-resid 1e-3; "
          f"{N_SEQS} seqs x {N_REP} draws; gap >= {MIN_GAP}\n"
          f"  prediction: curr + bidir(readonly) = EAGER (write pays), bidirqv = LAZY (trigger pays), "
          f"totals conserved\n", flush=True)
    rows = []
    for path in ckpts:
        m, ck = load_ckpt(path)
        gap = ck["stage_gap"]
        if gap < MIN_GAP:
            continue
        gen = torch.Generator().manual_seed(7)
        seqs = [sw.gen_mqar(1, gap, gen)[0] for _ in range(N_SEQS)]
        name = os.path.basename(path)
        rec = dict(wl=[], tr=[], rp=[], we=[], agree=[], dzq_write=[], dzq_trig=[])
        for toks in seqs:
            ff0, _ = make_ff(m, toks)
            z0, _ = counted_solve(m, ff0, torch.zeros(1, toks.shape[1], sw.d, device=sw.DEV))
            L = toks.shape[1]
            for _ in range(N_REP):
                pr = pick_protocol(toks, gen)
                if pr is None:
                    continue
                b, qi, new_val = pr
                qpos = L - sw.NQ + qi
                kb = toks[0, 2 * b].item()
                t_edit = substitute(toks, 2 * b + 1, new_val)
                t_both = substitute(t_edit, qpos, kb)
                t_ret = substitute(toks, qpos, kb)

                # LAZY: edit (reader absent) -> trigger (reader arrives)
                z1, it_wl, r1, dz1 = solve_step(m, t_edit, z0)
                z2, it_tr, r2, dz2 = solve_step(m, t_both, z1)
                # EAGER: reader first -> edit (reader present)
                z1e, it_rp, r3, _ = solve_step(m, t_ret, z0)
                z2e, it_we, r4, dz2e = solve_step(m, t_both, z1e)

                if max(r1, r2, r3, r4) > 1e-3:
                    continue
                rec["wl"].append(it_wl); rec["tr"].append(it_tr)
                rec["rp"].append(it_rp); rec["we"].append(it_we)
                rec["agree"].append((z2 - z2e).norm().item() / (z2e.norm().item() + 1e-9))
                rec["dzq_write"].append(float(dz1[qpos]))   # response AT the reader position, write step
                rec["dzq_trig"].append(float(dz2[qpos]))    # ... vs trigger step (where transport lands)
        if not rec["wl"]:
            print(f"[{name}] no usable draws\n", flush=True); continue
        wl, tr = np.mean(rec["wl"]), np.mean(rec["tr"])
        rp, we = np.mean(rec["rp"]), np.mean(rec["we"])
        agree = np.median(rec["agree"])
        print(f"[{name}] gap={gap} recall={ck['recall']:.3f} (n={len(rec['wl'])})\n"
              f"    LAZY : write={wl:5.1f}  trigger={tr:5.1f}   total={wl + tr:5.1f}\n"
              f"    EAGER: retarget={rp:5.1f}  write={we:5.1f}   total={rp + we:5.1f}\n"
              f"    split: lazy pays {tr / (wl + tr + 1e-9):.0%} at trigger vs eager {we / (rp + we + 1e-9):.0%} at write"
              f"   | final-state agree={agree:.1e}\n"
              f"    dz@reader-position: write-step={np.mean(rec['dzq_write']):.2e}  "
              f"trigger-step={np.mean(rec['dzq_trig']):.2e}"
              f"   (lazy predicts transport materializes at trigger)\n", flush=True)
        rows.append((name, gap, wl, tr, rp, we, agree))
    np.savez(os.path.join(CKPT_DIR, "c2t_ledger.npz"),
             rows=np.array([(r[0],) + tuple(map(float, r[1:])) for r in rows], dtype=object))
    print("READ: EAGER substrates -> write-heavy in both paths, trigger ~ free, dz@reader appears at the\n"
          "write step. LAZY substrate -> lazy path pays at trigger (dz@reader appears at trigger step),\n"
          "totals ~ conserved. final-state agree >> solver tol = branch divergence (report, don't hide).",
          flush=True)


if __name__ == "__main__":
    main()
