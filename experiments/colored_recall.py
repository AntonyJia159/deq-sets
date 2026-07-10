"""COLORED REGISTER RECALL -- a select-then-transform task with striped edit-responses (causal).

Items are (color, value) pairs; COLORS REPEAT (that is what makes stripes). A reader asks "latest value of
color c"; the answer is T(that value), where T is a FIXED PERMUTATION of the value set (the transform). So
the task is: SELECT the most-recent same-color write (induction head), then TRANSFORM it (a lookup). This
generalizes the even/odd interleaving from 2 stripes to C colored stripes, and stays in the associative-recall
learnable regime (NOT modular arithmetic).

STRIPED EDIT-RESPONSE (the point): editing a value at item t (color c) changes every later read of color c up
to the NEXT c-write -> a stripe segment. With C colors the sequence's dependency structure is C interleaved
stripes, KNOWN exactly. The certificate's reader-set / reach must recover these stripes -- the rich validation
MQAR lacks (MQAR readers scan; here each reader depends on exactly one upstream write, and edits paint stripes).

ENCODING (2 tokens per item, MQAR-style; fits SeqDEQ vocab):
  color c at even slot  -> token id c            (0..C-1, key range; also the query token)
  value v at odd slot   -> token id NKEY+v       (value range)
  reader (query)        -> the color id c        (READONLY_Q)     target = T[latest value of color c]
  head over NVAL classes reads out the transformed value (V<=NVAL).

Self-test (__main__): target == T[latest same-color value]; editing a value creates a STRIPE (only later
same-color reads change, and only up to the next same-color write) -- the ground-truth reader-set.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.colored_recall
"""
import torch

import experiments.sliding_window_reach as sw

C = 3                                 # number of colors (repeated keys -> stripes); C <= NKEY
V = sw.NVAL                           # number of values; head reads out T(value); V <= NVAL
VAL0 = sw.NKEY                        # value token base (disjoint from color ids 0..C-1)
FILLER0 = sw.NKEY + sw.NVAL

_TPERM = torch.randperm(V, generator=torch.Generator().manual_seed(1234))   # fixed transform T (a permutation)


def transform():
    return _TPERM


def gen_colored_recall(batch, n_items, gen, fill=0, C=C, V=V):
    """Returns toks (B,L), qmask (B,L), targ (B,L), deps (B x NQ dicts with the ground-truth stripe).
    Each reader asks a color present in its row; target = T[latest value written to that color]."""
    T = _TPERM
    colors = torch.randint(C, (batch, n_items), generator=gen)          # repeated keys
    values = torch.randint(V, (batch, n_items), generator=gen)
    L = 2 * n_items + fill + sw.NQ
    toks = torch.zeros(batch, L, dtype=torch.long)
    toks[:, 0:2 * n_items:2] = colors
    toks[:, 1:2 * n_items:2] = VAL0 + values
    if fill > 0:
        toks[:, 2 * n_items:2 * n_items + fill] = FILLER0 + torch.randint(sw.NFILL, (batch, fill), generator=gen)
    qbase = 2 * n_items + fill
    qmask = torch.zeros(batch, L, dtype=torch.bool); qmask[:, qbase:] = True
    targ = torch.zeros(batch, L, dtype=torch.long)
    deps = []
    for b in range(batch):
        present = colors[b].unique()
        dq = []
        for q in range(sw.NQ):
            c = int(present[torch.randint(len(present), (1,), generator=gen)])
            idxs = (colors[b] == c).nonzero().flatten()
            t = int(idxs[-1])                                           # latest write of color c
            v = int(values[b, t])
            qp = qbase + q
            toks[b, qp] = c                                            # query token = the color id
            targ[b, qp] = int(T[v])
            # the stripe = reads of color c that see item t as their latest write (here: the query qp).
            # ground-truth reader-set of this reader = {the color write 2t, the value write 2t+1}.
            dq.append({"query": qp, "color": c, "colorpos": 2 * t, "valpos": 2 * t + 1,
                       "writes": [2 * int(i) for i in idxs]})
        deps.append(dq)
    return toks.to(sw.DEV), qmask.to(sw.DEV), targ.to(sw.DEV), deps


def _latest_val(colors_row, values_row, c):
    idxs = (colors_row == c).nonzero().flatten()
    return int(values_row[int(idxs[-1])]) if len(idxs) else None


def main():
    g = torch.Generator().manual_seed(0)
    T = _TPERM
    print(f"transform T (value -> T[value]): {T.tolist()}  (C={C} colors, V={V} values)\n", flush=True)
    for n_items, fill in [(4, 0), (6, 4)]:
        toks, qmask, targ, deps = gen_colored_recall(1, n_items, g, fill=fill)
        L = toks.shape[1]
        colors = toks[0, 0:2 * n_items:2]; values = toks[0, 1:2 * n_items:2] - VAL0
        print(f"=== n_items={n_items} fill={fill} L={L} ===", flush=True)
        print("colors:", colors.tolist(), " values:", values.tolist(), flush=True)
        qpos = qmask[0].nonzero().flatten().tolist()
        ok = True
        for q, p in enumerate(qpos):
            c = int(toks[0, p]); lv = _latest_val(colors, values, c)
            ok &= int(targ[0, p]) == int(T[lv])
        print(f"targets = T[latest same-color value]: {ok}  (queries ask colors {[int(toks[0,p]) for p in qpos]})",
              flush=True)
        # striped edit: perturb the value at the first query's latest-write; only that stripe should move
        d0 = deps[0][0]; vp = d0["valpos"]; c = d0["color"]
        moved_same, moved_other = 0, 0
        for q, p in enumerate(qpos):
            cc = int(toks[0, p]); lv = _latest_val(colors, values, cc)
            base = int(T[lv])
            v2 = values.clone(); v2[(vp - 1) // 2] = (v2[(vp - 1) // 2] + 1) % V
            lv2 = _latest_val(colors, v2, cc); new = int(T[lv2])
            if cc == c:
                moved_same += int(new != base)
            else:
                moved_other += int(new != base)
        print(f"edit value at pos {vp} (color {c}): changes SAME-color reads ({moved_same}), "
              f"OTHER-color reads unchanged ({moved_other}==0 must hold) -> striped ground-truth\n", flush=True)
    print("READ: each reader depends on exactly ONE upstream write (its latest same-color value); an edit\n"
          "paints a color-stripe. Known reader-set + stripe = the reach/reader-set certificate's target.", flush=True)


if __name__ == "__main__":
    main()
