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


def gen_colored_recall(batch, n_items, gen, fill=0, C=C, V=V, mode="latest"):
    """Returns toks (B,L), qmask (B,L), targ (B,L), deps (B x NQ dicts with the ground-truth stripe).
    Each reader asks a color present in its row; target = T[SELECTED value written to that color].
      mode='latest'   -> the LAST same-color write (recency/induction; SHORT-range, easy).
      mode='earliest' -> the FIRST same-color write (write-once register; LONG-range dependency, confounds
                         recency -> a stronger reach-certificate test + a whole-suffix stripe)."""
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
            t = int(idxs[-1]) if mode == "latest" else int(idxs[0])     # last (recency) vs first (write-once)
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
    for mode in ("latest", "earliest"):
        print(f"########## mode = {mode} ##########", flush=True)
        for n_items, fill in [(8, 4)]:
            toks, qmask, targ, deps = gen_colored_recall(1, n_items, g, fill=fill, mode=mode)
            L = toks.shape[1]
            colors = toks[0, 0:2 * n_items:2]; values = toks[0, 1:2 * n_items:2] - VAL0
            print(f"n_items={n_items} fill={fill} L={L}  colors={colors.tolist()} values={values.tolist()}",
                  flush=True)
            qpos = qmask[0].nonzero().flatten().tolist()
            ok = True
            dists = []
            for q, p in enumerate(qpos):
                d = deps[0][q]
                idxs = [i for i in range(n_items) if int(colors[i]) == d["color"]]
                t = idxs[-1] if mode == "latest" else idxs[0]
                ok &= int(targ[0, p]) == int(T[int(values[t])])
                dists.append(p - d["valpos"])                          # query-to-dependency distance
            print(f"  targets correct: {ok}  | query->dependency distance: {dists}  "
                  f"({'SHORT (recency)' if mode == 'latest' else 'LONG (first occurrence)'})", flush=True)
            # soundness: perturb a NON-selected same-color write -> reader must NOT move (shadowed)
            c0 = deps[0][0]["color"]; sel = deps[0][0]["valpos"]
            same_writes = [w + 1 for w in deps[0][0]["writes"]]        # value positions of that color
            nonsel = [w for w in same_writes if w != sel]
            base = int(targ[0, qpos[0]])
            moved = 0
            for w in nonsel:
                v2 = values.clone(); ti = (w - 1) // 2; v2[ti] = (v2[ti] + 1) % V
                idxs = [i for i in range(n_items) if int(colors[i]) == c0]
                t = idxs[-1] if mode == "latest" else idxs[0]
                moved += int(int(T[int(v2[t])]) != base)
            print(f"  editing a NON-selected same-color write ({len(nonsel)} of them) moves the reader: "
                  f"{moved} (must be 0 = shadowed) -> sharp relevant/irrelevant taxonomy\n", flush=True)
    print("READ: 'earliest' makes the reader depend on a FAR first-occurrence (long-range) and shadows all\n"
          "later same-color writes -> a stronger reach test + whole-suffix stripe than 'latest' (recency).", flush=True)


if __name__ == "__main__":
    main()
