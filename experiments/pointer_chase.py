"""POINTER-CHASE-TO-ROOT task -- C6 hub/spoke substrate with a KNOWN dependency graph.

WHY chase-to-ROOT (not fixed-k): a DEQ settles to a k-INDEPENDENT fixed point, so it can't count "exactly 3
hops" (the query carries no hop marker). What an equilibrium DOES compute naturally is a fixed point of the
chase: follow pointers until a TERMINAL. So the task is "find the root your chain reaches" = fixed-point
root-label propagation root(i)=root(ptr(i)), root(r)=r -- exactly what a DEQ relaxes into.

STRUCTURE: a forest. n_roots terminals (self-loops); every other node points to a random node of strictly
SMALLER level (so chains strictly descend to a root in <= depth hops; no non-terminal cycles). Answer(start) =
the root its chain reaches. Editing ptr(j) reroutes j's WHOLE SUBTREE -> a node near a root is a HUB (big
subtree, many queries affected), a leaf affects only itself = genuinely heterogeneous footprints. The
ground-truth dependency lane of a query = the value tokens of the nodes on its start->root path.

ENCODING (fits SeqDEQ's vocab; N=NKEY nodes; values live in NODE space so a value is chased as the next key):
  [ n , ptr(n) ] x N pairs (random order; key even pos, pointer-target odd pos; both node ids)
  [ filler ] x Fill
  [ start ] x NQ queries      answer = root(start), read out at the query position (head over N<=NVAL classes)

Self-test (__main__): answer is a root (ptr(ans)==ans); editing an ON-PATH value can change the root; editing
an OFF-PATH value never does (soundness of the true lane). depth curriculum knob for the trainer.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.pointer_chase
"""
import torch

import experiments.sliding_window_reach as sw

N = sw.NKEY                      # nodes = 8 (answer head has NVAL=8 outputs, N<=NVAL)
FILLER0 = sw.NKEY + sw.NVAL      # filler token base (disjoint from node ids 0..N-1)


def _build_forest(batch, gen, depth, n_roots):
    """ptr (batch,N): n_roots self-loop terminals; every other node -> a random strictly-smaller-level node."""
    levels = torch.randint(1, depth + 1, (batch, N), generator=gen)
    root_idx = torch.rand(batch, N, generator=gen).argsort(1)[:, :n_roots]      # distinct roots per row
    levels.scatter_(1, root_idx, 0)
    ptr = torch.arange(N).repeat(batch, 1)                                       # default self (roots)
    for b in range(batch):
        for i in range(N):
            if levels[b, i] == 0:
                continue
            cand = (levels[b] < levels[b, i]).nonzero().flatten()               # strictly shallower nodes
            ptr[b, i] = cand[torch.randint(len(cand), (1,), generator=gen)]
    return ptr


def gen_pointer_chase(batch, Fill, gen, depth=3, n_roots=2):
    """Returns toks (B,L), qmask (B,L), targ (B,L), deps (B x NQ list of start->root value-token positions),
    ptr (B,N)."""
    ptr = _build_forest(batch, gen, depth, n_roots)
    order = torch.rand(batch, N, generator=gen).argsort(1)
    L = 2 * N + Fill + sw.NQ
    toks = torch.zeros(batch, L, dtype=torch.long)
    valpos = torch.zeros(batch, N, dtype=torch.long)
    for slot in range(N):
        node = order[:, slot]
        toks[:, 2 * slot] = node                                             # key token: node id (0..N-1)
        toks[:, 2 * slot + 1] = sw.NKEY + ptr.gather(1, node[:, None])[:, 0]  # value token: NKEY+target (its
        #   OWN vocab range 8..15, disjoint from keys 0..7 -- so a query key matches only keys, not value copies
        #   of that node. Chaining survives via a learned value->key offset (value NKEY+x aligns to key x).
        valpos.scatter_(1, node[:, None], torch.full((batch, 1), 2 * slot + 1))
    if Fill > 0:
        toks[:, 2 * N:2 * N + Fill] = FILLER0 + torch.randint(sw.NFILL, (batch, Fill), generator=gen)
    starts = torch.randint(N, (batch, sw.NQ), generator=gen)
    ans = starts.clone()
    for _ in range(N):                                                          # <=N hops reaches the root
        ans = ptr.gather(1, ans)
    qbase = 2 * N + Fill
    toks[:, qbase:] = starts
    qmask = torch.zeros(batch, L, dtype=torch.bool); qmask[:, qbase:] = True
    targ = torch.zeros(batch, L, dtype=torch.long); targ[:, qbase:] = ans
    deps = []
    for b in range(batch):
        dq = []
        for q in range(sw.NQ):
            cur = int(starts[b, q]); path = []
            for _ in range(N):
                path.append(cur)
                nxt = int(ptr[b, cur])
                if nxt == cur:
                    break
                cur = nxt
            dq.append([int(valpos[b, c]) for c in path] + [int(qbase + q)])
        deps.append(dq)
    return toks.to(sw.DEV), qmask.to(sw.DEV), targ.to(sw.DEV), deps, ptr


def _root(ptr_row, s):
    cur = int(s)
    for _ in range(N):
        nxt = int(ptr_row[cur])
        if nxt == cur:
            return cur
        cur = nxt
    return cur


def main():
    g = torch.Generator().manual_seed(0)
    for depth, n_roots in [(3, 2), (5, 1)]:
        toks, qmask, targ, deps, ptr = gen_pointer_chase(1, Fill=6, gen=g, depth=depth, n_roots=n_roots)
        L = toks.shape[1]
        print(f"\n=== depth={depth} n_roots={n_roots}  L={L} ===", flush=True)
        print("ptr:    ", ptr[0].tolist(), "  roots:", [i for i in range(N) if int(ptr[0, i]) == i], flush=True)
        print("tokens: ", toks[0].tolist(), flush=True)
        qpos = qmask[0].nonzero().flatten().tolist()
        ok_root = all(int(ptr[0, int(targ[0, p])]) == int(targ[0, p]) for p in qpos)        # answer is a root
        ok_ans = all(int(targ[0, p]) == _root(ptr[0], toks[0, p]) for p in qpos)
        print(f"answer is a root: {ok_root}   answer == root(start): {ok_ans}", flush=True)
        onpath_flip, offpath_same = [], []
        for qi, p in enumerate(qpos):
            base = int(targ[0, p])
            path_vp = [d for d in deps[0][qi] if d < 2 * N]
            for vp in path_vp:
                node = int(toks[0, vp - 1]); pr = ptr[0].clone(); pr[node] = (int(pr[node]) + 1) % N
                onpath_flip.append(_root(pr, toks[0, p]) != base)
            offpath = [2 * s + 1 for s in range(N) if (2 * s + 1) not in path_vp]
            for vp in offpath[:3]:
                node = int(toks[0, vp - 1]); pr = ptr[0].clone(); pr[node] = (int(pr[node]) + 1) % N
                offpath_same.append(_root(pr, toks[0, p]) == base)
        print(f"editing ON-PATH value can change the root: {sum(onpath_flip)}/{len(onpath_flip)}", flush=True)
        print(f"editing OFF-PATH value never changes it:   {sum(offpath_same)}/{len(offpath_same)} (must be all)",
              flush=True)
        print(f"example lane (q0): {deps[0][0]}", flush=True)
    print("\nREAD: sound ground-truth graph = OFF-PATH edits never move the answer. The start->root path is the\n"
          "TRUE dependency lane the certified reader-set must cover; subtree size = hub-ness.", flush=True)


if __name__ == "__main__":
    main()
