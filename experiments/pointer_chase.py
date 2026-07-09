"""POINTER-CHASE task -- C6 hub/spoke substrate with a KNOWN dependency graph.

The point of C6: MQAR's dependency structure is planted and near-uniform. Pointer-chase gives positions
GENUINELY heterogeneous influence footprints AND a ground-truth dependency graph, so we can test whether the
certified reader-set (resolvent influence cone) contains and tightly tracks the TRUE set -- and whether the
recompute "zigzags in narrow lanes" (permutation) or fills the cone (random-function = shared-tail hubs).

ENCODING (fits SeqDEQ's vocab; N=NKEY nodes, values live in NODE space so a value can be chased as a key --
that is what makes it multi-hop, unlike MQAR where values are a disjoint range):
  [ n , ptr(n) ] x N pairs in random order   (key at even pos, its pointer-target at odd pos; both node ids)
  [ filler ] x Fill
  [ start ] x NQ queries                      answer = ptr^chain_len(start), read out at the query position
Editing the VALUE token of node j's pair = re-pointing j. A query's answer depends on EXACTLY the value tokens
of the chain nodes {start, ptr(start), ..., ptr^{k-1}(start)} -> that set is the ground-truth dependency lane.

  perm mode  : ptr is a permutation -> disjoint cycles -> each chain is a narrow lane (the sparse ideal).
  random mode: ptr is a random function -> rho-shaped graphs with SHARED TAILS = super-hubs (heterogeneous).

Self-test (__main__): verifies ans==ptr^k(start), that editing a dependency value FLIPS the answer, and that
editing a NON-dependency value does NOT -> the ground-truth graph is correct (C6 hinges on this).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.pointer_chase
"""
import torch

import experiments.sliding_window_reach as sw

N = sw.NKEY                      # nodes = 8 (reuse key vocab; answer head has NVAL=8 outputs, N<=NVAL)
FILLER0 = sw.NKEY + sw.NVAL      # filler token base (disjoint from node ids 0..N-1)


def gen_pointer_chase(batch, Fill, gen, chain_len=3, mode="perm"):
    """Returns toks (B,L), qmask (B,L) bool, targ (B,L) long, deps (list len B of list len NQ of value-token
    positions the query depends on), ptr (B,N). Node ids 0..N-1; value tokens are also node ids (chainable)."""
    if mode == "perm":
        ptr = torch.rand(batch, N, generator=gen).argsort(1)                 # permutation per row
    else:
        ptr = torch.randint(N, (batch, N), generator=gen)                    # random function
    order = torch.rand(batch, N, generator=gen).argsort(1)                   # layout order of the pairs
    L = 2 * N + Fill + sw.NQ
    toks = torch.zeros(batch, L, dtype=torch.long)
    valpos = torch.zeros(batch, N, dtype=torch.long)                         # node -> position of its value tok
    for slot in range(N):
        node = order[:, slot]                                               # (B,)
        tgt = ptr.gather(1, node[:, None])[:, 0]                            # ptr(node)
        toks[:, 2 * slot] = node                                            # key token = node id
        toks[:, 2 * slot + 1] = tgt                                         # value token = ptr(node), a node id
        valpos.scatter_(1, node[:, None], torch.full((batch, 1), 2 * slot + 1))
    if Fill > 0:
        toks[:, 2 * N:2 * N + Fill] = FILLER0 + torch.randint(sw.NFILL, (batch, Fill), generator=gen)
    starts = torch.randint(N, (batch, sw.NQ), generator=gen)                 # query = start node
    ans = starts.clone()
    for _ in range(chain_len):
        ans = ptr.gather(1, ans)                                            # ptr^k(start)
    qbase = 2 * N + Fill
    toks[:, qbase:] = starts
    qmask = torch.zeros(batch, L, dtype=torch.bool)
    qmask[:, qbase:] = True
    targ = torch.zeros(batch, L, dtype=torch.long)
    targ[:, qbase:] = ans
    # ground-truth dependency lane per query: value-token positions of the chain nodes
    deps = []
    for b in range(batch):
        dq = []
        for q in range(sw.NQ):
            cur = int(starts[b, q]); chain = []
            for _ in range(chain_len):
                chain.append(cur); cur = int(ptr[b, cur])
            dq.append([int(valpos[b, c]) for c in chain])
            dq[-1].append(int(qbase + q))                                   # the query token itself
        deps.append(dq)
    return toks, qmask, targ, deps, ptr


def _follow(ptr_row, s, k):
    cur = int(s)
    for _ in range(k):
        cur = int(ptr_row[cur])
    return cur


def main():
    g = torch.Generator().manual_seed(0)
    for mode in ["perm", "random"]:
        toks, qmask, targ, deps, ptr = gen_pointer_chase(1, Fill=6, gen=g, chain_len=3, mode=mode)
        L = toks.shape[1]
        print(f"\n=== mode={mode}  N={N} chain_len=3  L={L} ===", flush=True)
        print("ptr:      ", ptr[0].tolist(), flush=True)
        print("tokens:   ", toks[0].tolist(), flush=True)
        qpos = qmask[0].nonzero().flatten().tolist()
        # 1) answers correct
        ok_ans = all(int(targ[0, p]) == _follow(ptr[0], toks[0, p], 3) for p in qpos)
        print(f"answers == ptr^3(start): {ok_ans}", flush=True)
        # 2) dependency graph: edit a dep value flips the answer; edit a non-dep value does not
        flips_on_dep, noflip_on_nondep = [], []
        for qi, p in enumerate(qpos):
            base_ans = int(targ[0, p])
            dep_valpos = [d for d in deps[0][qi] if d < 2 * N]              # value-token positions on the lane
            # edit each dependency: repoint that node -> re-derive answer
            for vp in dep_valpos:
                node = int(toks[0, vp - 1])                                 # key of this pair
                newtgt = (int(toks[0, vp]) + 1) % N
                pr = ptr[0].clone(); pr[node] = newtgt
                flips_on_dep.append(_follow(pr, toks[0, p], 3) != base_ans)
            # edit a value NOT on the lane -> answer must be unchanged
            nondep = [2 * s + 1 for s in range(N) if (2 * s + 1) not in dep_valpos]
            for vp in nondep[:3]:
                node = int(toks[0, vp - 1])
                newtgt = (int(toks[0, vp]) + 1) % N
                pr = ptr[0].clone(); pr[node] = newtgt
                noflip_on_nondep.append(_follow(pr, toks[0, p], 3) == base_ans)
        print(f"editing a DEPENDENCY value changes the answer: {sum(flips_on_dep)}/{len(flips_on_dep)} "
              f"(note: a repoint can coincidentally re-land, so <100% is OK)", flush=True)
        print(f"editing a NON-dependency value leaves it UNCHANGED: {sum(noflip_on_nondep)}/"
              f"{len(noflip_on_nondep)} (must be all)", flush=True)
        print(f"example lane (q0 dep positions): {deps[0][0]}", flush=True)
    print("\nREAD: ground-truth graph is correct if non-dependency edits NEVER change the answer (soundness of\n"
          "the true lane) and dependency edits usually do. This lane is what the certified reader-set must cover.",
          flush=True)


if __name__ == "__main__":
    main()
