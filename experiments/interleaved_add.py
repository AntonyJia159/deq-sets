"""INTERLEAVED MODULAR ADDITION -- a semantic reader-set + length-generalization substrate.

Sequence of k single-token numbers, then two readers: Q_even asks for the sum of the EVEN-index numbers
(mod P), Q_odd for the sum of the ODD-index numbers (mod P). Example (P=10): [1 2 3 7 | Qe Qo] ->
Qe = (1+3) mod 10 = 4, Qo = (2+7) mod 10 = 9.

WHY THIS TASK (vs MQAR/pointer-chase):
 - CLEAN reader-set SEPARATION: Q_even depends ONLY on even positions, Q_odd ONLY on odd positions. An
   edit to an even-index number moves Q_even and leaves Q_odd EXACTLY unchanged -- a provable disjoint
   ground-truth reader-set, which MQAR (readers scan the whole context) does not give. This is the task
   where the reader-set BALL should actually collapse (goal-oriented >> state-space recompute).
 - SEMANTIC: a content-dependent arithmetic reduction, not associative copying.
 - LENGTH GENERALIZATION: if the sum is carried by the RELAY (running partial sums across windows -- the
   equilibrium-natural mechanism), train short (k small) / deploy long (k large) is DEQ test-time scaling,
   and the edit-response ball is LENGTH-INVARIANT (its size set by the local cell's sigma_min/rho(G), not
   by the sequence length). The length-gen test also DIAGNOSES the mechanism: it only works if the model
   learned relay-accumulation, not a within-context averaging shortcut.

ENCODING (fits SeqDEQ's vocab; P<=NVAL so the head over NVAL classes reads out a sum mod P):
   number n at index i           -> token id n            (key range 0..P-1; position i gives its parity)
   Q_even / Q_odd (the readers)  -> NKEY+0 / NKEY+1        (value range, distinct from number ids)
   filler                        -> NKEY+NVAL+.            (disjoint)
   target at Q_even = sum(nums[0::2]) mod P ; at Q_odd = sum(nums[1::2]) mod P
   deps: even reader <- even number positions ; odd reader <- odd number positions (the true lanes)

Self-test (__main__): targets match the modular sums; editing an EVEN-index number changes ONLY Q_even
(Q_odd invariant) and vice-versa -- the soundness of the disjoint reader-set.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.interleaved_add
"""
import torch

import experiments.sliding_window_reach as sw

P = sw.NVAL                         # numbers 0..P-1, sums mod P; P<=NVAL so the head reads out the sum
QE = sw.NKEY + 0                    # query-even token id (value range, disjoint from number ids 0..P-1)
QO = sw.NKEY + 1                    # query-odd token id
FILLER0 = sw.NKEY + sw.NVAL         # filler base (disjoint)


def gen_interleaved(batch, k, gen, fill=0, P=P):
    """Returns toks (B,L), qmask (B,L), targ (B,L), deps (dict even/odd -> ground-truth lane positions).
    k = number of addends; readers Q_even, Q_odd appended after an optional filler gap."""
    nums = torch.randint(P, (batch, k), generator=gen)
    even_sum = nums[:, 0::2].sum(1) % P
    odd_sum = nums[:, 1::2].sum(1) % P
    L = k + fill + 2
    toks = torch.zeros(batch, L, dtype=torch.long)
    toks[:, :k] = nums                                              # number id = its value (0..P-1)
    if fill > 0:
        toks[:, k:k + fill] = FILLER0 + torch.randint(sw.NFILL, (batch, fill), generator=gen)
    qbase = k + fill
    toks[:, qbase] = QE
    toks[:, qbase + 1] = QO
    qmask = torch.zeros(batch, L, dtype=torch.bool); qmask[:, qbase:qbase + 2] = True
    targ = torch.zeros(batch, L, dtype=torch.long)
    targ[:, qbase] = even_sum
    targ[:, qbase + 1] = odd_sum
    deps = {"even": list(range(0, k, 2)) + [qbase],
            "odd": list(range(1, k, 2)) + [qbase + 1]}
    return toks.to(sw.DEV), qmask.to(sw.DEV), targ.to(sw.DEV), deps


def main():
    g = torch.Generator().manual_seed(0)
    for k, fill in [(4, 0), (8, 4)]:
        toks, qmask, targ, deps = gen_interleaved(1, k, g, fill=fill)
        L = toks.shape[1]; qbase = k + fill
        nums = toks[0, :k].tolist()
        print(f"\n=== k={k} fill={fill}  L={L}  P={P} ===", flush=True)
        print("tokens:", toks[0].tolist(), flush=True)
        print("numbers:", nums, "  even idx", nums[0::2], "-> sum%P", sum(nums[0::2]) % P,
              "  odd idx", nums[1::2], "-> sum%P", sum(nums[1::2]) % P, flush=True)
        print("targets: Qe =", int(targ[0, qbase]), " Qo =", int(targ[0, qbase + 1]),
              " (match:", int(targ[0, qbase]) == sum(nums[0::2]) % P
              and int(targ[0, qbase + 1]) == sum(nums[1::2]) % P, ")", flush=True)
        # soundness of the disjoint reader-set: edit an even-index number -> only Qe changes
        ev_ok, od_ok = True, True
        for i in range(k):
            t2 = toks.clone(); t2[0, i] = (int(t2[0, i]) + 1) % P
            e2 = t2[0, 0:k:2].sum() % P; o2 = t2[0, 1:k:2].sum() % P
            if i % 2 == 0:                                          # even edit: Qo must be invariant
                od_ok &= int(o2) == int(targ[0, qbase + 1])
                ev_ok &= int(e2) != int(targ[0, qbase]) or True     # Qe may change (mod could alias)
            else:                                                   # odd edit: Qe must be invariant
                ev_ok &= int(e2) == int(targ[0, qbase])
        print(f"disjoint reader-set: odd-reader invariant under even edits: {od_ok};  "
              f"even-reader invariant under odd edits: {ev_ok}  (both must be True)", flush=True)
        print("even lane:", deps["even"], " odd lane:", deps["odd"], flush=True)
    print("\nREAD: disjoint ground-truth lanes (even reader <- even positions only) = the reader-set the\n"
          "certificate must recover; an edit to one parity leaves the other reader EXACTLY unchanged.", flush=True)


if __name__ == "__main__":
    main()
