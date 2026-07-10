"""COLORED-RECALL reach validation -- does the certificate's reach length predict length-generalization?

The pair colored_recall_{latest,earliest}.pt isolates the reach limit: 'latest' (short-range dep) length-
generalizes; 'earliest' (long-range, first-occurrence dep) does NOT -- recall falls as the dependency distance
grows past what reach supports. This probe cashes that in:

  (A) RECALL vs DEPENDENCY DISTANCE: bin per-query correctness by (query_pos - selected_write_pos). If reach is
      the limiter, recall falls off at a characteristic distance xi, and the SAME curve holds across n (it is
      DISTANCE, not n, that matters) -- so length-gen degrades only because longer n samples larger distances.
  (B) REACH LENGTH from the operator: the resolvent block norm ||[R]_{i,j}|| = ||(I-J)^{-1}||-decay with |i-j|
      gives xi_resolvent (how far an edit/value actually propagates). Claim: xi_resolvent ~ the recall falloff
      distance -> the certificate PREDICTS the length-gen breakpoint. Report sigma_min too.
  (C) STRIPED READER-SET: for a value edit, forward reach (R.dh) INTERSECT adjoint reach (from the query) should
      recover the ground-truth stripe (the ONE selected same-color write); reader-set ball << forward ball.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.colored_reach_validate
"""
import os
from collections import defaultdict

import numpy as np
import torch

import experiments.sliding_window_reach as sw
from experiments.c2_edit_locality import load_checkpoint
from experiments.c2d_directional import dense_resolvent
import experiments.colored_recall as cr

CKPT_DIR = "checkpoints"


def recall_vs_distance(m, mode, ns=(8, 16, 24, 32), batch=256, fill=8):
    """Per-query (dependency distance, correct); binned recall by distance, per n and pooled."""
    by_dist = defaultdict(list)
    per_n = {}
    for n in ns:
        gen = torch.Generator().manual_seed(100 + n)
        toks, qmask, targ, deps = cr.gen_colored_recall(batch, n, gen, fill=fill, mode=mode)
        with torch.no_grad():
            pred = m.run(toks).argmax(-1)
        accs = []
        for b in range(batch):
            for dq in deps[b]:
                qp, vp = dq["query"], dq["valpos"]
                dist = qp - vp
                correct = int(pred[b, qp].item() == targ[b, qp].item())
                by_dist[dist].append(correct)
                accs.append(correct)
        per_n[n] = float(np.mean(accs))
    curve = {dist: (float(np.mean(v)), len(v)) for dist, v in sorted(by_dist.items())}
    return curve, per_n


def resolvent_reach(m, toks):
    """xi from the resolvent block-norm decay: mean ||[R]_{i,j}|| over causal pairs at each delta=i-j, fit
    log-linear; also sigma_min and rho."""
    z, ff, J, R = dense_resolvent(m, toks)
    L, d = toks.shape[1], sw.d
    Rb = R.reshape(L, d, L, d)
    delta_norm = defaultdict(list)
    for i in range(L):
        for j in range(i + 1):
            delta_norm[i - j].append(torch.linalg.matrix_norm(Rb[i, :, j, :], ord=2).item())
    deltas = np.array(sorted(delta_norm))
    means = np.array([np.mean(delta_norm[dl]) for dl in deltas])
    # fit xi on the decaying tail (delta>=1, positive norms)
    m_ = (deltas >= 1) & (means > 1e-9)
    xi = np.nan
    if m_.sum() >= 3:
        p = np.polyfit(deltas[m_], np.log(means[m_]), 1)
        xi = -1.0 / p[0] if p[0] < 0 else np.inf
    N = z.numel()
    zf = z.reshape(-1)
    ffl = lambda zv: ff(zv.view(z.shape)).reshape(-1)
    Jd = torch.func.jacrev(ffl)(zf).detach()
    smin = torch.linalg.svdvals(torch.eye(N, device=Jd.device) - Jd).min().item()
    rho = torch.linalg.eigvals(Jd).abs().max().item()
    return xi, smin, rho, dict(zip(deltas.tolist(), means.tolist()))


def striped_readerset(m, mode, n=12, fill=8, n_seq=6, tau=1e-2):
    """For a value edit, forward reach INTERSECT adjoint(query) reach vs the ground-truth selected write.
    Reports: does the reader-set ball recover the true dependency, and its size vs the forward ball."""
    reds, hits, fwd_sizes, rb_sizes = [], [], [], []
    gen = torch.Generator().manual_seed(7)
    for _ in range(n_seq):
        toks, qmask, targ, deps = cr.gen_colored_recall(1, n, gen, fill=fill, mode=mode)
        L = toks.shape[1]
        reader_pos = qmask[0].nonzero().flatten().tolist()
        z, ff, J, R = dense_resolvent(m, toks)
        # adjoint reach from readers (head-weighted resolvent rows)
        rr = torch.cat([torch.arange(r * sw.d, (r + 1) * sw.d, device=R.device) for r in reader_pos])
        Hw = m.head.weight.detach().double()
        Gr = torch.einsum("vd,rdc->rvc", Hw, R[rr].view(len(reader_pos), sw.d, L * sw.d)
                          ).reshape(len(reader_pos) * Hw.shape[0], L * sw.d)
        adjoint = Gr.view(Gr.shape[0], L, sw.d).norm(dim=(0, 2)).cpu().numpy()
        # edit the selected write of the FIRST reader's dependency; forward reach = R @ dh
        d0 = deps[0][0]; vp = d0["valpos"]
        toks2 = toks.clone()
        toks2[0, vp] = cr.VAL0 + (int(toks[0, vp]) - cr.VAL0 + 1) % cr.V
        with torch.no_grad():
            dh = (m.h0(toks2) - m.h0(toks)).reshape(-1).double()
        forward = (R @ dh).view(L, sw.d).norm(dim=-1).cpu().numpy()
        F = forward > tau * forward.max()
        A = adjoint > tau * adjoint.max()
        RB = F & A
        fwd_sizes.append(int(F.sum())); rb_sizes.append(int(RB.sum()))
        reds.append(int(RB.sum()) / max(int(F.sum()), 1))
        # does the reader-set ball include the TRUE dependency (the selected write & its color token)?
        hits.append(int(RB[vp]) if vp < L else 0)
    return np.mean(reds), np.mean(hits), np.mean(fwd_sizes), np.mean(rb_sizes)


def main():
    print(f"device={sw.DEV}  colored-recall reach validation: does reach length predict length-gen?\n",
          flush=True)
    for mode in ("latest", "earliest"):
        cands = [os.path.join(CKPT_DIR, f"colored_recall_{mode}.pt")]
        if mode == "latest":
            cands.append(os.path.join(CKPT_DIR, "colored_recall.pt"))   # pre-parameterization name
        path = next((p for p in cands if os.path.exists(p)), None)
        if path is None:
            print(f"[{mode}] checkpoint missing: {cands}\n"); continue
        m, ck = load_checkpoint(path)
        print(f"===== mode={mode}  (recall_by_len {ck.get('lengen')}) =====", flush=True)

        # (A) recall vs dependency distance
        curve, per_n = recall_vs_distance(m, mode)
        print(f"  (A) recall by n: {', '.join(f'n{n}={a:.2f}' for n, a in per_n.items())}", flush=True)
        print("      recall vs DEPENDENCY DISTANCE (query - selected write):", flush=True)
        # coarse-bin the distance for readability
        binned = defaultdict(lambda: [0, 0])
        for dist, (acc, cnt) in curve.items():
            bk = (dist // 6) * 6
            binned[bk][0] += acc * cnt; binned[bk][1] += cnt
        for bk in sorted(binned):
            s, c = binned[bk]
            print(f"        dist [{bk:>2},{bk+6:>2}): recall={s/max(c,1):.2f}  (n_q={c})", flush=True)

        # (B) reach length from the resolvent
        xis, smins, rhos = [], [], []
        g2 = torch.Generator().manual_seed(3)
        for _ in range(3):
            toks = cr.gen_colored_recall(1, 10, g2, fill=8, mode=mode)[0]
            xi, smin, rho, _ = resolvent_reach(m, toks)
            xis.append(xi); smins.append(smin); rhos.append(rho)
        print(f"  (B) resolvent reach xi={np.nanmean(xis):.1f} positions  sigma_min={np.mean(smins):.3f}  "
              f"rho={np.mean(rhos):.2f}", flush=True)

        # (C) striped reader-set
        red, hit, fwd, rb = striped_readerset(m, mode)
        print(f"  (C) striped reader-set: ball={rb:.1f}/{fwd:.1f} pos ({red*100:.0f}% of forward); "
              f"includes true dependency {hit*100:.0f}% of the time\n", flush=True)
    print("READ: if 'earliest' recall falls off at dist ~ xi (and the falloff curve is n-independent) while\n"
          "'latest' stays flat (deps within xi), the resolvent reach length PREDICTS length-generalization =\n"
          "reach = memory horizon, certified. Reader-set ball << forward + recovers the true dependency =\n"
          "the striped reader-set the colored-recall pivot was for.", flush=True)


if __name__ == "__main__":
    main()
