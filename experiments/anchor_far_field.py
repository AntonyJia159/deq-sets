"""ANCHOR far-field probe -- does the global register wreck banded locality (global diffusion), or does the
model learn to use it selectively (emergent filtering)? Runs on currnpanchor60 vs the no-anchor currnp40.

The resolvent R=(I-J)^{-1} of the anchor model lives on the L+1 state (slot 0 = register). Questions:
  (1) BODY LOCALITY: does ||R[i,j]|| still decay with |i-j| for body pairs, or is it flat (diffused)?
      Contrast the no-anchor banded currnp40 reach curve.
  (2) ANCHOR SELECTIVITY (emergent filtering): the anchor AGGREGATES context via ||R[0,j]|| and is READ via
      ||R[i,0]||. Break both down by token type (key/value/filler/query). Peaked on task-relevant = filtering;
      flat = diffusion.
  (3) BORDER RANK: the anchor's cross-blocks R[body,0] and R[0,body] -- effective rank (participation ratio);
      the "banded + rank-r border" certificate needs r small.
  (4) THROUGH-ANCHOR SHARE at long range: at |i-j|>W the direct banded path ~0, so any reach is anchor- (or
      multi-hop) mediated; compare ||R[i,j]|| to the 2-hop-through-hub proxy ||R[i,0]|| ||R[0,j]|| / ||R[0,0]||.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.anchor_far_field
"""
import os
from collections import defaultdict

import numpy as np
import torch

import experiments.sliding_window_reach as sw
from experiments.c2_edit_locality import load_checkpoint, counted_solve, make_ff

CKPT_DIR = "checkpoints"
N_SEQS = 3
sw.H, sw.dh = 4, sw.d // 4


def solve_resolvent(m, toks):
    ff, h0 = make_ff(m, toks)
    L1 = h0.shape[1]
    z, _ = counted_solve(m, ff, torch.zeros(1, L1, sw.d, device=sw.DEV))
    zf = z.reshape(-1).detach()
    ffl = lambda zv: ff(zv.view(z.shape)).reshape(-1)
    J = torch.func.jacrev(ffl)(zf).detach().double()
    N = zf.numel()
    ImJ = torch.eye(N, device=J.device, dtype=torch.float64) - J
    R = torch.linalg.inv(ImJ)
    smin = torch.linalg.svdvals(ImJ).min().item()
    rho = torch.linalg.eigvals(J).abs().max().item()
    return R, L1, smin, rho


def blk(p, d, dev):
    return torch.arange(p * d, (p + 1) * d, device=dev)


def tok_type(p, has_anchor, Fill):
    """token type of STATE index p (skip anchor slot 0 if present)."""
    t = (p - 1) if has_anchor else p
    if t < 0:
        return "anchor"
    if t < 2 * sw.D_PAIR:
        return "key" if t % 2 == 0 else "value"
    if t < 2 * sw.D_PAIR + Fill:
        return "filler"
    return "query"


def eff_rank(M):
    sv = torch.linalg.svdvals(M.double()).cpu().numpy()
    return float((sv.sum() ** 2) / ((sv ** 2).sum() + 1e-30)), float(sv[0] / (sv[min(4, len(sv)-1)] + 1e-30))


def reach_curve(R, L1, d, off):
    """mean ||R[i,j]|| by causal distance i-j over BODY pairs (both >= off; off=1 skips the anchor slot)."""
    dev = R.device
    by = defaultdict(list)
    for i in range(off, L1):
        for j in range(off, i + 1):
            by[i - j].append(torch.linalg.matrix_norm(R[blk(i, d, dev)][:, blk(j, d, dev)], ord=2).item())
    return {k: float(np.mean(v)) for k, v in sorted(by.items())}


def main():
    print(f"device={sw.DEV}  ANCHOR far-field probe: banded locality survival + anchor selectivity "
          f"(emergent filtering vs diffusion) + border rank.\n  currnpanchor60 vs no-anchor currnp40. "
          f"{N_SEQS} seqs.\n", flush=True)
    for fname, has_anchor, gap in [("currnpanchor60.pt", True, 60), ("currnp40.pt", False, 40)]:
        path = os.path.join(CKPT_DIR, fname)
        if not os.path.exists(path):
            print(f"[{fname}] missing\n"); continue
        sw.ANCHOR = has_anchor
        m, ck = load_checkpoint(path)
        d = sw.d
        gen = torch.Generator().manual_seed(31)
        curves, smins, rhos = [], [], []
        agg = defaultdict(list); rd = defaultdict(list)
        rank_col, rank_row, far_share = [], [], []
        for _ in range(N_SEQS):
            toks = sw.gen_mqar(1, gap, gen)[0]
            L = toks.shape[1]
            R, L1, smin, rho = solve_resolvent(m, toks)
            dev = R.device
            smins.append(smin); rhos.append(rho)
            off = 1 if has_anchor else 0
            curves.append(reach_curve(R, L1, d, off))
            if has_anchor:
                a0 = blk(0, d, dev)
                r00 = torch.linalg.matrix_norm(R[a0][:, a0], ord=2).item()
                Rcol = R[:, a0].clone(); Rrow = R[a0, :].clone()                # (L1 d, d), (d, L1 d)
                for p in range(1, L1):
                    pb = blk(p, d, dev)
                    agg[tok_type(p, True, gap)].append(torch.linalg.matrix_norm(R[a0][:, pb], ord=2).item())
                    rd[tok_type(p, True, gap)].append(torch.linalg.matrix_norm(R[pb][:, a0], ord=2).item())
                # border rank (body x anchor blocks)
                body = torch.cat([blk(p, d, dev) for p in range(1, L1)])
                rank_col.append(eff_rank(R[body][:, a0])[0])
                rank_row.append(eff_rank(R[a0][:, body])[0])
                # through-anchor share at long range (i-j > W, body)
                for i in range(1, L1):
                    for j in range(1, i - sw.W):
                        tot = torch.linalg.matrix_norm(R[blk(i, d, dev)][:, blk(j, d, dev)], ord=2).item()
                        proxy = (torch.linalg.matrix_norm(R[blk(i, d, dev)][:, a0], ord=2).item() *
                                 torch.linalg.matrix_norm(R[a0][:, blk(j, d, dev)], ord=2).item() / (r00 + 1e-30))
                        far_share.append(min(proxy / (tot + 1e-30), 1.0))
        # ---- report
        tag = "ANCHOR" if has_anchor else "no-anchor (banded)"
        print(f"[{fname}] {tag}  gap={gap}  recall={ck['recall']:.3f}  sigma_min={np.mean(smins):.4f}  "
              f"rho={np.mean(rhos):.2f}", flush=True)
        # merged reach curve
        alld = defaultdict(list)
        for c in curves:
            for k, v in c.items():
                alld[k].append(v)
        pts = [(k, float(np.mean(v))) for k, v in sorted(alld.items())]
        show = [p for p in pts if p[0] in (0, 1, 2, 5, 10, 15, 20, 30, 40, 50)]
        print("    body reach ||R[i,j]|| by causal dist: " +
              "  ".join(f"d{k}={v:.2e}" for k, v in show), flush=True)
        if has_anchor:
            print("    anchor AGGREGATION ||R[0,j]|| by type: " +
                  "  ".join(f"{t}={np.mean(agg[t]):.2e}" for t in ["key", "value", "filler", "query"] if agg[t]),
                  flush=True)
            print("    anchor READ ||R[i,0]|| by type:       " +
                  "  ".join(f"{t}={np.mean(rd[t]):.2e}" for t in ["key", "value", "filler", "query"] if rd[t]),
                  flush=True)
            kf = np.mean([np.mean(agg["key"] + agg["value"])]) / (np.mean(agg["filler"]) + 1e-30) if agg["filler"] else float("nan")
            print(f"    -> SELECTIVITY (key+value)/filler aggregation = {kf:.1f}x  "
                  f"(>>1 = emergent filtering; ~1 = diffuse averaging)", flush=True)
            print(f"    border eff-rank: R[body,anchor]={np.mean(rank_col):.1f}  R[anchor,body]={np.mean(rank_row):.1f} "
                  f"(of d={d})  |  through-anchor share of far reach (d>W): {np.mean(far_share)*100:.0f}%", flush=True)
        print(flush=True)
    print("READ: if the ANCHOR body reach still DECAYS with distance (like the banded baseline) then FLOORS,\n"
          "banded locality survives + the anchor adds a distance-independent channel. SELECTIVITY>>1 = the hub\n"
          "learned to carry task-relevant tokens (emergent filtering), not diffuse everything. Low border rank\n"
          "+ high through-anchor share = the far reach is a certifiable low-rank anchor border, as the theory wants.", flush=True)


if __name__ == "__main__":
    main()
