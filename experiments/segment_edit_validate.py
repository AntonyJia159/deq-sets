"""SEGMENT-AVERAGE edit validation -- the Green's-function certificate on the WELL-CONDITIONED (BVP) face.

The clean payoff colored-recall could not give (it was near-singular -> diffuse reader-set). Here averaging is
contractive -> well-conditioned, so an edit to one anchor value should produce a COMPACT two-sided tent, and:
  (A) the model's field response should match the GROUND-TRUTH segment tent  d target_i / d v_p =
      exp(-|i-p|/tau) * same_seg(i,p) / Z_i  -- segment-bounded, exp-decaying, ZERO past the boundaries;
  (B) the certified resolvent (I-J)^{-1} should PREDICT it (linear response accurate, since well-conditioned);
  (C) the resolvent reach xi should be FINITE (decaying blocks -> LOCAL edits) -- the contrast to colored-
      recall's xi=inf (near-singular, non-local);
  (D) the edit-response should be a CLEAN collapse onto the segment (energy-inside-segment ~1), vs the 66-71%
      diffuse ball on the near-singular recall face.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.segment_edit_validate
"""
import os
from collections import defaultdict

import numpy as np
import torch

import experiments.sliding_window_reach as sw
import experiments.segment_average as sa

CKPT_DIR = "checkpoints"


def load_factored(path):
    ck = torch.load(path, map_location=sw.DEV, weights_only=False)
    sw.BIDIR = ck["bidir"]; sw.REL_BIAS = ck["rel_bias"]; sw.READONLY_Q = ck["readonly_q"]
    sw.QUERY_FULL = ck["query_full"]; sw.NO_POSW = ck["no_posw"]
    sw.FACTORED = ck["factored"]; sw.D_VALUE = ck["d_value"]
    sw.H, sw.dh = ck["H"], sw.d // ck["H"]; sw.W = ck["W"]
    m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


def solve_only(m, toks, values):
    """Tight fixed point only (no Jacobian/resolvent) -- for the nonlinear re-solve."""
    h0 = m.h0(toks, values)
    mask = sw.band_causal_mask(toks.shape[1], toks.device)
    maskp = m._maskp(mask); wn = m.wn()
    ff = lambda z: m.f(z, h0, wn, maskp)
    with torch.no_grad():
        z = m.deq(ff, torch.zeros_like(h0))[0][-1]
    ffl = lambda zv: ff(zv.view(z.shape)).reshape(-1)
    for _ in range(5):
        r = (ff(z) - z).detach()
        if (r.norm() / (z.norm() + 1e-9)).item() < 1e-8:
            break
        J = torch.func.jacrev(ffl)(z.reshape(-1).detach())
        ImJ = torch.eye(z.numel(), device=J.device, dtype=torch.float64) - J.double()
        z = (z.reshape(-1) + torch.linalg.solve(ImJ, r.reshape(-1).double()).float()).view(z.shape).detach()
    return z


def solve_and_resolvent(m, toks, values):
    """Tight fixed point (Anderson + Newton polish), dense J, resolvent R = (I-J)^{-1} (fp64)."""
    h0 = m.h0(toks, values)
    mask = sw.band_causal_mask(toks.shape[1], toks.device)
    maskp = m._maskp(mask); wn = m.wn()
    ff = lambda z: m.f(z, h0, wn, maskp)
    with torch.no_grad():
        z = m.deq(ff, torch.zeros_like(h0))[0][-1]
    ffl = lambda zv: ff(zv.view(z.shape)).reshape(-1)
    for _ in range(5):                                                  # Newton polish (well-conditioned)
        r = (ff(z) - z).detach()
        if (r.norm() / (z.norm() + 1e-9)).item() < 1e-8:
            break
        J = torch.func.jacrev(ffl)(z.reshape(-1).detach())
        ImJ = torch.eye(z.numel(), device=J.device, dtype=torch.float64) - J.double()
        step = torch.linalg.solve(ImJ, r.reshape(-1).double())
        z = (z.reshape(-1) + step.float()).view(z.shape).detach()
    J = torch.func.jacrev(ffl)(z.reshape(-1).detach()).detach().double()
    N = z.numel()
    R = torch.linalg.inv(torch.eye(N, device=J.device, dtype=torch.float64) - J)
    smin = torch.linalg.svdvals(torch.eye(N, device=J.device, dtype=torch.float64) - J).min().item()
    rho = torch.linalg.eigvals(J).abs().max().item()
    return z, ff, J, R, smin, rho


def gt_tent(p, seg_id, is_bnd, L, tau):
    """Ground-truth response coefficient d target_i / d v_p  (segment-bounded exp tent)."""
    idx = torch.arange(L)
    val_mask = (~is_bnd).float()
    same = (seg_id[:, None] == seg_id[None, :]).float()
    w = torch.exp(-(idx[:, None] - idx[None, :]).abs().float() / tau) * same * val_mask[None, :]
    Z = w.sum(-1) + 1e-9
    return (w[:, p] / Z) * val_mask                                    # (L,) tent, zero outside segment / at bnds


def resolvent_reach_xi(R, L, d):
    Rb = R.reshape(L, d, L, d)
    dn = defaultdict(list)
    for i in range(L):
        for j in range(L):
            dn[abs(i - j)].append(torch.linalg.matrix_norm(Rb[i, :, j, :], ord=2).item())
    deltas = np.array(sorted(dn)); means = np.array([np.mean(dn[dl]) for dl in deltas])
    mkeep = (deltas >= 1) & (means > 1e-12)
    if mkeep.sum() < 3:
        return np.nan
    slope = np.polyfit(deltas[mkeep], np.log(means[mkeep]), 1)[0]
    return (-1.0 / slope) if slope < 0 else np.inf


def main():
    path = os.path.join(CKPT_DIR, "segment_average.pt")
    if not os.path.exists(path):
        print(f"missing {path}"); return
    m, ck = load_factored(path)
    print(f"segment-average edit validation  (d_value={sw.D_VALUE}, W={sw.W}, tau={sa.TAU})\n", flush=True)

    gen = torch.Generator().manual_seed(21)
    n_seq, edits_per = 6, 3
    corr_lin, corr_nl, inside, lin_vs_nl, xis, smins, rhos = [], [], [], [], [], [], []
    printed = False
    for s in range(n_seq):
        L = 32
        toks, values, target, tmask, seg_id = sa.gen_segment_average(1, L, gen)
        is_bnd = (toks[0] == sa.BOUNDARY_MODE).cpu()
        z, ff, J, R, smin, rho = solve_and_resolvent(m, toks, values)
        smins.append(smin); rhos.append(rho); xis.append(resolvent_reach_xi(R, L, sw.d))
        val_pos = (toks[0] == sa.VALUE_MODE).nonzero().flatten().tolist()
        for _ in range(edits_per):
            p = val_pos[torch.randint(len(val_pos), (1,), generator=gen).item()]
            seg_of_p = int(seg_id[0, p])
            delta = torch.zeros(1, L, sw.d, device=sw.DEV)              # perturb the value subspace at p
            dv = torch.randn(sw.D_VALUE, generator=gen).to(sw.DEV) * 1.5
            delta[0, p, sw.d - sw.D_VALUE:] = dv
            # (B) linear response via the resolvent; field = head_reg(dz)
            dz = (R @ delta.reshape(-1).double()).float().view(1, L, sw.d)
            resp_lin = m.head_reg(dz)[0].norm(dim=-1).detach().cpu().numpy()
            # nonlinear response via re-solve
            v2 = values.clone(); v2[0, p] += dv
            z2 = solve_only(m, toks, v2)
            resp_nl = (m.head_reg(z2) - m.head_reg(z))[0].norm(dim=-1).detach().cpu().numpy()
            # (A) ground-truth tent (times ||dv||, per-position norm)
            gt = (gt_tent(p, seg_id[0].cpu(), is_bnd, L, sa.TAU).numpy() * float(dv.norm().cpu()))
            in_seg = (seg_id[0].cpu().numpy() == seg_of_p) & (~is_bnd.numpy())
            inside.append(float((resp_nl[in_seg] ** 2).sum() / (resp_nl ** 2).sum() + 1e-12))
            if gt.std() > 1e-9 and resp_nl.std() > 1e-9:
                corr_lin.append(float(np.corrcoef(resp_lin, gt)[0, 1]))
                corr_nl.append(float(np.corrcoef(resp_nl, gt)[0, 1]))
            lin_vs_nl.append(float(np.linalg.norm(resp_lin - resp_nl) / (np.linalg.norm(resp_nl) + 1e-9)))
            if not printed and in_seg.sum() >= 5:
                bnds = is_bnd.nonzero().flatten().tolist()
                print(f"  example edit at value pos {p} (segment {seg_of_p}, boundaries {bnds}):", flush=True)
                print(f"    ground-truth tent : {[round(x,2) for x in gt.tolist()]}", flush=True)
                print(f"    model (resolvent) : {[round(x,2) for x in resp_lin.tolist()]}", flush=True)
                print(f"    model (re-solve)  : {[round(x,2) for x in resp_nl.tolist()]}\n", flush=True)
                printed = True

    print(f"  (A) response vs ground-truth tent: corr(resolvent)={np.mean(corr_lin):.3f}  "
          f"corr(re-solve)={np.mean(corr_nl):.3f}  (n={len(corr_nl)})", flush=True)
    print(f"  (B) resolvent linear response vs nonlinear re-solve: rel diff={np.mean(lin_vs_nl):.3f} "
          f"(small => well-conditioned linear response accurate)", flush=True)
    print(f"  (C) resolvent reach xi={np.nanmean(xis):.1f} positions (FINITE => local edits; contrast "
          f"colored-recall xi=inf)  sigma_min={np.mean(smins):.3f} rho={np.mean(rhos):.3f}", flush=True)
    print(f"  (D) edit-response energy INSIDE the segment: {np.mean(inside)*100:.0f}% "
          f"(clean collapse; contrast the 66-71% diffuse ball on the near-singular recall face)", flush=True)
    print("\nREAD: high corr + energy-inside ~1 + finite xi + linear~nonlinear = the resolvent certifies a\n"
          "compact two-sided Green's-function edit on the well-conditioned BVP face -- the clean edit-locality\n"
          "demo the near-singular recall tasks could not deliver.", flush=True)


if __name__ == "__main__":
    main()
