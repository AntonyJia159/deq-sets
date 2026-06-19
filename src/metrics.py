"""Layer-1 probe metrics.

These answer the question "how free are path independence and exact unlearning?"
All metrics run under no_grad and operate on a single set x: (N, d_in).

  1. path_independence_gap -- solve from several random inits; report the spread
     of fixed points and whether the downstream prediction agrees. Large fp_gap
     with agreeing predictions => non-unique latent but invariant readout;
     disagreeing predictions => genuine path dependence that matters.

  2. unlearning_gap -- remove one point, then compare a warm-started re-solve
     (init = old fixed point minus that row) against a cold from-scratch solve.
     A nonzero gap is the "illusion detector": warm-start converged to a stale,
     history-dependent basin (the leakage / non-exact-unlearning failure mode).

  3. efficiency -- warm vs cold solver iterations after a removal (the
     test-time-scaling / efficiency bonus), reported alongside (2).
"""

import torch


def _relnorm(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-8)


@torch.no_grad()
def path_independence_gap(model, x, n_inits=5, **solve_kw):
    xb = x.unsqueeze(0)
    zs, preds, conv = [], [], []
    for _ in range(n_inits):
        z0 = torch.randn(1, x.shape[0], model.d_latent, device=x.device)
        z, info = model.solve(xb, z0=z0, **solve_kw)
        zs.append(z)
        conv.append(1.0 if info["converged"] else 0.0)
        preds.append(int(model.readout(model.pool(z)).argmax(-1)))
    fp_gap = 0.0
    for i in range(len(zs)):
        for j in range(i + 1, len(zs)):
            fp_gap = max(fp_gap, _relnorm(zs[i], zs[j]))
    mode = max(set(preds), key=preds.count)
    pred_agreement = preds.count(mode) / len(preds)
    return {"fp_gap": fp_gap, "pred_agreement": pred_agreement, "preds": preds,
            "converged_frac": sum(conv) / len(conv)}


@torch.no_grad()
def unlearning_gap(model, x, remove_idx, **solve_kw):
    xb = x.unsqueeze(0)
    z_full, _ = model.solve(xb, **solve_kw)

    keep = [i for i in range(x.shape[0]) if i != remove_idx]
    x_minus = xb[:, keep, :]
    warm_init = z_full[:, keep, :].clone()
    cold_init = torch.randn_like(warm_init)

    z_warm, info_w = model.solve(x_minus, z0=warm_init, **solve_kw)
    z_cold, info_c = model.solve(x_minus, z0=cold_init, **solve_kw)

    pred_warm = int(model.readout(model.pool(z_warm)).argmax(-1))
    pred_cold = int(model.readout(model.pool(z_cold)).argmax(-1))
    return {
        "unlearn_gap": _relnorm(z_warm, z_cold),
        "pred_match": pred_warm == pred_cold,
        "warm_iters": info_w["n_iter"],
        "cold_iters": info_c["n_iter"],
        "warm_converged": info_w["converged"],
        "cold_converged": info_c["converged"],
    }
