"""Jacobian probe — the spectral lens on well-posedness and leakage.

At a converged fixed point Z* of the update f(., x), two Jacobians matter:

  J_f = df/dZ |_{Z*}    -- the UPDATE Jacobian. Its spectral radius rho(J_f) is
                           the central well-posedness number:
                             rho < 1  -> local contraction -> unique, attracting
                                         fixed point -> path independence holds.
                             rho -> 1 -> approaching a BIFURCATION (the fixed
                                         point is about to split / destabilize).
                             rho > 1  -> expansive; the "fixed point" is a
                                         repeller, uniqueness is gone.

  dZ*/dw_k = (I - J_f)^{-1} df/dw_k    -- the input-output SENSITIVITY to the
                           presence weight of point k (the continuous relaxation
                           of removal, w_k: 1 -> 0). The amplifier (I - J_f)^{-1}
                           is SHARED with every other input perturbation and
                           blows up exactly as rho -> 1: that is why the same
                           spectral quantity governs well-posedness AND the
                           leakage tail an attacker would exploit.

Everything is matrix-free: we never form J_f, only Jacobian-vector products,
computed by the reverse-over-reverse ("double-vjp") trick so the machinery works
through attention too (forward-mode AD has gaps for some attention ops).
"""

import contextlib

import torch

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend

    def _math_attention():
        # The fused/flash SDPA CPU kernel has no double-backward, which the
        # double-vjp trick requires; the math backend does. Force it during probes.
        return sdpa_kernel([SDPBackend.MATH])
except Exception:  # older torch without torch.nn.attention
    def _math_attention():
        return contextlib.nullcontext()


def _jvp(fn, x, v):
    """Jacobian-vector product (dfn/dx) @ v, via two reverse-mode passes.

    g(u) = d(u . fn(x))/dx = J^T u, then d(v . g)/du = J v. Reverse-mode only, so
    it is robust through ops without a forward-mode rule (e.g. attention).
    """
    with torch.enable_grad(), _math_attention():
        x = x.detach().requires_grad_(True)
        y = fn(x)
        u = torch.zeros_like(y, requires_grad=True)
        (g,) = torch.autograd.grad(y, x, grad_outputs=u, create_graph=True)
        (jv,) = torch.autograd.grad(g, u, grad_outputs=v)
    return y.detach(), jv.detach()


def spectral_radius(model, x, z_star, n_iter=50, tail=8, seed=None):
    """Estimate rho(J_f) at z_star by power iteration on Jacobian-vector products.

    Returns (rho, history). rho is the median per-step growth factor over the
    last `tail` iterations; for a non-symmetric J_f this converges to the
    largest eigenvalue magnitude under generic conditions.
    """
    if seed is not None:
        torch.manual_seed(seed)
    f = lambda z: model.update(z, x)
    v = torch.randn_like(z_star)
    v = v / (v.norm() + 1e-12)
    history = []
    for _ in range(n_iter):
        _, jv = _jvp(f, z_star, v)
        growth = jv.norm().item()  # v is unit-norm, so this is ||J_f v|| / ||v||
        history.append(growth)
        if growth < 1e-20:
            break
        v = jv / jv.norm()
    last = history[-tail:] if len(history) >= tail else history
    rho = float(torch.tensor(last).median())
    return rho, history


def amplifier(rho):
    """Scalar IFT amplifier bound 1/(1-rho) = ||(I - J_f)^{-1}|| upper proxy.

    Diverges as rho -> 1: the sensitivity (and leakage) blow-up at the
    bifurcation. Clamped/inf for rho >= 1 (no contraction, ill-posed).
    """
    if rho >= 1.0:
        return float("inf")
    return 1.0 / (1.0 - rho)


def removal_sensitivity(model, x, z_star, remove_idx, n_iter=300, tol=1e-5,
                        diverge_norm=1e6):
    """Linearized effect on the equilibrium of removing point `remove_idx`.

    Computes ||dZ*/dw_k|| = ||(I - J_f)^{-1} df/dw_k|| matrix-free: the Neumann
    series delta = sum_m J_f^m b (b = df/dw_k) is summed by the iteration
    delta <- b + J_f delta, which CONVERGES iff rho(J_f) < 1 -- so the predicted
    sensitivity is finite exactly when the map is contractive and diverges at the
    bifurcation, self-consistently with `spectral_radius`.

    Returns dict: pred_rel (||dZ*/dw_k|| / ||Z*||), n_iter, converged, diverged.
    Only the mean-pool updates expose the presence weight; attention raises
    NotImplementedError upstream (caught by the caller).
    """
    B, N, _ = z_star.shape
    w0 = torch.ones(B, N, 1)
    e_k = torch.zeros_like(w0)
    e_k[:, remove_idx, :] = 1.0
    fw = lambda w: model.update(z_star, x, w)
    _, b = _jvp(fw, w0, e_k)  # df/dw_k at w = 1

    f = lambda z: model.update(z, x)
    delta = b.clone()
    converged = diverged = False
    it = 0
    for it in range(1, n_iter + 1):
        _, jd = _jvp(f, z_star, delta)  # J_f @ delta, evaluated at the fixed point
        new = b + jd
        if not torch.isfinite(new).all() or new.norm().item() > diverge_norm:
            diverged = True
            break
        if (new - delta).norm().item() / (new.norm().item() + 1e-12) < tol:
            delta = new
            converged = True
            break
        delta = new
    pred_rel = delta.norm().item() / (z_star.norm().item() + 1e-12)
    return {"pred_rel": pred_rel, "n_iter": it,
            "converged": converged, "diverged": diverged}


@torch.no_grad()
def jacobian_report(model, x, z_star, remove_idx=None):
    """Bundle the spectral lens for one fixed point: rho, amplifier, contractive
    flag, and (mean-pool models only) the linearized removal sensitivity.
    """
    rho, _ = spectral_radius(model, x, z_star)
    rep = {"rho": rho, "amplifier": amplifier(rho), "contractive": rho < 1.0}
    if remove_idx is not None:
        try:
            rep["removal"] = removal_sensitivity(model, x, z_star, remove_idx)
        except NotImplementedError:
            rep["removal"] = None  # attention: no presence-weight knob
    return rep
