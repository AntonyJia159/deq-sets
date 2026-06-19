"""Fixed-point solver for the set-DEQ forward pass.

Layer 1 uses a damped fixed-point iteration: simple, robust, and works on
arbitrary tensor shapes (batched sets included). It keeps the unrolled graph so
autograd gives gradients directly -- fine for small N / few iterations. A later
layer will swap in torchdeq (Anderson + phantom gradients) for the real solver,
which is why `fixed_point_solve` returns a rich `info` dict matching that style.
"""

import torch


def fixed_point_solve(f, z0, max_iter=50, tol=1e-4, damping=0.5):
    """Solve z = f(z) by damped iteration: z <- (1-a) z + a f(z).

    Args:
        f: callable mapping z -> tensor of the same shape as z.
        z0: initial guess.
        max_iter: maximum iterations.
        tol: relative residual ||f(z) - z|| / ||z|| stopping threshold.
        damping: step size a in (0, 1]; <1 stabilizes non-contractive maps.

    Returns:
        (z_star, info) where info has 'n_iter', 'residuals', 'converged'.
    """
    z = z0
    residuals = []
    for k in range(1, max_iter + 1):
        fz = f(z)
        with torch.no_grad():
            res = (fz - z).norm().item() / (z.norm().item() + 1e-8)
        residuals.append(res)
        z = (1 - damping) * z + damping * fz
        if res < tol:
            return z, {"n_iter": k, "residuals": residuals, "converged": True}
    return z, {"n_iter": max_iter, "residuals": residuals, "converged": False}
