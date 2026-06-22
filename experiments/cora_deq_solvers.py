"""Cora graph-DEQ on TorchDEQ: standardize the solver stack and benchmark accelerations.

Moves the smoke-test map onto TorchDEQ (ecosystem standard: implicit-diff backward,
pluggable solvers) and compares the production-common accelerations against plain
fixed-point iteration on THIS map:

    fixed_point_iter  -- Picard iteration z <- f(z)         (baseline)
    anderson          -- Anderson acceleration (history mixing)
    broyden           -- quasi-Newton root-find on g(z)=f(z)-z

NOTE (prior, memory): on the *set*-DEQ maps Anderson STAGNATED (~5e-3) while
fixed_point_iter/Broyden converged. That was a different map; the graph map is a
contraction (s*||W||<1), so this re-tests whether the accelerations help or hurt here.

Metric of merit = solver cost at a fixed tolerance: nstep (function evals) and wall time,
plus that all three land on the SAME fixed point (rel gap) and the SAME accuracy.

Run:  D:\\deq-venv\\Scripts\\python.exe -m experiments.cora_deq_solvers
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdeq import get_deq

from experiments.cora_smoke import download, load_cora, normalize_adj

F_TOL, F_MAX = 1e-6, 80


class GraphDEQ(nn.Module):
    """Contractive graph-DEQ; the fixed-point solve is delegated to TorchDEQ."""

    def __init__(self, d_in, d_lat, k, solver):
        super().__init__()
        self.enc = nn.Linear(d_in, d_lat)
        self.W = nn.Parameter(torch.randn(d_lat, d_lat) * 0.1)
        self.readout = nn.Linear(d_lat, k)
        self.s = 0.9
        # implicit-diff backward (TorchDEQ default); same solver fwd/bwd for fairness
        self.deq = get_deq(f_solver=solver, f_max_iter=F_MAX, f_tol=F_TOL,
                           b_solver=solver, b_max_iter=F_MAX, b_tol=F_TOL)

    def _Wc(self):
        return self.W / (torch.linalg.matrix_norm(self.W, ord=2) + 1e-6)

    def forward(self, Ahat, X):
        h0 = self.enc(X)
        Wc = self._Wc()

        def f(z):
            return torch.tanh(self.s * (Ahat @ (z @ Wc)) + h0)

        z0 = torch.zeros_like(h0)
        z_out, info = self.deq(f, z0)
        z = z_out[-1]
        return self.readout(z), z, info


def info_get(info, *keys):
    for k in keys:
        if k in info:
            v = info[k]
            return float(v.mean()) if torch.is_tensor(v) else float(np.mean(v))
    return float("nan")


def main():
    torch.manual_seed(0); np.random.seed(0)
    download()
    adj, X, labels_oh, idx_train, idx_test = load_cora()
    y = torch.tensor(labels_oh.argmax(1), dtype=torch.long)
    Ahat = normalize_adj(adj)
    X_t = torch.tensor(X)
    itr, ite = torch.tensor(idx_train), torch.tensor(idx_test)
    print(f"Cora loaded: {X.shape[0]} nodes, {labels_oh.shape[1]} classes "
          f"(tol={F_TOL}, max_iter={F_MAX})\n")

    # Train ONCE with the reliable solver (fwd+bwd), then compare solvers at INFERENCE
    # only -- this is the regime the deletion experiment lives in (frozen weights, forward
    # solve) and it decouples solver speed from training stability.
    torch.manual_seed(0); np.random.seed(0)
    trainer = GraphDEQ(X.shape[1], 64, labels_oh.shape[1], "fixed_point_iter")
    opt = torch.optim.Adam(trainer.parameters(), lr=1e-2, weight_decay=5e-4)
    for ep in range(120):
        opt.zero_grad()
        out, _, _ = trainer(Ahat, X_t)
        F.cross_entropy(out[itr], y[itr]).backward()
        opt.step()
    sd = trainer.state_dict()
    print("trained (fixed_point_iter). Now compare solvers at inference on identical weights:\n")

    ref_z = None
    print(f"{'solver':<18}{'test acc':>9}{'nstep':>8}{'rel res':>11}"
          f"{'solve ms':>11}{'fp gap vs FP':>14}{'  status'}")
    for solver in ["fixed_point_iter", "anderson", "broyden"]:
        m = GraphDEQ(X.shape[1], 64, labels_oh.shape[1], solver)
        m.load_state_dict(sd)
        m.eval()
        try:
            with torch.no_grad():
                t1 = time.time()
                out, z, info = m(Ahat, X_t)
                solve_ms = (time.time() - t1) * 1000
            if not torch.isfinite(z).all():
                raise FloatingPointError("non-finite fixed point")
            acc = (out.argmax(1)[ite] == y[ite]).float().mean().item()
            nstep = info_get(info, "nstep")
            relres = info_get(info, "rel_lowest", "rel_trace")
            if ref_z is None:
                ref_z, gap = z.clone(), 0.0
            else:
                gap = (z - ref_z).norm().item() / (ref_z.norm().item() + 1e-8)
            status = "ok" if relres < 10 * F_TOL else "STAGNATED"
            print(f"{solver:<18}{acc:>9.3f}{nstep:>8.1f}{relres:>11.1e}"
                  f"{solve_ms:>11.1f}{gap:>14.2e}  {status}")
        except Exception as e:
            print(f"{solver:<18}{'--':>9}{'--':>8}{'--':>11}{'--':>11}{'--':>14}  "
                  f"FAILED: {type(e).__name__}")

    print("\nfp gap vs FP = does the accelerated solver reach the SAME fixed point as "
          "Picard? (should be ~tol if it truly converged)")


if __name__ == "__main__":
    main()
