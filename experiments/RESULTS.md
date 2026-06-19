# Results log

## Layer 1, run 1 — are path-independence & exact unlearning free? (2026-06-19)

Task: GMM cluster-count classification, 2000 train / 500 test, N=24 points, d=2,
k in {1..4}. SetDEQ d_latent=32, 8 epochs, unrolled-solver gradient.
Probes: 20 test sets, solver max_iter=200, tol=1e-6, damping=0.5.

| Variant | Test acc | fp_gap | pred_agree | unlearn_gap | warm/cold iters |
|---|---|---|---|---|---|
| DeepSets (unconstrained) | 0.71 | 0.08 | 1.00 | 123,507 | 188 / 195 (capped) |
| DeepSets + spectral norm | 0.71 | 0.11 | 1.00 | 11,416,646 | 185 / 188 (capped) |
| Attention (LayerNorm)    | 0.83 | 0.033 | 0.98 | 0.029 | 88 / 116 (converged) |

### Takeaways
1. **Not free.** Unconstrained DeepSets update has no well-posed fixed point: the
   iteration never converges (hits the 200 cap), the latent diverges, and the
   unlearning gap is divergence noise, not a real measurement.
2. **Naive spectral norm is insufficient (worse here).** Bounding each weight's
   spectral radius to 1 does not make the update a contraction in z, because it
   consumes both z and pooled-z (Lipschitz ~2x) and ReLU/concat compound it.
3. **LayerNorm attention update gives the properties approximately for free.**
   Warm and cold solves both converge to nearly the same fixed point
   (fp_gap 0.033), unlearning is near-exact (gap 0.029, preds match 100%), and
   the warm-start efficiency bonus is real (~24% fewer iters). Mechanism is
   boundedness (LayerNorm), not spectral contractivity.
4. Attention's residual non-uniqueness (fp_gap 0.033, pred_agree 0.98) is exactly
   the regime where the Layer-2 MIA / DP-noise analysis becomes the right tool.

### Verification notes
- The huge DeepSets unlearn_gap is genuine divergence, not a metric bug: the
  attention run uses identical metric code and returns sane finite values, and
  the DeepSets solver iters sit at the 200 cap (non-convergence).
- Gradient is unrolled through the solver (Layer-1 simplification); swap in
  torchdeq phantom/implicit gradients before trusting train-time behavior at depth.

### Next
- Tighten convergence test (detect divergence explicitly; report % converged).
- Sweep damping and a proper Lipschitz bound for DeepSets to see if it CAN be
  made well-posed, or if normalization is necessary.
- Cardinality shift: train N=24, probe N=100, watch where attention's properties
  degrade.
