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

## Layer 1, run 2 — controlled normalization test (2026-06-19)

Added NormDeepSetsUpdate: identical wrapper to AttnUpdate (input injection +
residual + 2 LayerNorms + FF) but mean-pool aggregator instead of attention.

| Variant | Mixing | Norm | Test acc | fp_gap | pred_agree | unlearn_gap | warm/cold |
|---|---|---|---|---|---|---|---|
| DeepSets (raw)      | mean-pool | no  | 0.71 | 0.08 | 1.00 | 123,507    | 188/195 (capped) |
| DeepSets + spectral | mean-pool | no  | 0.71 | 0.11 | 1.00 | 11,416,646 | 185/188 (capped) |
| NormDeepSets        | mean-pool | yes | 0.75 | 0.039 | 1.00 | 0.025 | 130/152 |
| Attention           | attention | yes | 0.83 | 0.033 | 0.98 | 0.029 | 88/116 |

### Clean dissociation
- **Normalization -> well-posedness.** Swapping only the norm (raw -> NormDeepSets,
  same aggregator) flips divergence (gap 1e5) to convergence + near-exact
  unlearning (gap 0.025). Aggregator was never the cause of divergence.
- **Attention -> accuracy.** Holding norm fixed (NormDeepSets -> Attention), acc
  rises 0.75 -> 0.83. Aggregator models the interactive cluster dynamics.

### Hypothesis to test (not asserted)
NormDeepSets is *cleaner* on equilibrium properties (pred_agree 1.00 vs 0.98,
slightly smaller unlearn_gap) despite lower accuracy. Possible expressiveness <->
uniqueness tradeoff: attention's richer fixed points may be marginally less
unique / more leaky. Stress-test under cardinality shift and harder configs.

## Layer 1, run 3 — cardinality shift (2026-06-19)

Train at N=24, probe at N in {24,48,96,192} (k fixed at 1..4; only density grows).
Probes: 30 sets, 5 inits, solver max_iter=200 tol=1e-5. Random single-point removal.

NormDeepSets:
| N | acc | fp_gap mean/max | agree | conv | unlearn mean/max | warm/cold |
|---|---|---|---|---|---|---|
| 24  | 0.685 | 0.051/0.322 | 1.00 | 0.51 | 0.039/0.221 | 132/156 |
| 48  | 0.730 | 0.033/0.339 | 1.00 | 0.64 | 0.044/0.587 | 107/137 |
| 96  | 0.715 | 0.037/0.258 | 1.00 | 0.73 | 0.049/0.353 | 95/134 |
| 192 | 0.675 | 0.020/0.226 | 1.00 | 0.77 | 0.025/0.357 | 78/131 |

Attention:
| N | acc | fp_gap mean/max | agree | conv | unlearn mean/max | match | warm/cold |
|---|---|---|---|---|---|---|---|
| 24  | 0.790 | 0.027/0.317 | 0.99 | 0.71 | 0.052/0.479 | 1.00 | 95/125 |
| 48  | 0.815 | 0.034/0.404 | 1.00 | 0.66 | 0.029/0.395 | 1.00 | 97/131 |
| 96  | 0.840 | 0.039/0.323 | 1.00 | 0.69 | 0.043/0.290 | 1.00 | 89/132 |
| 192 | 0.860 | 0.032/0.289 | 1.00 | 0.64 | 0.020/0.121 | 0.97 | 73/143 |

### Findings vs predictions
1. **Efficiency widens (confirmed).** warm iters fall with N, cold flat -> warm-start
   advantage grows (NormDeepSets 24->53, attn 30->70).
2. **Path independence does NOT crack with cardinality (prediction refuted).** fp_gap
   flat/improving, pred_agree ~1.0 at all N. Lesson: bifurcations live on the
   cluster-SEPARATION axis (geometric ambiguity), not the density axis. Sharper
   hypothesis for the next experiment.
3. **Unlearning: mean small/shrinking, the TAIL is the story (confirmed two-level).**
   NormDeepSets keeps a latent tail (max ~0.3-0.6, ~10x mean); attention's tail shrinks
   in magnitude but surfaces as a rare DECISION FLIP (pred_match 0.97 at N=192 = one
   pivotal removal changed the answer).
4. **Well-posedness: boundedness survives, strict convergence does NOT certify.** No
   divergence at any N (guard never fired, fp_gap small), BUT only ~50-77% of solves reach
   tol=1e-5 in 200 iters. This is solver slowness (damping 0.5, contraction ~0.96 needs
   ~280 iters), NOT non-uniqueness. NormDeepSets conv RATE improves with N (mean-field
   stabilization). DO NOT claim 100% converged.
5. **Bonus:** attention accuracy IMPROVES with N (0.79->0.86) = real "infer large = more
   power"; NormDeepSets plateaus (~0.70, mean-pool saturates).

## Layer 1, run 3 — TorchDEQ + fixed_point_iter (authoritative) (2026-06-19)

Switched to TorchDEQ (implicit/phantom backward) and the fixed_point_iter solver.
**Anderson stagnates at ~5e-3 on these maps** (fixed_point_iter -> ~3e-7,
broyden -> ~7e-8), so Anderson is documented-but-unused. Probe solver reaches
~1e-7, so reported fixed points are tight; fp_gap ~0 now means EXACTLY unique
(earlier ~0.03 was damped-solver under-convergence noise).

### Smoke (4 variants), tight solver
| Variant | Mixing | Norm | acc | fp_gap | unlearn_gap | conv | warm/cold |
|---|---|---|---|---|---|---|---|
| deepsets (raw)      | mean-pool | no  | 0.27 | nan   | nan   | -    | 0/0 (NaN in training) |
| deepsets + spectral | mean-pool | SN  | 0.59 | 0.000 | 0.000 | 1.00 | 43/65 |
| normdeepsets        | mean-pool | LN  | 0.69 | 0.000 | 0.000 | 1.00 | 60/74 |
| attn                | attention | LN  | 0.81 | 0.091 | 0.079 | 0.92 | 53/74 |

### CORRECTIONS to run 1/2 (those used unrolled grad + damped solver)
1. **Spectral norm is NOT insufficient.** Under proper DEQ training it yields a
   well-posed, exactly-unlearnable model (acc 0.59). Run-2's "spectral diverges
   (gap 1e7)" was an artifact. Both spectral AND LayerNorm give EXACT
   path-independence + unlearning (gap 0.000) via contraction.
2. **Mean-pool fixed points are EXACTLY unique** (gap 0.000), not approx. The
   earlier 0.03 was under-convergence. This isolates attention's 0.09 as GENUINE
   non-uniqueness.
3. Raw (no norm) NaNs under the implicit gradient — cleaner evidence of ill-posedness.
4. Likely (verify): unrolled grad exploited a truncated transient state for higher
   apparent accuracy; implicit grad forces honest fixed-point use (lower, trustworthy).

### Revised dissociation
- Contractive updates (spectral OR LayerNorm mean-pool) -> EXACTLY unique fixed
  point -> exact path-independence + unlearning. Among these, LayerNorm gives
  better accuracy than spectral (0.69 vs 0.59).
- Expressive attention -> higher accuracy (0.81) but GENUINE non-uniqueness
  (gap 0.09, conv 0.92). The expressiveness <-> uniqueness tradeoff, quantified.

## Cardinality shift — train N=24, probe N in {24,48,96,192} (2026-06-19)

| Model | N | acc | fp_gap mean/max | agree | conv | unlearn mean/max | match | warm/cold |
|---|---|---|---|---|---|---|---|---|
| normdeepsets | 24  | 0.630 | 0.000/0.000 | 1.00 | 1.00 | 0.000/0.000 | 1.00 | 66/84 |
| normdeepsets | 48  | 0.705 | 0.006/0.167 | 1.00 | 1.00 | 0.006/0.169 | 1.00 | 53/79 |
| normdeepsets | 96  | 0.680 | 0.000/0.000 | 1.00 | 1.00 | 0.000/0.000 | 1.00 | 62/94 |
| normdeepsets | 192 | 0.640 | 0.007/0.114 | 1.00 | 0.99 | 0.004/0.097 | 1.00 | 70/112 |
| attn | 24  | 0.780 | 0.033/0.769 | 0.99 | 0.92 | 0.028/0.750 | 0.97 | 58/92 |
| attn | 48  | 0.835 | 0.114/0.757 | 0.99 | 0.89 | 0.080/0.532 | 1.00 | 65/86 |
| attn | 96  | 0.830 | 0.080/0.661 | 0.96 | 0.93 | 0.051/0.579 | 1.00 | 54/95 |
| attn | 192 | 0.835 | 0.090/0.543 | 0.99 | 0.96 | 0.074/0.516 | 1.00 | 94/120 |

### Predictions vs outcome
1. **Well-posedness survives scaling: CONFIRMED.** normdeepsets conv ~1.0 all N;
   attn conv 0.89-0.96, FLAT (doesn't degrade with N). Boundedness is cardinality-free.
2. **Path independence degrades with N: NOT confirmed.** Mean-pool stays EXACT
   across all N; attention's non-uniqueness is roughly N-INVARIANT (a property of the
   architecture, not of scale). The predicted "crack at scale" did not materialize.
3. **Unlearning mean shrinks / tail persists: tail CONFIRMED, mean-shrink NOT.**
   attn max unlearn_gap stays large (~0.5-0.75) at ALL N (the leakage tail / MIA
   target); mean does not monotonically shrink. mean-pool stays ~0 with rare small tail.
4. **Efficiency widens with N: CONFIRMED.** cold iters grow with N (84->112 norm,
   92->120 attn); warm grows slower -> advantage widens.

### Headline
Train-small/infer-large PRESERVES the equilibrium properties: both well-posedness
and the exact-vs-nonunique character carry from N=24 to N=192 unchanged. Attention
even improves accuracy with N (0.78->0.835) — a clean cardinality-generalization
result — while carrying a persistent worst-case leakage tail that is the natural
target for the Layer-2 membership-inference analysis.

### Next
- **Swap damped iteration for Anderson / TorchDEQ** — top priority; needed to certify
  convergence at scale before any of this becomes a figure.
- **Cluster-separation sweep** (the real bifurcation knob): vary inter-cluster distance,
  watch path independence / unlearning break as clusters merge. This is where the
  pivotal-removal tail and the MIA story should concentrate.
- Then the adversarial phase (MIA, DP-noise) in the bifurcation regime.

## Layer 3, run 1 — Jacobian probe (2026-06-20)

Built `src/jacobian.py` (matrix-free, double-vjp JVPs; math-SDPA backend forced so
attention's double-backward works) and `experiments/jacobian_probe.py`. Measures
rho(J_f) at trained fixed points (power iteration), the IFT amplifier 1/(1-rho),
and — for mean-pool models — the linearized removal sensitivity ||dZ*/dw_k|| via a
Neumann solve of (I-J_f)^{-1} df/dw_k. 30 probe sets at N=24.

| Model | rho mean/median/max | contractive | unlearn_gap mean/max | both-conv gap max (n) | corr(amp,gap) |
|---|---|---|---|---|---|
| normdeepsets | 0.793 / 0.766 / 0.929 | 100% | 0.000 / 0.000 | 0.000 (30/30) | 0.51* |
| attn         | 0.775 / 0.789 / 0.921 | 100% | 0.051 / 0.734 | **0.734 (29/30)** | -0.25 |

\* on ~1e-5 numerical noise (gap is exactly 0 to 3 dp) — vacuous.

### The finding — local contraction is necessary but NOT sufficient

The clean "rho -> 1 causes the leak" picture is REFUTED by our own probe, and the
real mechanism is sharper:

1. **Attention is LOCALLY contractive at every reached fixed point** (rho < 1, max
   0.92) — yet carries a large unlearning tail. So the leak is NOT a local
   bifurcation (no eigenvalue crosses the unit circle at the attractors we land on).
2. **The tail is TRUE MULTISTABILITY, not under-convergence.** Restricting to sets
   where BOTH warm and cold solves converged, the gap max is still 0.734 (29/30
   cases). Warm and cold land in DIFFERENT, each-locally-stable basins.
3. **Therefore the local Jacobian cannot certify the unlearning property.** A single
   attractor's spectral radius is blind to the existence of *other* coexisting
   attractors. corr(amplifier, gap) = -0.25 confirms the local amplifier does not
   predict the (global) leak.

### Refined diagnostic — a two-coordinate classifier

The clean separation needs BOTH numbers:
- **local rho(J_f)** (solver well-posedness / are reached points attracting?) AND
- **global fp_gap / both-converged gap** (is the attractor unique, or multistable?).

  mean-pool : rho < 1  AND  fp_gap = 0   -> single GLOBAL attractor (Banach-style).
  attention : rho < 1  BUT  fp_gap > 0   -> multiple LOCAL attractors (multistable).

So attention's non-uniqueness (Reports #1/#2) is **multistability / coexisting
attractors**, not local expansion. The reports' headline claims stand; the
*mechanism* is corrected — and the MIA target is basin multiplicity, diagnosed by
multi-init fp_gap, not by the local amplifier.

### Next (revised)
- **Continuation / homotopy**, not local Jacobian, to map basin structure: track
  fixed points as cluster separation varies; watch where a second branch is born
  (saddle-node) — that birth is the real bifurcation behind the multistability.
- The linearized removal sensitivity (mean-pool) is built and finite (Neumann
  converges since rho<1); it will become the redundant-vs-pivotal detector once we
  add a presence-weight knob to attention (or work on a contractive expressive model).
- MIA still targets attention's multistable tail — but indexed by fp_gap, not rho.

## Layer 3, run 2 — the decomposability pivot (linear attention) (2026-06-20)

Added LinAttnUpdate: linear attention (phi=elu+1), same wrapper as softmax attn,
but its aggregate S=sum_j phi(k_j)(x)v_j, zsum=sum_j phi(k_j) is an ADDITIVE
sufficient statistic — decomposable/federatable, and it DOES admit the presence
weight w (unlike softmax). The pivot: is exact unlearning governed by
DECOMPOSABILITY or by raw EXPRESSIVENESS? linattn is expressive-ish AND decomposable.

Robust metric = both-converged unlearn gap (conditioned on both solves reaching tol):

| model | decomp | acc | both-conv gap: 8ep / 15ep |
|---|---|---|---|
| normdeepsets | yes | 0.69-0.76 | 0.000 / 0.000 |
| linattn      | yes | 0.52-0.57 | 0.000(17/30) / **0.336(28/30)** |
| attn         | no  | 0.81-0.82 | 0.442 / 0.588 |

### Verdict: INCONCLUSIVE, leaning AGAINST the decomposability hypothesis

1. **linattn's first-run "uniqueness" was an artifact of under-convergence** (only
   17/30 converged). With 15 epochs it is 100% locally contractive AND 28/30
   converge — and then reveals GENUINE multistability (both-conv 0.336). So a
   *decomposable* model IS multistable. Decomposability did NOT buy global
   uniqueness. This flips the lean toward EXPRESSIVENESS, not decomposability:
   mean-pool may be unique because it is WEAK, and any added expressive capacity
   (linear or softmax) admits coexisting basins.
2. **But two confounds forbid a firm claim:**
   - linattn will not train (acc 0.52-0.57, BELOW mean-pool both runs). Linear
     attention underperforms softmax on tasks needing sharp selection; cluster-
     counting may be such a task. A model that does not learn the task is a poor
     test of "expressive AND unique" — its multistability may be a bad landscape.
   - single-seed variance is too high for this fine distinction: normdeepsets
     fp_gap_max swung 0.000->0.356 and rho_max 0.92->1.27 from an epoch change
     alone. Only the both-converged gap is stable across runs.

### What IS robust (holds both runs)
- mean-pool: globally unique when converged (both-conv 0.000). Genuinely weak-but-exact.
- softmax attn: genuinely multistable (both-conv 0.44-0.59). The Hopfield-predicted
  associative-memory regime.

### Next (to actually resolve the pivot)
- **Seed-average everything** (>=5 seeds): the single-seed noise is dominating the
  fine comparisons; only both-conv gap survived it.
- **Find a fairly-trained expressive-AND-contractive model** — linear attention is
  not it on this task (won't reach competitive acc). Candidates: L2/Lipschitz
  attention (Kim et al. 2021), monotone-operator design, or a harder-task where
  linear attention is competitive. Without a model that BOTH learns the task AND is
  decomposable/contractive, the decomposability-vs-expressiveness question stays open.

## Layer 3, run 3 — the Anil et al. (2022) PI recipe on attention (2026-06-20)

Tested whether path independence can be TRAINED into expressive (non-contractive)
attention via the Anil et al. recipe: mixed init (zeros on half the batch + noise)
and randomized solver budget (SetDEQ(pi_train=True), implemented in model.forward).
No contraction constraint. Single seed, N=24, 15 epochs.

| attn | acc | AA_mean | fp_gap_max | both-conv gap | both-conv n |
|---|---|---|---|---|---|
| baseline  | 0.824 | 0.997 | 0.571 | 0.500 | 30/30 |
| PI-recipe | 0.850 | 0.995 | 0.527 | **0.071** | 26/30 |

### Reading it
1. **Works, in the right direction, at no accuracy cost.** Worst-case genuine
   multistability dropped 7x (0.500 -> 0.071) and acc went UP (0.824 -> 0.850).
   Path independence IS trainable on expressive attention without contraction.
2. **It tamed the WARM-START channel, not global multistability.** fp_gap (random-
   init spread) barely moved (0.571 -> 0.527), but warm-vs-cold both-conv collapsed.
   Convergence counts explain it: under the recipe the multistable sets now FAIL TO
   CONVERGE (26/30) instead of silently settling in a wrong basin -- so a converged
   answer is basin-consistent, and ambiguity surfaces as detectable non-convergence
   (the safer failure mode). PI here is localized to the operating region, not global.
3. **AA score is the WRONG instrument for unlearning.** 0.997 vs 0.995 -- saturated,
   uninformative. Cosine alignment washes out the magnitude differences between
   basins that unlearning cares about. The both-converged relative-norm gap is the
   metric that discriminates; AA says baseline was "already PI" when it wasn't.

### Caveat / next
- SINGLE SEED, and variance is known-high here. 0.500 -> 0.071 is encouraging but
  unconfirmed. Seed-average (>=5) before trusting the magnitude.
- Then the MIA on the residual: does the (now-localized) warm-start channel still
  leak a deleted point, and does leakage track sensitivity vs basin?
- The recipe slightly hurt convergence reliability (30/30 -> 26/30; rho_max ~1.01).
